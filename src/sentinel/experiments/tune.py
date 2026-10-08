"""LightGBM hyperparameter search with rolling-origin time-series CV.

Random K-fold would let the model train on next month to predict last month,
which inflates scores and picks hyperparameters that do not survive deployment.
Here every fold trains on the past and validates on the following months:

    fold 1: train Jan-Jun 2019   -> validate Jul-Aug 2019
    fold 2: train Jan-Aug 2019   -> validate Sep-Oct 2019
    fold 3: train Jan-Oct 2019   -> validate Nov 2019-Jan 2020

Only data before the main validation window is used, so the main validation
and test periods stay untouched. Optuna's TPE sampler proposes parameters and
a median pruner stops trials that are clearly worse after their first folds.
"""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sentinel.config import settings
from sentinel.data.dataset import CATEGORICAL, model_features, observed_labels, window
from sentinel.eval.metrics import average_precision
from sentinel.experiments.common import metric_table, splits, write_report
from sentinel.models.zoo import LGB_DEFAULTS, LightGBM

log = logging.getLogger(__name__)

FOLDS = [
    ("2019-01-01", "2019-07-01", "2019-09-01"),
    ("2019-01-01", "2019-09-01", "2019-11-01"),
    ("2019-01-01", "2019-11-01", "2020-02-01"),
]


def _folds(frame: pd.DataFrame, feats: list[str]):
    as_of = settings.timeline.deploy_at
    out = []
    for a, b, c in FOLDS:
        tr = window(frame, pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
        va = window(frame, pd.Timestamp(b, tz="UTC"), pd.Timestamp(c, tz="UTC"))
        dtr = lgb.Dataset(
            tr[feats].to_numpy(np.float64), observed_labels(tr, as_of),
            feature_name=feats, categorical_feature=CATEGORICAL, free_raw_data=False,
        )  # fmt: skip
        out.append((dtr, va[feats].to_numpy(np.float64), observed_labels(va, as_of)))
    return out


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
    }


def run(n_trials: int = 20) -> str:
    s = splits()
    frame = pd.concat([s["train"].frame, s["valid"].frame])
    feats = model_features()
    folds = _folds(frame, feats)

    def objective(trial: optuna.Trial) -> float:
        params = LGB_DEFAULTS | search_space(trial) | {"metric": "average_precision"}
        scores, rounds = [], []
        for k, (dtr, Xv, yv) in enumerate(folds):
            dva = dtr.create_valid(Xv, yv)
            booster = lgb.train(
                params, dtr, 2000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )  # fmt: skip
            scores.append(
                average_precision(yv, booster.predict(Xv, num_iteration=booster.best_iteration))
            )
            rounds.append(booster.best_iteration)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_pr_auc", scores)
        trial.set_user_attr("rounds", rounds)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    # trial 0 = the hand-picked defaults, so tuning has to beat them to win
    study.enqueue_trial({
        "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "lambda_l2": 1.0, "min_gain_to_split": 0.0,
    })  # fmt: skip
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    out = settings.paths.lightgbm_params
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best.params, indent=2) + "\n")

    tr, va, te = s["train"], s["valid"], s["test"]
    default = LightGBM().fit(tr.frame, tr.y, va.frame, va.y)
    tuned = LightGBM(params=best.params).fit(tr.frame, tr.y, va.frame, va.y)
    table, _ = metric_table(
        {
            "LightGBM defaults": default.predict_proba(te.frame),
            "LightGBM tuned": tuned.predict_proba(te.frame),
        },
        te,
    )

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials = trials.sort_values("value", ascending=False).head(10)
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    text = "\n".join(
        [
            "# Hyperparameter search (Optuna TPE, rolling-origin CV)",
            "",
            f"{len(study.trials)} trials: {complete} complete, {pruned} pruned. "
            "Trial 0 is the hand-picked default configuration.",
            "",
            f"Best CV PR-AUC {best.value:.4f} (folds: "
            + ", ".join(f"{v:.4f}" for v in best.user_attrs["fold_pr_auc"])
            + f"; default config: {study.trials[0].value:.4f}).",
            "",
            "```json",
            json.dumps(best.params, indent=2),
            "```",
            "",
            "## Defaults vs tuned on the untouched test period",
            "",
            table.to_markdown(index=False),
            "",
            "## Top trials",
            "",
            trials.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    )
    write_report("tuning", text)
    return text
