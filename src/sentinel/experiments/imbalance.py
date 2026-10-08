"""Does fixing the class imbalance help?

Same LightGBM, same features, six ways of handling 0.6% positives. Each is
judged on ranking (PR-AUC, recall at a 0.5% alert rate) and on calibration,
before and after isotonic recalibration on the validation period.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sentinel.data.dataset import CATEGORICAL, model_features
from sentinel.eval import metrics as m
from sentinel.experiments.common import metric_table, save_predictions, splits, write_report
from sentinel.models.calibration import IsotonicCalibrator, prior_shift
from sentinel.models.imbalance import smote, undersample_negatives
from sentinel.models.zoo import LightGBM

log = logging.getLogger(__name__)


def run() -> str:
    s = splits()
    tr, va, te = s["train"], s["valid"], s["test"]
    feats = model_features()
    neg_pos = float((tr.y == 0).sum() / tr.y.sum())
    rate = float(tr.y.mean())

    def fit(model: LightGBM, frame=None, y=None) -> LightGBM:
        return model.fit(
            tr.frame if frame is None else frame, tr.y if y is None else y, va.frame, va.y
        )

    runs: dict[str, tuple[LightGBM, callable]] = {}
    identity = lambda p: p  # noqa: E731

    runs["none"] = (fit(LightGBM()), identity)
    runs[f"class weight sqrt ({np.sqrt(neg_pos):.0f}x)"] = (
        fit(LightGBM(params={"scale_pos_weight": np.sqrt(neg_pos)})),
        identity,
    )
    runs[f"class weight balanced ({neg_pos:.0f}x)"] = (
        fit(LightGBM(params={"scale_pos_weight": neg_pos})),
        identity,
    )

    idx = undersample_negatives(tr.y, keep_rate=0.1, seed=0)
    under = fit(LightGBM(), tr.frame.iloc[idx], tr.y[idx])
    sampled_rate = float(tr.y[idx].mean())
    runs["undersample negatives to 10%"] = (under, identity)
    runs["undersample + prior-shift correction"] = (
        under,
        lambda p: prior_shift(p, train_rate=sampled_rate, true_rate=rate),
    )

    X = tr.frame[feats].to_numpy(np.float64)
    Xs, ys = smote(X, tr.y, target_ratio=0.05, discrete=[feats.index(c) for c in CATEGORICAL])
    runs["SMOTE to 1:20"] = (fit(LightGBM(), pd.DataFrame(Xs, columns=feats), ys), identity)
    runs["focal loss (gamma=2, alpha=0.25)"] = (fit(LightGBM(focal=(2.0, 0.25))), identity)

    preds, rows = {}, []
    for name, (model, post) in runs.items():
        log.info("evaluating %s", name)
        p_te, p_va = post(model.predict_proba(te.frame)), post(model.predict_proba(va.frame))
        preds[name] = p_te
        iso = IsotonicCalibrator().fit(p_va, va.y).predict(p_te)
        legit = te.y == 0
        rows.append(
            {
                "strategy": name,
                "fit (s)": f"{model.fit_seconds:.0f}",
                "rounds": model.booster.best_iteration,
                "median p, legit": f"{np.median(p_te[legit]):.2e}",
                "mean p, legit": f"{p_te[legit].mean():.2e}",
                "log loss raw": f"{m.log_loss(te.y, p_te):.5f}",
                "log loss after isotonic": f"{m.log_loss(te.y, iso):.5f}",
                "ECE raw": f"{m.expected_calibration_error(te.y, p_te):.4f}",
                "ECE after isotonic": f"{m.expected_calibration_error(te.y, iso):.4f}",
            }
        )
    save_predictions("imbalance_test", te.frame, preds)
    ranking, _ = metric_table(preds, te)
    calib = pd.DataFrame(rows)

    text = "\n".join(
        [
            "# Class imbalance strategies (LightGBM, test period)",
            "",
            f"Training data: {len(tr.y):,} rows, {tr.y.sum():,} fraud ({rate:.2%}).",
            "",
            "## Ranking (95% card-level bootstrap intervals)",
            "",
            ranking.rename(columns={"model": "strategy"}).to_markdown(index=False),
            "",
            "## Calibration",
            "",
            "The classes are almost separable here, so nearly every prediction sits close to "
            "0 or 1 and ECE/Brier barely move between strategies. Reweighting and resampling "
            "show up where they should: in the probabilities given to legitimate transactions "
            "(true value ~0) and in log loss, which punishes confident mistakes.",
            "",
            calib.to_markdown(index=False),
            "",
        ]
    )
    write_report("imbalance", text)
    return text
