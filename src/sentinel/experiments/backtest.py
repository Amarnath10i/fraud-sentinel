"""Retraining policy backtest over 18 months of simulated deployment.

At the start of every month T from July 2019 to December 2020, each policy
decides which model scores that month. A model trained at time T may only use
transactions older than T - 60 days (matured labels), labelled with the
chargebacks reported before T. Each month is then scored against ground truth.

Policies
* static            - train once in July 2019, never retrain;
* monthly expanding - retrain every month on all matured history;
* monthly sliding   - retrain every month on the latest 6 matured months;
* drift-triggered   - retrain only when label-free monitoring of last month
                      vs the training reference fires: score PSI > 0.1,
                      top-feature PSI > 0.25, or the alert rate (share of
                      scores >= 1%) moving by more than 1.5x either way;
* GBDT + FTRL       - the static GBDT is frozen as a feature transform (leaf
                      indices, as in He et al., "Practical Lessons from
                      Predicting Clicks on Ads at Facebook", 2014) and an
                      FTRL-Proximal logistic layer on top keeps learning from
                      matured labels month by month, without retraining trees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.config import settings
from sentinel.data.dataset import CATEGORICAL, load_frame, model_features, observed_labels, window
from sentinel.eval import metrics as m
from sentinel.experiments.common import write_report
from sentinel.models.scratch.logistic import FTRLProximal
from sentinel.models.zoo import LGB_DEFAULTS
from sentinel.monitoring.drift import SCORE_EDGES, alert_rate_ratio, psi

log = logging.getLogger(__name__)

ROUNDS = 400  # fixed: no validation split inside each monthly window
SLIDING = timedelta(days=183)
PSI_SCORE, PSI_FEATURE, ALERT_RATIO = 0.1, 0.25, 1.5


@dataclass
class Trained:
    booster: lgb.Booster
    trained_at: pd.Timestamp
    reference_scores: np.ndarray
    reference: pd.DataFrame
    top_features: list[str] = field(default_factory=list)


def _months(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS", tz="UTC"))


def _train(frame: pd.DataFrame, at: pd.Timestamp, start: pd.Timestamp, feats: list[str]) -> Trained:
    tr = window(frame, start, at - settings.timeline.maturity)
    y = observed_labels(tr, at)
    ds = lgb.Dataset(
        tr[feats].to_numpy(np.float64), y, feature_name=feats, categorical_feature=CATEGORICAL
    )
    booster = lgb.train(LGB_DEFAULTS, ds, ROUNDS)
    # reference = most recent matured month, so drift is measured against "recent normal"
    ref = tr[tr["ts"] >= tr["ts"].max() - pd.Timedelta(days=30)]
    gain = pd.Series(booster.feature_importance("gain"), index=feats).sort_values(ascending=False)
    return Trained(
        booster, at, booster.predict(ref[feats].to_numpy(np.float64)), ref, list(gain.index[:5])
    )


def _drifted(model: Trained, recent: pd.DataFrame, feats: list[str]) -> tuple[bool, str]:
    scores = model.booster.predict(recent[feats].to_numpy(np.float64))
    s = psi(model.reference_scores, scores, edges=SCORE_EDGES)
    feature_psi = {
        c: psi(model.reference[c].to_numpy(), recent[c].to_numpy()) for c in model.top_features
    }
    worst = max(feature_psi, key=feature_psi.get)
    f = feature_psi[worst]
    r = alert_rate_ratio(model.reference_scores, scores)
    drifted = s > PSI_SCORE or f > PSI_FEATURE or not (1 / ALERT_RATIO < r < ALERT_RATIO)
    return (
        drifted,
        f"score PSI {s:.3f}, top-feature PSI {f:.3f} (`{worst}`), alert-rate ratio {r:.2f}",
    )


class LeafFTRL:
    """Frozen GBDT leaves -> one-hot -> FTRL-Proximal, updated as labels mature."""

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster
        dump = booster.dump_model()
        leaves = [t["num_leaves"] for t in dump["tree_info"]]
        self.offsets = np.r_[0, np.cumsum(leaves)[:-1]]
        self.ftrl = FTRLProximal(int(np.sum(leaves)), alpha=0.05, beta=1.0, l1=1.0, l2=1.0)

    def _active(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, pred_leaf=True).astype(np.int64) + self.offsets

    def learn(self, X: np.ndarray, y: np.ndarray) -> None:
        ones = np.ones(len(self.offsets))
        for idx, label in zip(self._active(X), y, strict=True):
            self.ftrl.update(idx, ones, float(label))

    def predict(self, X: np.ndarray) -> np.ndarray:
        w = self.ftrl.weights
        margin = w[self._active(X)].sum(axis=1) + w[-1]
        return 1 / (1 + np.exp(-margin))


def run() -> str:
    frame = load_frame()
    feats = model_features()
    tl = settings.timeline
    start = pd.Timestamp(tl.data_start)
    months = _months("2019-07-01", "2020-12-01")

    static = _train(frame, months[0], start, feats)
    hybrid = LeafFTRL(static.booster)
    warm = window(frame, start, months[0] - tl.maturity)
    hybrid.learn(warm[feats].to_numpy(np.float64), observed_labels(warm, months[0]))
    learned_until = months[0] - tl.maturity
    drift_model = static
    rows, events = [], []

    for i, T in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else pd.Timestamp(tl.data_end)
        test = window(frame, T, end)
        X, y = test[feats].to_numpy(np.float64), test["is_fraud"].to_numpy(np.int8)
        log.info("month %s: %d rows", T.date(), len(y))

        expanding = static if i == 0 else _train(frame, T, start, feats)
        sliding = static if i == 0 else _train(frame, T, T - tl.maturity - SLIDING, feats)
        if i > 0:
            recent = window(frame, months[i - 1], T)
            drifted, why = _drifted(drift_model, recent, feats)
            if drifted:
                drift_model = _train(frame, T, start, feats)
                events.append(f"{T:%Y-%m}: retrained ({why})")
            # the online layer learns from everything that matured since last month
            newly = window(frame, learned_until, T - tl.maturity)
            hybrid.learn(newly[feats].to_numpy(np.float64), observed_labels(newly, T))
            learned_until = T - tl.maturity

        preds = {
            "static": static.booster.predict(X),
            "monthly expanding": expanding.booster.predict(X),
            "monthly sliding 6m": sliding.booster.predict(X),
            "drift-triggered": drift_model.booster.predict(X),
            "static GBDT + online FTRL": hybrid.predict(X),
        }
        for name, p in preds.items():
            rows.append(
                {
                    "month": f"{T:%Y-%m}",
                    "policy": name,
                    "pr_auc": m.average_precision(y, p),
                    "recall": m.recall_at_rate(y, p, 0.005),
                    "fraud": int(y.sum()),
                }
            )

    res = pd.DataFrame(rows)
    res.to_csv(settings.paths.reports / "backtest_monthly.csv", index=False)
    pivot = res.pivot(index="month", columns="policy", values="pr_auc")
    order = [
        "static",
        "monthly expanding",
        "monthly sliding 6m",
        "drift-triggered",
        "static GBDT + online FTRL",
    ]
    pivot = pivot[order]
    summary = (
        res.groupby("policy")
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            worst_month_pr_auc=("pr_auc", "min"),
            mean_recall=("recall", "mean"),
        )
        .loc[order]
    )
    retrains = {"static": 0, "monthly expanding": len(months) - 1, "monthly sliding 6m": len(months) - 1,
                "drift-triggered": len(events), "static GBDT + online FTRL": 0}  # fmt: skip
    summary = summary.map(lambda v: f"{v:.4f}")
    summary.insert(0, "tree retrains", [str(retrains[p]) for p in order])

    text = "\n".join(
        [
            "# Retraining policy backtest (Jul 2019 - Dec 2020)",
            "",
            "Every month is scored by the model each policy would have had in production at "
            "the start of that month; training only ever uses matured labels (60 days) as "
            "reported by that date. LightGBM with fixed hyperparameters and "
            f"{ROUNDS} rounds for every retrain.",
            "",
            "## Summary",
            "",
            summary.to_markdown(),
            "",
            "## PR-AUC by month",
            "",
            pivot.to_markdown(floatfmt=".4f"),
            "",
            "## Drift-triggered retrains",
            "",
            *([f"- {e}" for e in events] or ["- none: monitoring never crossed the thresholds"]),
            "",
        ]
    )
    write_report("backtest", text)
    return text
