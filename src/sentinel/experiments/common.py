"""Shared plumbing for experiments: splits, bootstrap metric tables, artifacts."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from sentinel.config import settings
from sentinel.data.dataset import Split, load_frame, split
from sentinel.eval import metrics as m
from sentinel.eval.bootstrap import ClusterBootstrap, Interval

ALERT_RATE = 0.005


@cache
def splits() -> dict[str, Split]:
    return split(load_frame())


def ranking_metrics(y: np.ndarray, p: np.ndarray, amount: np.ndarray):
    def fn(idx: np.ndarray) -> dict[str, float]:
        yy, pp, aa = y[idx], p[idx], amount[idx]
        return {
            "PR-AUC": m.average_precision(yy, pp),
            "ROC-AUC": m.roc_auc(yy, pp),
            "Recall@0.5%": m.recall_at_rate(yy, pp, ALERT_RATE),
            "$Recall@0.5%": m.recall_at_rate(yy, pp, ALERT_RATE, weight=aa),
            "Prec@Recall80": m.precision_at_recall(yy, pp, 0.8),
        }

    return fn


def metric_table(
    preds: dict[str, np.ndarray], test: Split, n_boot: int = 200
) -> tuple[pd.DataFrame, dict[str, dict[str, Interval]]]:
    boot = ClusterBootstrap(test.frame["card_id"].to_numpy(), n_boot=n_boot, seed=settings.seed)
    amount = test.frame["amount"].to_numpy()
    rows, intervals = [], {}
    for name, p in preds.items():
        iv = boot.intervals(ranking_metrics(test.y, p, amount), len(test.y))
        intervals[name] = iv
        rows.append(
            {"model": name}
            | {k: f"{v.point:.4f} ({v.low:.3f}-{v.high:.3f})" for k, v in iv.items()}
        )
    return pd.DataFrame(rows), intervals


def save_predictions(name: str, frame: pd.DataFrame, preds: dict[str, np.ndarray]) -> None:
    out = settings.paths.artifacts / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"txn_id": frame["txn_id"].to_numpy()} | preds)
    df.to_parquet(out / f"{name}.parquet", index=False)


def write_report(name: str, text: str) -> None:
    path = settings.paths.reports / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
