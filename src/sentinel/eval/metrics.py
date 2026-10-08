"""Ranking, operating-point and calibration metrics for rare-event detection.

With ~0.4% positives, ROC-AUC is dominated by the huge pool of easy negatives
and looks excellent for almost any model, so the headline metrics here are:

* PR-AUC (average precision): quality of the ranking among the alerts;
* recall / precision / $-recall at a fixed alert rate: what a fraud team with
  a fixed review capacity actually gets;
* Brier score, log loss and ECE: whether the probabilities can be trusted for
  cost-based decisions.

ROC-AUC and average precision are implemented from scratch and tested against
scikit-learn (tests/test_metrics.py).
"""

from __future__ import annotations

import math

import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    first = np.r_[True, sorted_x[1:] != sorted_x[:-1]]
    group = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x)]
    avg = (starts + ends + 1) / 2.0  # mean of 1-based ranks starts+1 .. ends
    ranks = np.empty(len(x))
    ranks[order] = avg[group]
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative (Mann-Whitney U)."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _average_ranks(np.asarray(s, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at every distinct score threshold, highest first."""
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    s_sorted = np.asarray(s)[order]
    y_sorted = np.asarray(y, dtype=np.float64)[order]
    last_of_tie = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[last_of_tie]
    precision = tps / (last_of_tie + 1)
    recall = tps / tps[-1] if tps[-1] > 0 else np.full_like(tps, np.nan)
    return precision, recall


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Step-wise area under the PR curve: sum over thresholds of dR * P."""
    if np.sum(y) == 0:
        return math.nan
    precision, recall = _pr_points(y, s)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def precision_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    precision, recall = _pr_points(y, s)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else 0.0


def _top(s: np.ndarray, rate: float) -> np.ndarray:
    k = max(1, math.ceil(rate * len(s)))
    return np.argpartition(-np.asarray(s), k - 1)[:k]


def recall_at_rate(y, s, rate: float, weight: np.ndarray | None = None) -> float:
    """Share of positives (or of positive $ if `weight` is the amount) caught when
    alerting on the top `rate` fraction of transactions."""
    y = np.asarray(y, dtype=np.float64)
    w = y if weight is None else y * np.asarray(weight, dtype=np.float64)
    total = w.sum()
    return float(w[_top(s, rate)].sum() / total) if total > 0 else math.nan


def precision_at_rate(y, s, rate: float) -> float:
    return float(np.mean(np.asarray(y)[_top(s, rate)]))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def log_loss(y, p, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - np.asarray(y)) * np.log1p(-p)))


def calibration_bins(y, p, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-mass bins of predicted probability: (mean predicted, observed rate, count).

    Equal-width bins would put almost every transaction in the first bin when
    the base rate is 0.4%, so bins hold equal numbers of predictions instead.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    mean_p = np.array([p[c].mean() for c in chunks if len(c)])
    rate = np.array([y[c].mean() for c in chunks if len(c)])
    counts = np.array([len(c) for c in chunks if len(c)])
    return mean_p, rate, counts


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    mean_p, rate, counts = calibration_bins(y, p, n_bins)
    return float(np.sum(counts * np.abs(mean_p - rate)) / counts.sum())


def summarize(y, s, amount, alert_rate: float = 0.005, probabilities: bool = True) -> dict:
    """The standard metric set reported for every model."""
    out = {
        "pr_auc": average_precision(y, s),
        "roc_auc": roc_auc(y, s),
        f"recall@{alert_rate:.1%}": recall_at_rate(y, s, alert_rate),
        f"precision@{alert_rate:.1%}": precision_at_rate(y, s, alert_rate),
        f"dollar_recall@{alert_rate:.1%}": recall_at_rate(y, s, alert_rate, weight=amount),
        "precision@recall80": precision_at_recall(y, s, 0.80),
    }
    if probabilities:
        out |= {
            "brier": brier(y, s),
            "log_loss": log_loss(y, s),
            "ece": expected_calibration_error(y, s),
        }
    return out
