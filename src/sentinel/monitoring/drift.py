"""Label-free drift detection.

Fraud labels arrive weeks late, so a deployed model cannot be monitored by its
accuracy in real time. What can be watched immediately is whether the inputs
and the score distribution still look like the training data:

* PSI (population stability index) per feature and for the score, the
  standard credit-risk statistic: < 0.1 stable, 0.1-0.25 shifting, > 0.25
  significant;
* the two-sample Kolmogorov-Smirnov statistic (largest gap between the two
  empirical CDFs);
* adversarial validation: train a classifier to tell reference rows from
  current rows. AUC ~ 0.5 means the periods are indistinguishable; its most
  important features are the ones that drifted.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.eval.metrics import roc_auc

_EPS = 1e-4

# Fixed log-spaced bins for fraud scores. Calibrated scores are zero-inflated
# (most transactions get ~0), so reference-quantile bins collapse into one or
# two bins and PSI goes blind to exactly the tail where alerts live.
SCORE_EDGES = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.9])


def psi(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, edges: np.ndarray | None = None
) -> float:
    """PSI over reference-quantile bins (or explicit `edges`); NaNs form their own bin."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if edges is not None:
        edges = np.asarray(edges, dtype=np.float64)
    elif len(ref_ok):
        edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))
    else:
        edges = np.array([])

    def shares(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1)
        counts = np.append(counts, len(x) - len(ok)).astype(np.float64)
        return np.maximum(counts / max(len(x), 1), _EPS)

    e, a = shares(ref, ref_ok), shares(cur, cur_ok)
    return float(np.sum((a - e) * np.log(a / e)))


def alert_rate_ratio(reference: np.ndarray, current: np.ndarray, threshold: float = 0.01) -> float:
    """Share of current scores >= threshold divided by the reference share.

    For a rare-event score this is the drift signal that matters: halving the
    alert volume moves PSI by only ~0.01 (97% of the mass sits unchanged near 0
    and PSI weights bins by mass), but it halves this ratio.
    """
    ref_rate = float(np.mean(np.asarray(reference) >= threshold))
    cur_rate = float(np.mean(np.asarray(current) >= threshold))
    return cur_rate / ref_rate if ref_rate > 0 else float("inf")


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """sup_x |F_a(x) - F_b(x)| over the pooled sample points (NaNs ignored)."""
    a = np.sort(np.asarray(a, dtype=np.float64)[~np.isnan(a)])
    b = np.sort(np.asarray(b, dtype=np.float64)[~np.isnan(b)])
    if not len(a) or not len(b):
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = []
    for f in features:
        r, c = reference[f].to_numpy(np.float64), current[f].to_numpy(np.float64)
        rows.append({"feature": f, "psi": psi(r, c), "ks": ks_statistic(r, c)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    seed: int = 0,
    max_rows: int = 200_000,
) -> tuple[float, pd.Series]:
    """Two-fold cross-validated AUC of a reference-vs-current classifier."""
    rng = np.random.default_rng(seed)
    ref = reference.sample(min(len(reference), max_rows), random_state=seed)
    cur = current.sample(min(len(current), max_rows), random_state=seed)
    X = np.vstack([ref[features].to_numpy(np.float64), cur[features].to_numpy(np.float64)])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    fold = rng.integers(0, 2, len(y))
    pred = np.empty(len(y))
    gain = np.zeros(len(features))
    params = dict(
        objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, seed=seed, num_threads=8
    )
    for k in (0, 1):
        tr = fold != k
        booster = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 100)
        pred[~tr] = booster.predict(X[~tr])
        gain += booster.feature_importance("gain")
    importance = pd.Series(gain / gain.sum(), index=features).sort_values(ascending=False)
    return roc_auc(y, pred), importance
