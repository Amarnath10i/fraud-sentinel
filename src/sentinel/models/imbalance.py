"""Resampling strategies for a 0.6%-positive training set.

* `undersample_negatives` keeps every fraud and a random share of legitimate
  transactions. Probabilities must then be corrected with
  `calibration.prior_shift`.
* `smote` (Chawla et al., 2002) creates synthetic frauds on the line between a
  fraud and one of its k nearest fraud neighbours. Two adjustments make it
  usable on this data: neighbours are searched on imputed, standardized
  features, but a synthetic row keeps its base row's missing values instead
  of inventing values; and discrete columns (category, flags) are copied from
  one parent instead of interpolated into impossible values like
  category 4.37.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def undersample_negatives(y: np.ndarray, keep_rate: float, seed: int = 0) -> np.ndarray:
    """Row indices: all positives plus `keep_rate` of the negatives."""
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < keep_rate)
    return np.flatnonzero(keep)


def smote(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float,
    k: int = 5,
    discrete: list[int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Append synthetic positives until positives / negatives == target_ratio."""
    rng = np.random.default_rng(seed)
    pos = X[y == 1]
    n_new = int(target_ratio * (y == 0).sum()) - len(pos)
    if n_new <= 0:
        return X, y

    med = np.nanmedian(X, axis=0)
    filled = np.where(np.isnan(pos), med, pos)
    mu, sd = filled.mean(axis=0), filled.std(axis=0)
    sd[sd == 0] = 1.0
    z = (filled - mu) / sd
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, neigh = nn.kneighbors(z)  # first neighbour is the point itself

    base = rng.integers(0, len(pos), n_new)
    mate = neigh[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1))
    a, b = pos[base], pos[mate]
    b_filled = np.where(np.isnan(b), a, b)  # a missing neighbour value contributes nothing
    synth = a + gap * (b_filled - a)  # NaN in the base row stays NaN
    for j in discrete or []:
        take_mate = gap[:, 0] >= 0.5
        synth[:, j] = np.where(take_mate & ~np.isnan(b[:, j]), b[:, j], a[:, j])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
