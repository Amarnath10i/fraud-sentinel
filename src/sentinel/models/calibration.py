"""Probability calibration, fitted on validation data only.

Ranking quality (PR-AUC) does not need calibrated scores, but cost-based
decisions do: "decline if p * amount > cost of a false decline" is only
correct if p really is the probability of fraud. Calibration matters most
after anything that distorts probabilities on purpose, e.g. class weights or
undersampling.

* `PlattScaler` - logistic regression on the logit of the score. Two
  parameters, so it is data-efficient, but it can only fix a sigmoid-shaped
  distortion.
* `IsotonicCalibrator` - the best monotone step function, fitted with the
  pool-adjacent-violators algorithm. Nonparametric, so it fixes any monotone
  distortion, but it needs more data.
* `prior_shift` - the closed-form correction for training on a resampled
  class balance.
"""

from __future__ import annotations

import numpy as np

from sentinel.models.scratch.logistic import LogisticRegressionIRLS, sigmoid

_EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattScaler:
    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattScaler:
        self.lr = LogisticRegressionIRLS(l2=1e-6).fit(_logit(p)[:, None], y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(p)[:, None])


def pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit to y (already ordered by x).

    Walk left to right keeping a stack of blocks (weighted mean, weight,
    length). Whenever the newest block's mean is below the previous one, the
    monotonicity constraint is violated and the two blocks are pooled into
    their weighted average. Each point is pushed once and merged at most
    once, so the whole pass is O(n).
    """
    means: list[float] = []
    weights: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        means.append(yi)
        weights.append(wi)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, s2 = means.pop(), weights.pop(), sizes.pop()
            m1, w1, s1 = means.pop(), weights.pop(), sizes.pop()
            wt = w1 + w2
            means.append((m1 * w1 + m2 * w2) / wt)
            weights.append(wt)
            sizes.append(s1 + s2)
    return np.repeat(means, sizes)


class IsotonicCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = (
            np.ones_like(p)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        # tied scores must receive one value: collapse them to their weighted mean first
        x, inv = np.unique(p, return_inverse=True)
        wsum = np.bincount(inv, weights=w)
        ymean = np.bincount(inv, weights=w * y) / wsum
        self.x_ = x
        self.y_ = pool_adjacent_violators(ymean, wsum)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        # linear between fitted points, constant beyond the ends
        return np.interp(np.asarray(p, dtype=np.float64), self.x_, self.y_)


def prior_shift(p: np.ndarray, train_rate: float, true_rate: float) -> np.ndarray:
    """Re-weight the odds when the model saw positives at `train_rate` but the
    world produces them at `true_rate` (e.g. after undersampling negatives):

        odds_true = odds_model * [true / (1 - true)] / [train / (1 - train)]
    """
    shift = _logit(np.array(true_rate)) - _logit(np.array(train_rate))
    return sigmoid(_logit(p) + shift)
