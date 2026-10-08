"""Logistic regression two ways.

`LogisticRegressionIRLS` is the batch solver: Newton-Raphson on the L2-penalized
log-likelihood. Each step solves

    (X^T W X + lambda I) delta = X^T (p - y) + lambda w,    W = diag(p (1 - p))

which is the same as iteratively reweighted least squares. With a few dozen
features the Hessian is tiny, so this converges in ~10 iterations to machine
precision, far fewer than gradient descent would need.

`FTRLProximal` is the online learner from McMahan et al., "Ad Click Prediction:
a View from the Trenches" (KDD 2013). It keeps per-coordinate learning rates,
updates one example at a time and produces exact zeros through its L1 term,
which is what you want when the model has to keep learning from a stream of
labels after deployment.
"""

from __future__ import annotations

import math

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # split by sign so exp never overflows
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionIRLS:
    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-10) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol

    def _objective(self, X1, y, sw, w) -> float:
        z = X1 @ w
        # log(1 + e^z) - y z, written stably
        nll = np.sum(sw * (np.logaddexp(0.0, z) - y * z))
        return nll + 0.5 * self.l2 * np.dot(w[1:], w[1:])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        X1 = np.hstack([np.ones((n, 1)), X])  # intercept first, not penalized
        penalty = np.full(d + 1, self.l2)
        penalty[0] = 0.0

        w = np.zeros(d + 1)
        prior = np.sum(sw * y) / np.sum(sw)
        w[0] = math.log(prior / (1 - prior))
        obj = self._objective(X1, y, sw, w)
        for it in range(1, self.max_iter + 1):
            p = sigmoid(X1 @ w)
            grad = X1.T @ (sw * (p - y)) + penalty * w
            hess = (X1 * (sw * p * (1 - p))[:, None]).T @ X1 + np.diag(penalty)
            hess[np.diag_indices_from(hess)] += 1e-12  # guard against an exactly singular Hessian
            step = np.linalg.solve(hess, grad)
            # backtracking keeps Newton safe far from the optimum
            t = 1.0
            while True:
                cand = w - t * step
                cand_obj = self._objective(X1, y, sw, cand)
                if cand_obj <= obj or t < 1e-8:
                    break
                t *= 0.5
            w, prev, obj = cand, obj, cand_obj
            self.n_iter_ = it
            if prev - obj <= self.tol * max(1.0, abs(obj)):
                break
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))


class FTRLProximal:
    """Per-coordinate FTRL-Proximal for logistic loss.

    Weights are not stored; they are derived lazily from the accumulated
    gradients (z) and squared gradients (n):

        w_i = 0                                                   if |z_i| <= l1
        w_i = -(z_i - sign(z_i) l1) / ((beta + sqrt(n_i)) / alpha + l2)   otherwise
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 0.05,
        beta: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
    ) -> None:
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.z = np.zeros(n_features + 1)  # last slot is the bias
        self.n = np.zeros(n_features + 1)

    def _weights(self, idx: np.ndarray) -> np.ndarray:
        z, n = self.z[idx], self.n[idx]
        w = -(z - np.sign(z) * self.l1) / ((self.beta + np.sqrt(n)) / self.alpha + self.l2)
        return np.where(np.abs(z) <= self.l1, 0.0, w)

    @property
    def weights(self) -> np.ndarray:
        return self._weights(np.arange(len(self.z)))

    def _margin(self, idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, float]:
        w = self._weights(idx)
        return w, float(np.dot(w, val))

    def update(self, idx: np.ndarray, val: np.ndarray, y: float, weight: float = 1.0) -> float:
        """One sparse example: active feature indices and their values (bias added
        automatically). Returns the prediction made *before* the update."""
        idx = np.append(idx, len(self.z) - 1)
        val = np.append(val, 1.0)
        w, margin = self._margin(idx, val)
        p = 1.0 / (1.0 + math.exp(-max(min(margin, 35.0), -35.0)))
        g = weight * (p - y) * val
        sigma = (np.sqrt(self.n[idx] + g * g) - np.sqrt(self.n[idx])) / self.alpha
        self.z[idx] += g - sigma * w
        self.n[idx] += g * g
        return p

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Stream dense rows through `update` in order."""
        X = np.asarray(X, dtype=np.float64)
        idx = np.arange(X.shape[1])
        sw = np.ones(len(y)) if sample_weight is None else sample_weight
        for i in range(len(y)):
            self.update(idx, X[i], float(y[i]), float(sw[i]))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        w = self.weights
        return np.asarray(X, dtype=np.float64) @ w[:-1] + w[-1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))
