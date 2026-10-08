"""Gradients and Hessians w.r.t. the raw margin z (p = sigmoid(z)).

Shared by the from-scratch GBDT and by LightGBM through its custom-objective
hook, so both learners optimize exactly the same function.
"""

from __future__ import annotations

import numpy as np

from sentinel.models.scratch.logistic import sigmoid

HESS_FLOOR = 1e-6


def logloss_grad_hess(y: np.ndarray, z: np.ndarray, w: np.ndarray | None = None):
    p = sigmoid(z)
    g, h = p - y, p * (1 - p)
    if w is not None:
        g, h = g * w, h * w
    return g, np.maximum(h, HESS_FLOOR)


def focal_grad_hess(y: np.ndarray, z: np.ndarray, gamma: float = 2.0, alpha: float = 0.5):
    """Focal loss (Lin et al., 2017): -a_t (1 - p_t)^gamma log(p_t).

    With u = p_t and s = 2y - 1 (so du/dz = s u (1 - u)):

        dL/dz   = a_t s (1-u)^gamma (gamma u log u - (1 - u))
        d2L/dz2 = a_t u (1-u) [ (1-u)^gamma (gamma (log u + 1) + 1)
                                - gamma (1-u)^(gamma-1) (gamma u log u - (1 - u)) ]

    gamma = 0 and alpha = 0.5 recover (half) the log loss. The Hessian is
    negative for confidently wrong examples, so it is floored to keep Newton
    leaf values well defined, as XGBoost/LightGBM users do in practice.
    """
    p = sigmoid(z)
    s = 2.0 * y - 1.0
    u = np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1 - 1e-12)
    a = np.where(y == 1, alpha, 1 - alpha)
    log_u = np.log(u)
    one_m = 1.0 - u
    inner = gamma * u * log_u - one_m
    g = a * s * one_m**gamma * inner
    h = (
        a
        * u
        * one_m
        * (one_m**gamma * (gamma * (log_u + 1) + 1) - gamma * one_m ** (gamma - 1) * inner)
    )
    return g, np.maximum(h, HESS_FLOOR)


def focal_loss(y: np.ndarray, z: np.ndarray, gamma: float = 2.0, alpha: float = 0.5) -> np.ndarray:
    p = sigmoid(z)
    u = np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1.0)
    a = np.where(y == 1, alpha, 1 - alpha)
    return -a * (1 - u) ** gamma * np.log(u)
