"""From probabilities to actions: approve, review or decline.

A fraud score is only useful through the decision it drives, and the right
decision depends on money, not on a fixed threshold: missing a $1,200 fraud
costs far more than missing a $4 one, while a false decline costs roughly the
same for both. Given a calibrated probability p and the amount A, the expected
cost of each action is

    approve : p * A
    review  : c_review + (1 - p) * c_friction + p * (1 - catch) * A
    decline : (1 - p) * (c_decline + margin * A)

and the Bayes-optimal policy takes the cheapest. Two practical constraints
are layered on top:

* analysts can only review so many cases a day, so review candidates compete
  for capacity by how much reviewing saves (a `TopK` heap per day);
* `ConformalPolicy` gives a distribution-free guarantee instead: at most
  alpha_fraud of frauds are auto-approved, provided test data is exchangeable
  with the calibration data (drift breaks this, and the reports check it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sentinel.ds import TopK

APPROVE, REVIEW, DECLINE = 0, 1, 2
ACTION_NAMES = ("approve", "review", "decline")


@dataclass(frozen=True)
class CostModel:
    review_cost: float = 3.0  # analyst time per case
    review_catch_rate: float = 0.95  # share of reviewed frauds that are stopped
    review_friction: float = 1.0  # delay cost to a legitimate customer under review
    decline_fixed: float = 10.0  # goodwill lost on a false decline
    decline_margin: float = 0.02  # interchange revenue lost on a false decline, share of amount

    def expected(self, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """(n, 3) expected cost of approve / review / decline."""
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(amount, dtype=np.float64)
        approve = p * a
        review = (
            self.review_cost + (1 - p) * self.review_friction + p * (1 - self.review_catch_rate) * a
        )
        decline = (1 - p) * (self.decline_fixed + self.decline_margin * a)
        return np.column_stack([approve, review, decline])

    def realized(self, action: np.ndarray, y: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """Cost actually incurred per transaction given the true label."""
        y = np.asarray(y, dtype=np.float64)
        return self.expected(y, amount)[np.arange(len(y)), action]


def bayes_policy(p, amount, costs: CostModel) -> np.ndarray:
    return np.argmin(costs.expected(p, amount), axis=1)


def capacity_policy(p, amount, day: np.ndarray, capacity: int, costs: CostModel) -> np.ndarray:
    """Bayes policy with at most `capacity` reviews per day.

    Each review candidate's priority is how much reviewing saves over the best
    alternative. A per-day TopK heap keeps the most valuable cases; the rest
    fall back to approve or decline, whichever is cheaper.
    """
    exp = costs.expected(p, amount)
    action = np.argmin(exp, axis=1)
    fallback = np.where(exp[:, APPROVE] <= exp[:, DECLINE], APPROVE, DECLINE)
    saving = np.minimum(exp[:, APPROVE], exp[:, DECLINE]) - exp[:, REVIEW]
    for d in np.unique(day):
        cand = np.flatnonzero((day == d) & (action == REVIEW))
        if len(cand) <= capacity:
            continue
        top: TopK[int] = TopK(capacity)
        for i in cand.tolist():
            top.push(float(saving[i]), i)
        kept = {i for _, i in top.items()}
        dropped = [i for i in cand.tolist() if i not in kept]
        action[dropped] = fallback[dropped]
    return action


def threshold_policy(p, threshold: float) -> np.ndarray:
    return np.where(np.asarray(p) >= threshold, DECLINE, APPROVE)


def budget_policy(p, rate: float) -> np.ndarray:
    """Send the top `rate` share of scores to review, approve the rest."""
    p = np.asarray(p)
    k = max(1, math.ceil(rate * len(p)))
    action = np.full(len(p), APPROVE)
    action[np.argpartition(-p, k - 1)[:k]] = REVIEW
    return action


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


class ConformalPolicy:
    """Class-conditional (Mondrian) split conformal prediction sets.

    Nonconformity of label "fraud" is 1 - p, of label "legit" is p. Calibrating
    each label on its own examples gives per-class guarantees:

        P(auto-approve | fraud) <= alpha_fraud,  P(decline | legit) <= alpha_legit.

    Set {legit} -> approve, {fraud} -> decline, both or neither -> review.
    """

    def __init__(self, alpha_fraud: float = 0.05, alpha_legit: float = 0.002) -> None:
        self.alpha_fraud = alpha_fraud
        self.alpha_legit = alpha_legit

    def fit(self, p: np.ndarray, y: np.ndarray) -> ConformalPolicy:
        p, y = np.asarray(p), np.asarray(y)
        self.q_fraud = conformal_quantile(1 - p[y == 1], self.alpha_fraud)
        self.q_legit = conformal_quantile(p[y == 0], self.alpha_legit)
        return self

    def sets(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p)
        return (1 - p) <= self.q_fraud, p <= self.q_legit

    def decide(self, p: np.ndarray) -> np.ndarray:
        fraud_in, legit_in = self.sets(p)
        action = np.full(len(fraud_in), REVIEW)
        action[legit_in & ~fraud_in] = APPROVE
        action[fraud_in & ~legit_in] = DECLINE
        return action
