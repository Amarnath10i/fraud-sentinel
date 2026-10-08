import numpy as np
import pytest

from sentinel.decision import (
    APPROVE,
    DECLINE,
    REVIEW,
    ConformalPolicy,
    CostModel,
    bayes_policy,
    budget_policy,
    capacity_policy,
    conformal_quantile,
)

COSTS = CostModel()


def test_bayes_policy_depends_on_stakes_not_just_probability():
    # same 5% probability: a $5 purchase is not worth an analyst, $5,000 is
    assert bayes_policy([0.05, 0.05], [5.0, 5_000.0], COSTS).tolist() == [APPROVE, REVIEW]
    # at p = 0.05 reviewing ($16 expected) beats declining ($104): a false
    # decline of a legitimate $5,000 purchase is expensive
    exp = COSTS.expected(np.array([0.05]), np.array([5_000.0]))[0]
    assert exp[REVIEW] < exp[DECLINE] < exp[APPROVE]
    # near-certain fraud on a large amount: decline outright
    assert bayes_policy([0.9], [5_000.0], COSTS).tolist() == [DECLINE]


def test_bayes_policy_is_optimal_when_probabilities_are_true():
    rng = np.random.default_rng(0)
    n = 200_000
    p = rng.beta(0.2, 10, n)
    amount = rng.lognormal(4, 1.2, n)
    y = (rng.random(n) < p).astype(float)
    bayes = COSTS.realized(bayes_policy(p, amount, COSTS), y, amount).sum()
    for other in (np.full(n, APPROVE), np.where(p > 0.5, DECLINE, APPROVE), budget_policy(p, 0.01)):
        assert bayes < COSTS.realized(other, y, amount).sum()


def test_capacity_policy_respects_daily_capacity_and_keeps_best_cases():
    rng = np.random.default_rng(1)
    n = 5_000
    p = rng.uniform(0.02, 0.2, n)
    amount = rng.uniform(100, 400, n)
    day = rng.integers(0, 5, n)
    unconstrained = bayes_policy(p, amount, COSTS)
    action = capacity_policy(p, amount, day, capacity=50, costs=COSTS)
    for d in range(5):
        assert (action[day == d] == REVIEW).sum() <= 50
    # whatever was kept must be at least as valuable as whatever was dropped
    exp = COSTS.expected(p, amount)
    saving = np.minimum(exp[:, 0], exp[:, 2]) - exp[:, 1]
    for d in range(5):
        kept = (day == d) & (action == REVIEW)
        dropped = (day == d) & (unconstrained == REVIEW) & (action != REVIEW)
        if kept.any() and dropped.any():
            assert saving[kept].min() >= saving[dropped].max()


def test_conformal_quantile_definition():
    s = np.arange(1, 20, dtype=float)  # n = 19
    assert conformal_quantile(s, 0.1) == 18.0  # ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(s, 0.01) == np.inf


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conformal_guarantee_holds_on_exchangeable_data(seed):
    rng = np.random.default_rng(seed)
    n = 100_000
    y = (rng.random(n) < 0.02).astype(int)
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 3 * y - 4)))
    cal, test = slice(0, 50_000), slice(50_000, None)
    pol = ConformalPolicy(alpha_fraud=0.05, alpha_legit=0.01).fit(p[cal], y[cal])
    act, yt = pol.decide(p[test]), y[test]
    assert np.mean(act[yt == 1] == APPROVE) <= 0.05 + 0.015
    assert np.mean(act[yt == 0] == DECLINE) <= 0.01 + 0.003
