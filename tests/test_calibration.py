import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from sentinel.eval.metrics import expected_calibration_error
from sentinel.models.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    pool_adjacent_violators,
    prior_shift,
)
from sentinel.models.scratch.logistic import sigmoid


def _miscalibrated(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.beta(0.3, 20, n)
    y = (rng.random(n) < true_p).astype(float)
    distorted = np.round(true_p**0.5, 3)  # monotone distortion with many ties
    return distorted, y


def test_pav_is_monotone_and_minimizes_squared_error():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200).cumsum() * 0.1 + rng.normal(size=200)
    fit = pool_adjacent_violators(y, np.ones_like(y))
    assert np.all(np.diff(fit) >= -1e-12)
    ref = IsotonicRegression().fit(np.arange(200), y).predict(np.arange(200))
    np.testing.assert_allclose(fit, ref, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_isotonic_matches_sklearn_with_ties_and_weights(seed):
    p, y = _miscalibrated(seed=seed)
    w = np.random.default_rng(seed).uniform(0.5, 2.0, len(p))
    ours = IsotonicCalibrator().fit(p, y, sample_weight=w)
    ref = IsotonicRegression(out_of_bounds="clip").fit(p, y, sample_weight=w)
    grid = np.linspace(-0.1, 1.1, 2_001)
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), atol=1e-12)


def test_calibrators_reduce_calibration_error():
    p, y = _miscalibrated()
    pv, yv = _miscalibrated(seed=7)
    before = expected_calibration_error(yv, pv)
    iso = expected_calibration_error(yv, IsotonicCalibrator().fit(p, y).predict(pv))
    platt = expected_calibration_error(yv, PlattScaler().fit(p, y).predict(pv))
    assert iso < before / 5
    assert platt < before / 2


def test_prior_shift_undoes_undersampling():
    rng = np.random.default_rng(0)
    n = 400_000
    x = rng.normal(size=n)
    y = (rng.random(n) < sigmoid(2 * x - 5)).astype(float)
    keep = (y == 1) | (rng.random(n) < 0.1)  # keep 10% of negatives
    # the Bayes posterior on the undersampled data has odds inflated by 10x
    p_sampled = sigmoid(2 * x - 5 + np.log(10))
    corrected = prior_shift(p_sampled, train_rate=y[keep].mean(), true_rate=y.mean())
    np.testing.assert_allclose(corrected, sigmoid(2 * x - 5), rtol=0.05)
