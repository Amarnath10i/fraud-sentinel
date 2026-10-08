import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sentinel.monitoring.drift import adversarial_validation, alert_rate_ratio, ks_statistic, psi


def test_psi_is_near_zero_for_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=50_000)
    assert psi(ref, rng.normal(size=50_000)) < 0.01
    assert 0.1 < psi(ref, rng.normal(0.5, 1, 50_000)) < psi(ref, rng.normal(1.0, 1, 50_000))


def test_psi_notices_a_change_in_missingness():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20_000)
    cur = ref.copy()
    cur[rng.random(len(cur)) < 0.3] = np.nan
    assert psi(ref, cur) > 0.25


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.5])
def test_ks_matches_scipy(shift):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=3_000), rng.normal(shift, 1, 2_000)
    assert ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic, abs=1e-12)


def test_adversarial_validation_finds_the_drifting_feature():
    rng = np.random.default_rng(3)
    n = 20_000
    ref = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    same = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    moved = same.assign(b=same["b"] + 0.7)
    auc_same, _ = adversarial_validation(ref, same, ["a", "b", "c"])
    auc_moved, importance = adversarial_validation(ref, moved, ["a", "b", "c"])
    assert auc_same < 0.53
    assert auc_moved > 0.65
    assert importance.index[0] == "b"


def test_score_psi_needs_fixed_bins_on_zero_inflated_scores():
    from sentinel.monitoring.drift import SCORE_EDGES

    rng = np.random.default_rng(4)
    n = 200_000
    ref = np.where(rng.random(n) < 0.97, 0.0, rng.uniform(0.001, 1, n))
    cur = np.where(rng.random(n) < 0.985, 0.0, rng.uniform(0.001, 1, n))  # half the alert volume
    # quantile bins collapse onto the zeros and see nothing at all
    assert psi(ref, cur) == 0.0
    # fixed bins see it, but PSI stays tiny: it weights bins by mass, and 97% of
    # the mass did not move, so the usual 0.1 / 0.25 thresholds never fire
    assert 0.0 < psi(ref, cur, edges=SCORE_EDGES) < 0.1
    # the alert-rate ratio is the signal that tracks what changed
    assert alert_rate_ratio(ref, cur) == pytest.approx(0.5, abs=0.05)
