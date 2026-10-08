import numpy as np
import pytest
from sklearn import metrics as skm

from sentinel.eval import metrics as m
from sentinel.eval.bootstrap import ClusterBootstrap


@pytest.fixture(params=[0, 1, 2])
def scored(request):
    rng = np.random.default_rng(request.param)
    n = 5_000
    y = (rng.random(n) < 0.02).astype(int)
    s = np.round(rng.normal(y * 1.5, 1.0), 1)  # rounding creates many ties
    return y, s


def test_roc_auc_matches_sklearn(scored):
    y, s = scored
    assert m.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)


def test_average_precision_matches_sklearn(scored):
    y, s = scored
    assert m.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)


def test_operating_point_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    amount = np.array([100, 1, 300, 1, 1, 1, 1, 1, 1, 600])
    assert m.recall_at_rate(y, s, 0.3) == pytest.approx(2 / 3)
    assert m.precision_at_rate(y, s, 0.3) == pytest.approx(2 / 3)
    assert m.recall_at_rate(y, s, 0.3, weight=amount) == pytest.approx(400 / 1000)
    assert m.precision_at_recall(y, s, 0.6) == pytest.approx(2 / 3)


def test_calibration_metrics_reward_honest_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert m.expected_calibration_error(y, p) < 0.003
    assert m.expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.05
    assert m.brier(y, p) < m.brier(y, np.clip(p * 2, 0, 1))
    assert m.log_loss(y, p) == pytest.approx(skm.log_loss(y, p), rel=1e-9)


def test_cluster_bootstrap_interval_contains_point_and_widens_with_clustering():
    rng = np.random.default_rng(0)
    n_cards, per_card = 200, 50
    card_effect = rng.normal(0, 1, n_cards)
    clusters = np.repeat(np.arange(n_cards), per_card)
    x = card_effect[clusters] + rng.normal(0, 0.1, len(clusters))

    def mean(idx):
        return {"mean": float(x[idx].mean())}

    clustered = ClusterBootstrap(clusters, n_boot=300).intervals(mean, len(x))["mean"]
    naive = ClusterBootstrap(np.arange(len(x)), n_boot=300).intervals(mean, len(x))["mean"]
    assert clustered.low <= clustered.point <= clustered.high
    assert (clustered.high - clustered.low) > 3 * (naive.high - naive.low)
