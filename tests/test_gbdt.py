import lightgbm as lgb
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.eval.metrics import log_loss, roc_auc
from sentinel.models.losses import focal_grad_hess, focal_loss, logloss_grad_hess
from sentinel.models.scratch.gbdt import MISSING, Binner, GBDTClassifier, build_tree
from sentinel.models.scratch.logistic import sigmoid


def _data(n=20_000, seed=0, missing=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    X[:, 3] = rng.integers(0, 5, n)  # low-cardinality feature
    z = 1.5 * X[:, 0] - 2 * (X[:, 1] > 0.5) + X[:, 2] * X[:, 0] + 0.8 * (X[:, 3] == 2) - 3
    y = (rng.random(n) < sigmoid(z)).astype(float)
    X[rng.random((n, 8)) < missing] = np.nan
    return X, y


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=2, max_size=300), st.integers(2, 30)
)
def test_binning_is_consistent_with_raw_thresholds(values, max_bins):
    x = np.array(values)[:, None]
    b = Binner(max_bins=max_bins).fit(x)
    codes = b.transform(x)[0]
    thr = b.thresholds[0]
    for j in range(len(thr)):
        np.testing.assert_array_equal(codes <= j, x[:, 0] <= thr[j])


def test_binned_and_raw_traversal_agree_including_missing():
    X, y = _data(n=5_000)
    m = GBDTClassifier(n_estimators=20, max_depth=5).fit(X, y)
    codes = m.binner_.transform(X)
    assert (codes == MISSING).any()
    for t in m.trees_:
        np.testing.assert_array_equal(t.apply(X), t.apply_binned(codes))


def test_leaf_values_equal_newton_step_on_their_rows():
    """Histogram subtraction must not corrupt the gradient sums of any node."""
    X, y = _data(n=3_000)
    b = Binner().fit(X)
    codes = b.transform(X)
    g, h = logloss_grad_hess(y, np.full(len(y), -2.0))
    lam, lr = 2.0, 0.3
    t = build_tree(
        codes, b.thresholds, g, h, np.arange(len(y)), np.arange(X.shape[1]),
        max_depth=4, learning_rate=lr, reg_lambda=lam, gamma=0.0, min_child_weight=1.0,
    )  # fmt: skip
    leaf = t.apply_binned(codes)
    for node in np.unique(leaf):
        rows = leaf == node
        assert t.value[node] == pytest.approx(-g[rows].sum() / (h[rows].sum() + lam) * lr)


def test_root_split_is_the_exhaustive_best():
    # <= 254 distinct values per feature, so bins sit at exact midpoints and the
    # histogram search must find the same split as an exhaustive search
    rng = np.random.default_rng(3)
    X = np.round(rng.normal(size=(400, 3)), 1)
    y = (X[:, 1] > 0.3).astype(float)
    b = Binner().fit(X)
    codes = b.transform(X)
    g, h = logloss_grad_hess(y, np.zeros(len(y)))
    t = build_tree(
        codes, b.thresholds, g, h, np.arange(400), np.arange(3),
        max_depth=1, learning_rate=1.0, reg_lambda=1.0, gamma=0.0, min_child_weight=0.0,
    )  # fmt: skip
    best, best_f = -np.inf, None
    for f in range(3):
        for thr in np.unique(X[:, f]):
            left = X[:, f] <= thr
            GL, HL, GR, HR = g[left].sum(), h[left].sum(), g[~left].sum(), h[~left].sum()
            gain = 0.5 * (GL**2 / (HL + 1) + GR**2 / (HR + 1) - g.sum() ** 2 / (h.sum() + 1))
            if gain > best:
                best, best_f = gain, f
    assert t.feature[0] == best_f == 1
    assert t.gain[0] == pytest.approx(best, rel=1e-9)


def test_learns_where_missing_values_should_go():
    rng = np.random.default_rng(0)
    n = 10_000
    X = rng.normal(size=(n, 2))
    y = (rng.random(n) < 0.05).astype(float)
    X[y == 1, 0] = np.where(
        rng.random(int(y.sum())) < 0.9, np.nan, X[y == 1, 0]
    )  # NaN signals fraud
    m = GBDTClassifier(n_estimators=30, max_depth=2).fit(X, y)
    p = m.predict_proba(X)
    assert p[np.isnan(X[:, 0])].mean() > 5 * p[~np.isnan(X[:, 0])].mean()


def test_contributions_sum_to_margin():
    X, y = _data(n=3_000)
    m = GBDTClassifier(n_estimators=15, max_depth=4).fit(X, y)
    contrib = m.predict_contributions(X[:500])
    np.testing.assert_allclose(contrib.sum(axis=1), m.decision_function(X[:500]), atol=1e-9)


def test_close_to_lightgbm_with_matching_hyperparameters():
    X, y = _data(n=30_000, seed=1)
    Xt, yt = _data(n=20_000, seed=2)
    params = dict(n_estimators=150, learning_rate=0.1, max_depth=5, reg_lambda=1.0)
    ours = GBDTClassifier(**params, min_child_weight=1.0).fit(X, y)
    ref = lgb.LGBMClassifier(
        **params, num_leaves=32, min_child_weight=1.0, min_child_samples=1, verbose=-1
    ).fit(X, y)
    p_ours, p_ref = ours.predict_proba(Xt), ref.predict_proba(Xt)[:, 1]
    assert roc_auc(yt, p_ours) == pytest.approx(roc_auc(yt, p_ref), abs=0.01)
    assert log_loss(yt, p_ours) == pytest.approx(log_loss(yt, p_ref), rel=0.03)


def test_early_stopping_truncates_to_best_iteration():
    X, y = _data(n=5_000)
    Xv, yv = _data(n=5_000, seed=9)
    m = GBDTClassifier(n_estimators=400, learning_rate=0.3, max_depth=6, early_stopping_rounds=10)
    m.fit(X, y, eval_set=(Xv, yv))
    assert len(m.trees_) < 400
    assert len(m.trees_) == int(np.argmin(m.eval_history_)) + 1


@pytest.mark.parametrize("gamma,alpha", [(0.0, 0.5), (1.0, 0.25), (2.0, 0.5), (3.0, 0.75)])
def test_focal_gradients_match_finite_differences(gamma, alpha):
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 200)
    y = (rng.random(200) < 0.3).astype(float)
    eps = 1e-5
    g, h = focal_grad_hess(y, z, gamma, alpha)
    g_num = (focal_loss(y, z + eps, gamma, alpha) - focal_loss(y, z - eps, gamma, alpha)) / (
        2 * eps
    )
    np.testing.assert_allclose(g, g_num, rtol=1e-5, atol=1e-8)
    g_up, _ = focal_grad_hess(y, z + eps, gamma, alpha)
    g_dn, _ = focal_grad_hess(y, z - eps, gamma, alpha)
    h_num = (g_up - g_dn) / (2 * eps)
    ok = h_num > 1e-5  # where the true Hessian is positive the floor is inactive
    np.testing.assert_allclose(h[ok], h_num[ok], rtol=1e-4, atol=1e-8)


def test_focal_with_gamma_zero_is_half_logloss():
    rng = np.random.default_rng(1)
    z, y = rng.normal(size=50), (rng.random(50) < 0.5).astype(float)
    g_f, h_f = focal_grad_hess(y, z, gamma=0.0, alpha=0.5)
    g_l, h_l = logloss_grad_hess(y, z)
    np.testing.assert_allclose(g_f, 0.5 * g_l, atol=1e-12)
    np.testing.assert_allclose(h_f, np.maximum(0.5 * h_l, 1e-6), atol=1e-12)
