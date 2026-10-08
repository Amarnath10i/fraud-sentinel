import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from sentinel.eval.metrics import roc_auc
from sentinel.models.scratch.logistic import FTRLProximal, LogisticRegressionIRLS, sigmoid


def _data(n=4_000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (rng.random(n) < sigmoid(X @ w - 2.0)).astype(float)
    return X, y


@pytest.mark.parametrize("l2", [0.1, 1.0, 10.0])
def test_irls_matches_sklearn(l2):
    X, y = _data()
    ours = LogisticRegressionIRLS(l2=l2).fit(X, y)
    ref = LogisticRegression(C=1 / l2, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_.ravel(), atol=1e-6)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-6)
    assert ours.n_iter_ < 20


def test_irls_sample_weights_equal_row_duplication():
    X, y = _data(n=500)
    w = np.where(y == 1, 3.0, 1.0)
    weighted = LogisticRegressionIRLS(l2=1.0).fit(X, y, sample_weight=w)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    duplicated = LogisticRegressionIRLS(l2=1.0).fit(X[rep], y[rep])
    np.testing.assert_allclose(weighted.coef_, duplicated.coef_, atol=1e-8)


def test_irls_survives_separable_data_thanks_to_l2():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    m = LogisticRegressionIRLS(l2=0.1).fit(X, y)
    assert np.isfinite(m.coef_).all() and m.coef_[0] > 0


def test_ftrl_learns_online_and_l1_produces_exact_zeros():
    rng = np.random.default_rng(1)
    n, d = 20_000, 20
    X = rng.normal(size=(n, d))
    true_w = np.zeros(d)
    true_w[:3] = [2.0, -1.5, 1.0]  # only three informative features
    y = (rng.random(n) < sigmoid(X @ true_w - 1.0)).astype(float)

    # FTRL's L1 acts on accumulated gradients, which grow with the number of
    # examples, so its strength is on a much larger scale than a batch penalty.
    model = FTRLProximal(d, alpha=0.1, l1=80.0, l2=1.0).partial_fit(X[:15_000], y[:15_000])
    batch = LogisticRegressionIRLS(l2=1.0).fit(X[:15_000], y[:15_000])
    auc_online = roc_auc(y[15_000:], model.predict_proba(X[15_000:]))
    auc_batch = roc_auc(y[15_000:], batch.predict_proba(X[15_000:]))
    assert auc_online > auc_batch - 0.005
    w = model.weights[:-1]
    assert (w[3:] == 0).sum() >= 14  # most noise features pruned to exactly zero
    np.testing.assert_allclose(w[:3], true_w[:3], atol=0.25)
