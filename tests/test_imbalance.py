import numpy as np

from sentinel.models.imbalance import smote, undersample_negatives


def test_undersampling_keeps_all_positives():
    rng = np.random.default_rng(0)
    y = (rng.random(100_000) < 0.01).astype(int)
    idx = undersample_negatives(y, 0.1)
    assert y[idx].sum() == y.sum()
    assert abs((y[idx] == 0).sum() - 0.1 * (y == 0).sum()) < 0.01 * (y == 0).sum()


def test_smote_reaches_ratio_and_stays_inside_the_minority_hull():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5_000, 3))
    X[:, 2] = rng.integers(0, 4, 5_000)
    y = (rng.random(5_000) < 0.02).astype(int)
    X[y == 1, 0] += 5
    X[rng.random(X.shape) < 0.05] = np.nan
    Xs, ys = smote(X, y, target_ratio=0.25, discrete=[2])
    new = Xs[len(X) :]
    assert abs(ys.sum() / (ys == 0).sum() - 0.25) < 0.01
    pos = X[y == 1]
    assert np.nanmin(new[:, 0]) >= np.nanmin(pos[:, 0]) - 1e-9
    assert np.nanmax(new[:, 0]) <= np.nanmax(pos[:, 0]) + 1e-9
    cats = new[:, 2][~np.isnan(new[:, 2])]
    assert set(np.unique(cats)) <= {0.0, 1.0, 2.0, 3.0}  # never interpolated
    assert np.isnan(new).any()  # missingness pattern of base rows is preserved
