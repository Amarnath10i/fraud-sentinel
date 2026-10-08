"""Histogram-based gradient boosted decision trees, from scratch (NumPy only).

The algorithm follows XGBoost / LightGBM:

* **Second-order boosting.** Each tree fits a Newton step on the loss: for a
  node with gradient sum G and Hessian sum H the optimal output is
  -G / (H + lambda), and a split's gain is

      1/2 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ] - gamma.

  Any twice-differentiable loss plugs in through (gradient, Hessian), which
  is how focal loss is supported.
* **Histogram split finding.** Features are bucketed once into at most 254
  quantile bins (uint8). A node's split search is then two bincounts per
  feature plus a cumulative sum over bins, independent of the node size.
* **Histogram subtraction.** Only the smaller child's histogram is built;
  the larger child's is parent minus smaller, which roughly halves the work.
* **Sparsity-aware missing values.** Missing values get their own bin. At
  each split both directions are tried and the better one is stored as the
  node's default direction, so NaNs are routed by what the data says.
* **Flat-array trees.** A tree is a handful of parallel arrays, and
  prediction walks every row down the tree at once with vectorized indexing.
* **Saabas contributions.** Every node stores its Newton value, so the
  margin decomposes exactly into bias + per-feature contributions along the
  decision path (useful as fast reason codes).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sentinel.models.losses import logloss_grad_hess
from sentinel.models.scratch.logistic import sigmoid

MISSING = 255
MAX_VALUE_BINS = 255  # value bins 0..254, bin 255 = missing

GradHess = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Binner:
    """Per-feature thresholds; bin(x) = number of thresholds strictly below x.

    With that definition `bin(x) <= j` is equivalent to `x <= thresholds[j]`,
    so a split learned on bins can be applied to raw values directly.
    """

    def __init__(self, max_bins: int = MAX_VALUE_BINS, sample: int = 200_000, seed: int = 0):
        self.max_bins = min(max_bins, MAX_VALUE_BINS)
        self.sample = sample
        self.seed = seed

    def fit(self, X: np.ndarray) -> Binner:
        rng = np.random.default_rng(self.seed)
        self.thresholds: list[np.ndarray] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            v = col[~np.isnan(col)]
            if len(v) > self.sample:
                v = rng.choice(v, self.sample, replace=False)
            uniq = np.unique(v)
            if len(uniq) <= self.max_bins:
                thr = (uniq[:-1] + uniq[1:]) / 2  # midpoints: exact for low-cardinality features
            else:
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                thr = np.unique(np.quantile(v, qs, method="inverted_cdf"))
            self.thresholds.append(thr.astype(np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Binned codes in feature-major layout (n_features, n_rows), uint8."""
        out = np.empty((X.shape[1], X.shape[0]), dtype=np.uint8)
        for j, thr in enumerate(self.thresholds):
            col = X[:, j]
            codes = np.searchsorted(thr, col, side="left")
            codes[np.isnan(col)] = MISSING
            out[j] = codes
        return out


@dataclass
class Tree:
    feature: np.ndarray  # int32, -1 at leaves
    bin: np.ndarray  # int32 split bin: go left if code <= bin
    threshold: np.ndarray  # float64 raw threshold: go left if x <= threshold
    missing_left: np.ndarray  # bool default direction for NaN
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray  # leaf output; Newton value at internal nodes (both scaled by lr)
    gain: np.ndarray
    cover: np.ndarray  # Hessian sum per node
    depth: int

    def _walk(self, n: int, goes_left: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        node = np.zeros(n, dtype=np.int32)
        for _ in range(self.depth):
            f = self.feature[node]
            internal = f >= 0
            if not internal.any():
                break
            rows = np.flatnonzero(internal)
            nd = node[rows]
            left = goes_left(rows, nd)
            node[rows] = np.where(left, self.left[nd], self.right[nd])
        return node

    def apply(self, X: np.ndarray) -> np.ndarray:
        def goes_left(rows, nd):
            x = X[rows, self.feature[nd]]
            return np.where(np.isnan(x), self.missing_left[nd], x <= self.threshold[nd])

        return self._walk(X.shape[0], goes_left)

    def apply_binned(self, codes: np.ndarray) -> np.ndarray:
        def goes_left(rows, nd):
            c = codes[self.feature[nd], rows]
            return np.where(c == MISSING, self.missing_left[nd], c <= self.bin[nd])

        return self._walk(codes.shape[1], goes_left)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.value[self.apply(X)]


def _histograms(codes: np.ndarray, g: np.ndarray, h: np.ndarray, idx: np.ndarray, feats):
    n_feat = codes.shape[0]
    G = np.zeros((n_feat, 256))
    H = np.zeros((n_feat, 256))
    gi, hi = g[idx], h[idx]
    for j in feats:
        c = codes[j, idx]
        G[j] = np.bincount(c, weights=gi, minlength=256)
        H[j] = np.bincount(c, weights=hi, minlength=256)
    return G, H


def _best_split(G, H, g_tot, h_tot, feats_mask, lam, min_child_weight):
    """Vectorized over features x bins x missing direction."""
    Gc = np.cumsum(G[:, :MISSING], axis=1)
    Hc = np.cumsum(H[:, :MISSING], axis=1)
    Gm, Hm = G[:, MISSING : MISSING + 1], H[:, MISSING : MISSING + 1]
    parent = g_tot * g_tot / (h_tot + lam)
    best = (-np.inf, -1, -1, False)
    for miss_left in (False, True):
        GL = Gc + Gm if miss_left else Gc
        HL = Hc + Hm if miss_left else Hc
        GR, HR = g_tot - GL, h_tot - HL
        gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
        valid = (min_child_weight <= HL) & (min_child_weight <= HR) & feats_mask[:, None]
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        f, b = divmod(k, MISSING)
        if gain[f, b] > best[0]:
            best = (float(gain[f, b]), f, b, miss_left)
    return best


def build_tree(
    codes: np.ndarray,
    thresholds: list[np.ndarray],
    g: np.ndarray,
    h: np.ndarray,
    rows: np.ndarray,
    feats: np.ndarray,
    *,
    max_depth: int,
    learning_rate: float,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> Tree:
    n_feat = codes.shape[0]
    feats_mask = np.zeros(n_feat, dtype=bool)
    feats_mask[feats] = True
    lists: dict[str, list] = {k: [] for k in Tree.__dataclass_fields__ if k != "depth"}

    def new_node(g_tot: float, h_tot: float) -> int:
        lists["feature"].append(-1)
        lists["bin"].append(0)
        lists["threshold"].append(np.nan)
        lists["missing_left"].append(False)
        lists["left"].append(-1)
        lists["right"].append(-1)
        lists["value"].append(-g_tot / (h_tot + reg_lambda) * learning_rate)
        lists["gain"].append(0.0)
        lists["cover"].append(h_tot)
        return len(lists["feature"]) - 1

    G, H = _histograms(codes, g, h, rows, feats)
    j0 = int(feats[0])
    root = new_node(G[j0].sum(), H[j0].sum())
    level = [(root, rows, G, H)]
    depth = 0
    while level and depth < max_depth:
        nxt = []
        for node, idx, G, H in level:
            g_tot, h_tot = G[j0].sum(), H[j0].sum()
            gain, f, b, miss_left = _best_split(
                G, H, g_tot, h_tot, feats_mask, reg_lambda, min_child_weight
            )
            gain = 0.5 * gain - gamma
            if not np.isfinite(gain) or gain <= 0:
                continue
            c = codes[f, idx]
            go_left = (c <= b) | ((c == MISSING) & miss_left)
            li, ri = idx[go_left], idx[~go_left]
            small, large = (li, ri) if len(li) <= len(ri) else (ri, li)
            Gs, Hs = _histograms(codes, g, h, small, feats)
            Gl, Hl = G - Gs, np.maximum(H - Hs, 0.0)
            (GL, HL, GR, HR) = (Gs, Hs, Gl, Hl) if small is li else (Gl, Hl, Gs, Hs)

            lists["feature"][node] = f
            lists["bin"][node] = b
            thr = thresholds[f]
            lists["threshold"][node] = thr[b] if b < len(thr) else np.inf
            lists["missing_left"][node] = miss_left
            lists["gain"][node] = gain
            left = new_node(GL[j0].sum(), HL[j0].sum())
            right = new_node(GR[j0].sum(), HR[j0].sum())
            lists["left"][node], lists["right"][node] = left, right
            nxt += [(left, li, GL, HL), (right, ri, GR, HR)]
        level = nxt
        depth += 1

    return Tree(
        feature=np.array(lists["feature"], dtype=np.int32),
        bin=np.array(lists["bin"], dtype=np.int32),
        threshold=np.array(lists["threshold"], dtype=np.float64),
        missing_left=np.array(lists["missing_left"], dtype=bool),
        left=np.array(lists["left"], dtype=np.int32),
        right=np.array(lists["right"], dtype=np.int32),
        value=np.array(lists["value"], dtype=np.float64),
        gain=np.array(lists["gain"], dtype=np.float64),
        cover=np.array(lists["cover"], dtype=np.float64),
        depth=depth,
    )


class GBDTClassifier:
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample: float = 1.0,
        max_bins: int = MAX_VALUE_BINS,
        scale_pos_weight: float = 1.0,
        objective: GradHess | None = None,
        early_stopping_rounds: int | None = None,
        seed: int = 0,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.scale_pos_weight = scale_pos_weight
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.verbose = verbose

    def _grad_hess(self, y, z, w):
        if self.objective is not None:
            g, h = self.objective(y, z)
            return g * w, h * w
        return logloss_grad_hess(y, z, w)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: tuple | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.binner_ = Binner(self.max_bins, seed=self.seed).fit(X)
        codes = self.binner_.transform(X)
        w = np.where(y == 1, self.scale_pos_weight, 1.0)

        pos = np.sum(w * y)
        self.base_score_ = 0.0 if self.objective else math.log(pos / (np.sum(w) - pos))
        z = np.full(n, self.base_score_)
        self.trees_: list[Tree] = []
        self.eval_history_: list[float] = []
        if eval_set is not None:
            Xe = np.asarray(eval_set[0], dtype=np.float64)
            ye = np.asarray(eval_set[1], dtype=np.float64)
            ze = np.full(len(ye), self.base_score_)
        best, best_iter = np.inf, 0
        n_cols = max(1, round(self.colsample * d))

        for it in range(self.n_estimators):
            g, h = self._grad_hess(y, z, w)
            rows = (
                np.flatnonzero(rng.random(n) < self.subsample)
                if self.subsample < 1
                else np.arange(n)
            )
            feats = np.sort(rng.choice(d, n_cols, replace=False)) if n_cols < d else np.arange(d)
            tree = build_tree(
                codes, self.binner_.thresholds, g, h, rows, feats,
                max_depth=self.max_depth, learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda, gamma=self.gamma,
                min_child_weight=self.min_child_weight,
            )  # fmt: skip
            self.trees_.append(tree)
            z += tree.value[tree.apply_binned(codes)]

            if eval_set is not None:
                ze += tree.predict(Xe)
                pe = np.clip(sigmoid(ze), 1e-15, 1 - 1e-15)
                loss = float(-np.mean(ye * np.log(pe) + (1 - ye) * np.log(1 - pe)))
                self.eval_history_.append(loss)
                if loss < best - 1e-9:
                    best, best_iter = loss, it + 1
                elif (
                    self.early_stopping_rounds and it + 1 - best_iter >= self.early_stopping_rounds
                ):
                    break
                if self.verbose and (it + 1) % self.verbose == 0:
                    print(f"[{it + 1}] eval logloss {loss:.6f}")
        if eval_set is not None and self.early_stopping_rounds:
            self.trees_ = self.trees_[:best_iter]
        self.best_iteration_ = len(self.trees_)
        self.n_features_ = d
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        z = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            z += tree.predict(X)
        return z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Total split gain per feature, normalized to sum to 1."""
        imp = np.zeros(self.n_features_)
        for t in self.trees_:
            internal = t.feature >= 0
            np.add.at(imp, t.feature[internal], t.gain[internal])
        return imp / imp.sum() if imp.sum() > 0 else imp

    def predict_contributions(self, X: np.ndarray) -> np.ndarray:
        """Saabas decomposition: (n, n_features + 1), last column is the bias.

        Along each decision path, the change in node value at every split is
        credited to the split feature. Rows sum exactly to the raw margin.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros((n, self.n_features_ + 1))
        out[:, -1] = self.base_score_
        rows = np.arange(n)
        for t in self.trees_:
            node = np.zeros(n, dtype=np.int32)
            out[:, -1] += t.value[0]
            for _ in range(t.depth):
                f = t.feature[node]
                internal = f >= 0
                if not internal.any():
                    break
                r, nd = rows[internal], node[internal]
                x = X[r, f[internal]]
                left = np.where(np.isnan(x), t.missing_left[nd], x <= t.threshold[nd])
                child = np.where(left, t.left[nd], t.right[nd])
                np.add.at(out, (r, f[internal]), t.value[child] - t.value[nd])
                node[r] = child
        return out
