"""Preprocessing for linear models (tree models consume raw features).

Fitted on training data only, then frozen:
* median imputation plus a missing-indicator column wherever training data
  had NaNs (missingness itself is often informative, e.g. "first transaction
  on this card");
* log1p for non-negative heavy-tailed columns (amounts, counts, seconds);
* standardization, clipped to +-10 so a single extreme value cannot dominate;
* one-hot encoding for categoricals (unseen categories map to all zeros).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LinearPreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str], clip: float = 10.0) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.clip = clip

    def fit(self, df: pd.DataFrame) -> LinearPreprocessor:
        self.params_: dict[str, tuple[bool, bool, float, float, float]] = {}
        for c in self.numeric:
            x = df[c].to_numpy(np.float64)
            nan = np.isnan(x)
            v = x[~nan]
            log = bool(v.min() >= 0 and np.quantile(v, 0.99) >= 50)
            median = float(np.median(v))
            t = np.log1p(np.where(nan, median, x)) if log else np.where(nan, median, x)
            std = float(t.std()) or 1.0
            self.params_[c] = (bool(nan.any()), log, median, float(t.mean()), std)
        self.levels_ = {c: np.sort(df[c].dropna().unique()) for c in self.categorical}
        return self

    @property
    def feature_names(self) -> list[str]:
        names = []
        for c, (has_nan, *_rest) in self.params_.items():
            names.append(c)
            if has_nan:
                names.append(f"{c}__missing")
        for c, levels in self.levels_.items():
            names += [f"{c}={v:g}" if isinstance(v, float) else f"{c}={v}" for v in levels]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c, (has_nan, log, median, mean, std) in self.params_.items():
            x = df[c].to_numpy(np.float64)
            nan = np.isnan(x)
            x = np.where(nan, median, x)
            if log:
                x = np.log1p(np.maximum(x, 0.0))
            cols.append(np.clip((x - mean) / std, -self.clip, self.clip))
            if has_nan:
                cols.append(nan.astype(np.float64))
        for c, levels in self.levels_.items():
            x = df[c].to_numpy()
            cols += [(x == v).astype(np.float64) for v in levels]
        return np.column_stack(cols)
