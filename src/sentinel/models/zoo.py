"""Every model behind one interface: fit(train, y, valid, y_valid) / predict_proba(frame).

Validation data is used only for early stopping; nothing here touches test data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.data.dataset import CATEGORICAL, model_features
from sentinel.models.losses import focal_grad_hess
from sentinel.models.preprocess import LinearPreprocessor
from sentinel.models.scratch.gbdt import GBDTClassifier
from sentinel.models.scratch.logistic import LogisticRegressionIRLS, sigmoid


@dataclass
class BaseModel:
    features: list[str] = field(default_factory=model_features)
    fit_seconds: float = 0.0
    is_probability: bool = True

    def X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(np.float64)

    def fit(self, train, y, valid=None, y_valid=None):
        t0 = time.perf_counter()
        self._fit(train, y, valid, y_valid)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _fit(self, train, y, valid, y_valid) -> None:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RulesBaseline(BaseModel):
    """What a team without ML might start with: large amounts, worse at night."""

    is_probability: bool = False

    def _fit(self, train, y, valid, y_valid) -> None:
        pass

    def predict_proba(self, df):
        return df["amount"].to_numpy() * (1 + 2 * df["is_night"].to_numpy())


@dataclass
class ScratchLogistic(BaseModel):
    l2: float = 1.0

    def _fit(self, train, y, valid, y_valid) -> None:
        numeric = [f for f in self.features if f not in CATEGORICAL]
        self.prep = LinearPreprocessor(numeric, CATEGORICAL).fit(train)
        self.model = LogisticRegressionIRLS(l2=self.l2).fit(self.prep.transform(train), y)

    def predict_proba(self, df):
        return self.model.predict_proba(self.prep.transform(df))


@dataclass
class ScratchGBDT(BaseModel):
    params: dict[str, Any] = field(
        default_factory=lambda: dict(
            n_estimators=600, learning_rate=0.1, max_depth=6, min_child_weight=1.0,
            reg_lambda=1.0, subsample=0.8, colsample=0.8, early_stopping_rounds=40,
        )
    )  # fmt: skip

    def _fit(self, train, y, valid, y_valid) -> None:
        eval_set = (self.X(valid), y_valid) if valid is not None else None
        self.model = GBDTClassifier(**self.params).fit(self.X(train), y, eval_set=eval_set)

    def predict_proba(self, df):
        return self.model.predict_proba(self.X(df))


LGB_DEFAULTS: dict[str, Any] = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
    seed=0,
    num_threads=8,
    # multithreaded histogram sums are otherwise order-dependent: same seed, different model
    deterministic=True,
    force_row_wise=True,
)


@dataclass
class LightGBM(BaseModel):
    params: dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    focal: tuple[float, float] | None = None  # (gamma, alpha) -> custom focal objective

    def _fit(self, train, y, valid, y_valid) -> None:
        params = LGB_DEFAULTS | self.params
        cats = [f for f in CATEGORICAL if f in self.features]
        self.base_margin = 0.0
        if self.focal is not None:
            # custom objectives start from margin 0 (p = 0.5) unless told otherwise
            rate = float(np.mean(y))
            self.base_margin = float(np.log(rate / (1 - rate)))
            gamma, alpha = self.focal
            params = params | {
                "objective": lambda preds, data: focal_grad_hess(
                    data.get_label(), preds, gamma, alpha
                )
            }
        dtrain = lgb.Dataset(
            self.X(train), y, feature_name=self.features, categorical_feature=cats,
            init_score=np.full(len(y), self.base_margin) if self.focal else None,
        )  # fmt: skip
        callbacks, valid_sets = [], []
        if valid is not None:
            dvalid = dtrain.create_valid(
                self.X(valid), y_valid,
                init_score=np.full(len(y_valid), self.base_margin) if self.focal else None,
            )  # fmt: skip
            valid_sets = [dvalid]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            params = params | {"metric": "average_precision"}
        self.booster = lgb.train(
            params, dtrain, self.num_boost_round, valid_sets=valid_sets, callbacks=callbacks
        )

    def margin(self, df) -> np.ndarray:
        it = self.booster.best_iteration or None
        return self.booster.predict(self.X(df), num_iteration=it, raw_score=True) + self.base_margin

    def predict_proba(self, df):
        return sigmoid(self.margin(df))

    def contributions(self, df) -> np.ndarray:
        """TreeSHAP values (n, n_features + 1), last column = expected value."""
        return self.booster.predict(
            self.X(df), num_iteration=self.booster.best_iteration or None, pred_contrib=True
        )
