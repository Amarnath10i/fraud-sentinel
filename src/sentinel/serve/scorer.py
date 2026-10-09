"""Online scoring: one transaction in, one decision out.

    request -> streaming feature engine -> row + derived features
            -> LightGBM -> isotonic calibration -> Bayes cost policy
            -> reason codes (TreeSHAP), only for review/decline

The row and derived features are the same NumPy functions used to build the
training frame, applied to length-1 arrays, and the aggregates come from the
engine that is tested for exact parity with the SQL training features. The
replay check (`serve/replay.py`) confirms the whole path end to end: served
probabilities equal the offline ones for every test transaction.

Scoring is idempotent: payment networks retry authorizations, and a retried
request must not be counted twice in the card's velocity features. A repeated
txn_id returns the original decision from an LRU of recent transactions and
leaves the online state untouched.

The engine is not thread-safe, so scoring is serialized with a lock. Scaling
out would mean sharding by card so each card's state lives in one worker; the
merchant/category/global aggregates would then have to move to a shared store.
"""

from __future__ import annotations

import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.decision import ACTION_NAMES, APPROVE, bayes_policy
from sentinel.ds import LRUCache
from sentinel.features.events import engine_frames
from sentinel.features.online import Event, OnlineFeatureEngine
from sentinel.features.rows import derived_features, row_features
from sentinel.features.spec import SPARKOV
from sentinel.serve.bundle import Bundle

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class Scored:
    txn_id: str
    p_fraud: float
    decision: str
    reasons: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    duplicate: bool = False  # a retry of a transaction already scored
    x: np.ndarray | None = field(default=None, repr=False)  # model input, for explanations


class UnknownCard(KeyError):
    pass


def _txn_key(txn_id) -> str:
    """Canonical id so "6c3f...70" and "6c3f-...-70" are the same transaction."""
    try:
        return uuid.UUID(str(txn_id)).hex
    except ValueError:
        return str(txn_id)


def _epoch(ts: datetime | pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int((ts - _EPOCH) // pd.Timedelta(seconds=1))


class Scorer:
    def __init__(
        self,
        bundle: Bundle,
        customers: pd.DataFrame,
        capacity: int | None = None,
        recent_capacity: int = 200_000,
    ):
        if bundle.feature_spec != SPARKOV.version:
            raise ValueError(
                f"model trained on feature spec {bundle.feature_spec}, engine runs {SPARKOV.version}"
            )
        self.bundle = bundle
        self.engine = OnlineFeatureEngine(SPARKOV, capacity=capacity)
        dob_days = (pd.to_datetime(customers["dob"]) - pd.Timestamp("1970-01-01")).dt.days
        self.customers = {
            int(r.card_id): (
                r.gender,
                int(d),
                int(r.city_pop),
                float(r.home_lat),
                float(r.home_lon),
            )
            for r, d in zip(customers.itertuples(index=False), dob_days, strict=True)
        }
        self.recent: LRUCache[str, Scored] = LRUCache(recent_capacity)
        self.lock = threading.Lock()

    # -- state ------------------------------------------------------------

    def warm(self, transactions: pd.DataFrame, chargebacks: pd.DataFrame, until: datetime) -> int:
        """Rebuild state from history strictly before `until`."""
        cutoff = pd.Timestamp(until)
        tx = transactions[transactions["ts"] < cutoff]
        cb = chargebacks[chargebacks["reported_at"] < cutoff]
        return self.engine.warm(*engine_frames(tx, cb))

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.engine, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: Path) -> None:
        with open(path, "rb") as f:
            engine = pickle.load(f)  # noqa: S301 - our own snapshot file
        if engine.spec.version != SPARKOV.version:
            raise ValueError("state snapshot was built for a different feature spec")
        self.engine = engine

    # -- scoring ----------------------------------------------------------

    def _vector(
        self, txn: dict, ts: int, record: bool = True, hour: int | None = None
    ) -> np.ndarray:
        """Model input for a transaction. `record=False` reads the online state
        without adding the transaction to it (what-if analysis); `hour` then
        overrides the time-of-day features."""
        try:
            gender, dob_days, city_pop, home_lat, home_lon = self.customers[int(txn["card_id"])]
        except KeyError as e:
            raise UnknownCard(txn["card_id"]) from e
        amount = float(txn["amount"])
        ev = Event(
            ts,
            round(amount * 100),
            int(txn["card_id"]),
            int(txn["merchant_id"]),
            str(txn["category"]),
        )
        agg_vec = self.engine.process(ev, "txn") if record else self.engine.peek(ev)
        one = lambda v: np.array([v])  # noqa: E731
        rows = row_features(
            {
                "ts": one(ts), "amount": one(amount), "category": one(ev.category),
                "gender": one(gender), "dob_days": one(dob_days), "city_pop": one(city_pop),
                "home_lat": one(home_lat), "home_lon": one(home_lon),
                "merch_lat": one(float(txn["merch_lat"])), "merch_lon": one(float(txn["merch_lon"])),
            }
        )  # fmt: skip
        if hour is not None:
            rows["hour"] = one(float(hour))
            rows["is_night"] = one(float(hour >= 22 or hour < 4))
        aggs = {name: agg_vec[i : i + 1] for i, name in enumerate(SPARKOV.names)}
        values = rows | aggs | derived_features(rows["amount"], aggs)
        return np.array([[values[f][0] for f in self.bundle.features]])

    def explain(self, x: np.ndarray) -> dict:
        """Full TreeSHAP decomposition of one prediction, largest effects first.

        Contributions are in log-odds of the raw model and sum, with the base
        value, to its margin; calibration then maps that margin's probability
        monotonically to the served p_fraud.
        """
        contrib = self.bundle.booster.predict(x, pred_contrib=True, num_threads=1)[0]
        base, parts = float(contrib[-1]), contrib[:-1]
        order = np.argsort(-np.abs(parts))
        return {
            "base_value": base,
            "margin": float(base + parts.sum()),
            "contributions": [
                {
                    "feature": self.bundle.features[i],
                    "value": None if np.isnan(x[0, i]) else float(x[0, i]),
                    "contribution": float(parts[i]),
                }
                for i in order
            ],
        }

    def reasons(self, x: np.ndarray, k: int = 3) -> list[dict]:
        contrib = self.bundle.booster.predict(x, pred_contrib=True, num_threads=1)[0][:-1]
        top = np.argsort(contrib)[::-1][:k]
        return [
            {
                "feature": self.bundle.features[i],
                "value": None if np.isnan(x[0, i]) else float(x[0, i]),
                "contribution": float(contrib[i]),
            }
            for i in top
            if contrib[i] > 0
        ]

    def score(self, txn: dict) -> Scored:
        t0 = time.perf_counter()
        key = _txn_key(txn["txn_id"])
        with self.lock:
            seen = self.recent.get(key)
            if seen is not None:
                return replace(seen, duplicate=True, latency_ms=(time.perf_counter() - t0) * 1e3)
            ts = txn["ts"] if isinstance(txn["ts"], int) else _epoch(txn["ts"])
            x = self._vector(txn, ts)
            p = float(self.bundle.predict(x)[0])
            action = int(bayes_policy([p], [float(txn["amount"])], self.bundle.costs)[0])
            reasons = [] if action == APPROVE else self.reasons(x)
            out = Scored(str(txn["txn_id"]), p, ACTION_NAMES[action], reasons, x=x)
            self.recent.put(key, out)
        return replace(out, latency_ms=(time.perf_counter() - t0) * 1e3)

    def whatif(self, txn: dict, hour: int | None = None) -> dict:
        """Score a hypothetical transaction at the current clock without recording it."""
        with self.lock:
            now = self.engine.now or 0
            x = self._vector(txn, now, record=False, hour=hour)
        p = float(self.bundle.predict(x)[0])
        action = int(bayes_policy([p], [float(txn["amount"])], self.bundle.costs)[0])
        expected = self.bundle.costs.expected(np.array([p]), np.array([float(txn["amount"])]))[0]
        return {
            "p_fraud": p,
            "decision": ACTION_NAMES[action],
            "expected_cost": dict(zip(ACTION_NAMES, map(float, expected), strict=True)),
            "explanation": self.explain(x),
        }

    def chargeback(self, cb: dict) -> None:
        ts = cb["reported_at"] if isinstance(cb["reported_at"], int) else _epoch(cb["reported_at"])
        ev = Event(
            ts,
            round(float(cb["amount"]) * 100),
            int(cb["card_id"]),
            int(cb["merchant_id"]),
            str(cb["category"]),
        )
        with self.lock:
            self.engine.process(ev, "cb")
