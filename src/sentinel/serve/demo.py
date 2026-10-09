"""Live demo: replay the deployment period through the production scorer.

The runner feeds test-period transactions and chargebacks, in time order, to
the same `Scorer` the API uses, at an adjustable rate. For display only it also
knows the ground truth of replayed transactions, so the dashboard can show
what the decisions were worth; the model never sees it.

Analysts work a review queue (highest expected saving first). Marking a case
as fraud sends a chargeback into the online state, so the card's next
transactions are scored with that knowledge, exactly as in production.
"""

from __future__ import annotations

import contextlib
import ctypes
import heapq
import logging
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sentinel.config import settings
from sentinel.decision import ACTION_NAMES, APPROVE, DECLINE, REVIEW
from sentinel.features.online import Event
from sentinel.features.spec import SPARKOV
from sentinel.monitoring.drift import SCORE_EDGES, alert_rate_ratio, psi
from sentinel.serve.scorer import Scorer

log = logging.getLogger(__name__)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_ACTION = {"approve": APPROVE, "review": REVIEW, "decline": DECLINE}
CARD_FEATURES = [
    "card__count_1h", "card__count_24h", "card__sum_amount_24h", "card__max_amount_7d",
    "card__distinct_merchant_24h", "card__mean_amount_all", "card__since_last", "card__cb_count_all",
]  # fmt: skip


@dataclass
class Item:
    seq: int
    txn_id: str
    ts: int
    card_id: int
    merchant: str
    category: str
    amount: float
    p_fraud: float
    decision: str
    reasons: list = field(default_factory=list)
    truth: bool | None = None  # None for manual transactions
    manual: bool = False
    latency_ms: float = 0.0
    merchant_id: int = 0


def _outcome(decision: str, truth: bool | None) -> str:
    if truth is None:
        return "manual"
    if truth:
        return {"approve": "missed fraud", "review": "fraud in review", "decline": "blocked fraud"}[
            decision
        ]
    return {"approve": "ok", "review": "friction", "decline": "false decline"}[decision]


class DemoRunner:
    def __init__(self, scorer: Scorer, start: pd.Timestamp | None = None) -> None:
        self.scorer = scorer
        self.costs = scorer.bundle.costs
        self.start = pd.Timestamp(start or settings.timeline.deploy_at)
        p = settings.paths.processed
        # Read only what the replay needs, filtered inside Parquet: loading the
        # whole history and slicing it costs ~400 MB of RSS that the allocator
        # keeps afterwards, which is what a small deployment pays for.
        test = pd.read_parquet(p / "transactions.parquet", filters=[("ts", ">=", self.start)])
        test["txn_id"] = test["txn_id"].astype(str)
        truth = pd.read_parquet(
            p / "ground_truth.parquet", filters=[("txn_id", "in", test["txn_id"].tolist())]
        ).astype({"txn_id": str})
        cbs = pd.read_parquet(
            p / "chargebacks.parquet", filters=[("reported_at", ">=", self.start)]
        ).astype({"txn_id": str})
        reported = pd.read_parquet(
            p / "transactions.parquet",
            columns=["txn_id", "card_id", "merchant_id", "category", "amount"],
            filters=[("txn_id", "in", cbs["txn_id"].tolist())],
        ).astype({"txn_id": str})
        merchants = pd.read_parquet(p / "merchants.parquet")
        customers = pd.read_parquet(p / "customers.parquet")

        self.merchant_names = dict(zip(merchants["merchant_id"], merchants["name"], strict=True))
        self.customers = customers.set_index("card_id")
        self._tx = test.merge(truth, on="txn_id").astype({"category": "category"})
        self._cb = cbs.merge(reported, on="txn_id").reset_index(drop=True)
        self.categories = sorted(self._tx["category"].unique())
        del test, truth, cbs, reported
        _release_memory()
        ts = np.concatenate([self._tx["ts"].to_numpy(), self._cb["reported_at"].to_numpy()])
        kind = np.r_[np.zeros(len(self._tx), np.int8), np.ones(len(self._cb), np.int8)]
        self._order = np.lexsort((kind, ts))
        self._n_tx = len(self._tx)

        self.lock = threading.RLock()
        self.rate = 25.0
        self.running = False
        self._reset_state()
        self._snapshot_path = None
        self._thread = threading.Thread(target=self._loop, name="demo", daemon=True)
        self._thread.start()

    # -- state ------------------------------------------------------------

    def _reset_state(self) -> None:
        self.cursor = 0
        self.seq = 0
        self.feed: deque[Item] = deque(maxlen=400)
        self.by_card: dict[int, deque[Item]] = {}
        self.items: dict[str, Item] = {}  # recent items by txn_id, for the inspector
        self.vectors: dict[str, np.ndarray] = {}  # their model inputs, for full explanations
        self.daily: dict[str, dict] = {}
        self.queue: dict[str, tuple[float, Item]] = {}
        self.alerts: deque[Item] = deque(maxlen=200)
        self.live_scores: deque[float] = deque(maxlen=20_000)
        self.latency: deque[float] = deque(maxlen=5_000)
        self.stats = {
            "transactions": 0, "frauds": 0, "fraud_amount": 0.0, "fraud_blocked_amount": 0.0,
            "fraud_missed_amount": 0.0, "reviews": 0, "declines": 0, "false_declines": 0,
            "chargebacks": 0, "cost": 0.0, "manual": 0,
        }  # fmt: skip

    def reset(self, snapshot) -> None:
        with self.lock:
            self.running = False
            self.scorer.load_state(snapshot)
            self._reset_state()

    def control(self, action: str, rate: float | None = None) -> None:
        with self.lock:
            if rate is not None:
                self.rate = float(min(max(rate, 1.0), 2_000.0))
            if action == "start":
                self.running = True
            elif action == "pause":
                self.running = False

    def _loop(self) -> None:
        tick = 0.05
        budget = 0.0
        while True:
            if not self.running:
                time.sleep(tick)
                continue
            t0 = time.perf_counter()
            budget += self.rate * tick
            while budget >= 1 and self.running:
                with self.lock:
                    if not self._step():
                        self.running = False
                        break
                budget -= 1
            time.sleep(max(0.0, tick - (time.perf_counter() - t0)))

    def _step(self) -> bool:
        if self.cursor >= len(self._order):
            return False
        j = int(self._order[self.cursor])
        self.cursor += 1
        if j < self._n_tx:
            r = self._tx.iloc[j]
            txn = r.to_dict()
            s = self.scorer.score(txn)
            self._record(s, txn, truth=bool(r["is_fraud"]))
        else:
            self.scorer.chargeback(self._cb.iloc[j - self._n_tx].to_dict())
            self.stats["chargebacks"] += 1
        return True

    def _record(self, s, txn: dict, truth: bool | None, manual: bool = False) -> Item:
        self.seq += 1
        ts = (
            txn["ts"]
            if isinstance(txn["ts"], int)
            else int((pd.Timestamp(txn["ts"]) - _EPOCH).total_seconds())
        )
        item = Item(
            seq=self.seq, txn_id=s.txn_id, ts=ts, card_id=int(txn["card_id"]),
            merchant=self.merchant_names.get(int(txn["merchant_id"]), str(txn["merchant_id"])),
            category=str(txn["category"]), amount=float(txn["amount"]), p_fraud=s.p_fraud,
            decision=s.decision, reasons=s.reasons, truth=truth, manual=manual,
            latency_ms=s.latency_ms, merchant_id=int(txn["merchant_id"]),
        )  # fmt: skip
        self.feed.append(item)
        self.by_card.setdefault(item.card_id, deque(maxlen=12)).append(item)
        self.items[item.txn_id] = item
        if s.x is not None:
            self.vectors[item.txn_id] = s.x
        if len(self.items) > 50_000:
            for k in list(self.items)[:10_000]:
                del self.items[k]
                self.vectors.pop(k, None)
        self.live_scores.append(s.p_fraud)
        self.latency.append(s.latency_ms)
        if s.decision != "approve":
            self.alerts.append(item)

        self.stats["manual"] += manual
        for bucket in (self.stats, self._day(ts)):
            self._bump(bucket, s.decision, truth, item.amount)
        if s.decision == "review":
            exp = self.costs.expected(np.array([s.p_fraud]), np.array([item.amount]))[0]
            saving = float(min(exp[APPROVE], exp[DECLINE]) - exp[REVIEW])
            self.queue[item.txn_id] = (saving, item)
            if len(self.queue) > 300:  # analysts can only hold so much: drop the least valuable
                for k, _ in heapq.nsmallest(
                    len(self.queue) - 300, self.queue.items(), key=lambda kv: kv[1][0]
                ):
                    del self.queue[k]
        return item

    def _day(self, ts: int) -> dict:
        key = time.strftime("%Y-%m-%d", time.gmtime(ts))
        day = self.daily.get(key)
        if day is None:
            day = self.daily[key] = {
                "date": key, "transactions": 0, "frauds": 0, "fraud_amount": 0.0,
                "fraud_blocked_amount": 0.0, "fraud_missed_amount": 0.0, "reviews": 0,
                "declines": 0, "false_declines": 0, "cost": 0.0,
            }  # fmt: skip
        return day

    def _bump(self, st: dict, decision: str, truth: bool | None, amount: float) -> None:
        st["transactions"] += 1
        st["reviews"] += decision == "review"
        st["declines"] += decision == "decline"
        if truth is None:
            return
        action = np.array([_ACTION[decision]])
        st["cost"] += float(
            self.costs.realized(action, np.array([float(truth)]), np.array([amount]))[0]
        )
        if truth:
            st["frauds"] += 1
            st["fraud_amount"] += amount
            if decision == "approve":
                st["fraud_missed_amount"] += amount
            else:
                st["fraud_blocked_amount"] += amount
        elif decision == "decline":
            st["false_declines"] += 1

    # -- views ------------------------------------------------------------

    def daily_view(self) -> list[dict]:
        with self.lock:
            days = [dict(d) for _, d in sorted(self.daily.items())]
        for d in days:
            d["saving_vs_approve_all"] = d["fraud_amount"] - d["cost"]
        return days

    def transaction(self, txn_id: str) -> dict:
        """One scored transaction with its full explanation and per-action expected cost."""
        with self.lock:
            item = self.items.get(txn_id)
            x = self.vectors.get(txn_id)
            in_queue = txn_id in self.queue
        if item is None:
            raise KeyError(txn_id)
        expected = self.costs.expected(np.array([item.p_fraud]), np.array([item.amount]))[0]
        return self._item_dict(item) | {
            "in_queue": in_queue,
            "expected_cost": dict(zip(ACTION_NAMES, map(float, expected), strict=True)),
            "explanation": self.scorer.explain(x) if x is not None else None,
        }

    def whatif(
        self, card_id: int, merchant_id: int, category: str, amount: float, hour: int | None,
        distance_km: float,
    ) -> dict:  # fmt: skip
        """Score a hypothetical transaction for this card right now, without recording it."""
        c = self.customers.loc[card_id]
        txn = {
            "card_id": card_id, "merchant_id": merchant_id, "category": category, "amount": amount,
            # due north of home: 1 degree of latitude is ~111.2 km
            "merch_lat": float(c["home_lat"]) + distance_km / 111.2, "merch_lon": float(c["home_lon"]),
        }  # fmt: skip
        out = self.scorer.whatif(txn, hour=hour)
        out["clock"] = self.scorer.engine.now
        return out

    def view(self, after: int = 0) -> dict:
        with self.lock:
            st = dict(self.stats)
            items = [self._item_dict(i) for i in self.feed if i.seq > after][-200:]
            alerts = [self._item_dict(i) for i in self.alerts] if after == 0 else []
            lat = np.array(self.latency) if self.latency else np.array([0.0])
            now = self.scorer.engine.now
        st["saving_vs_approve_all"] = st["fraud_amount"] - st["cost"]
        return {
            "running": self.running,
            "rate": self.rate,
            "clock": now,
            "progress": self.cursor / max(len(self._order), 1),
            "stats": st,
            "latency_p50": float(np.percentile(lat, 50)),
            "latency_p99": float(np.percentile(lat, 99)),
            "items": items,
            "alerts": alerts,
            "model": self.scorer.bundle.version,
        }

    def _item_dict(self, i: Item) -> dict:
        d = asdict(i)
        d["outcome"] = _outcome(i.decision, i.truth)
        return d

    def card(self, card_id: int) -> dict:
        if card_id not in self.customers.index:
            raise KeyError(card_id)
        c = self.customers.loc[card_id]
        with self.lock:
            now = self.scorer.engine.now or 0
            vec = self.scorer.engine.peek(Event(now, 0, card_id, 0, ""))
            recent = [self._item_dict(i) for i in reversed(self.by_card.get(card_id, []))]
        names = SPARKOV.names
        age = (pd.Timestamp(now, unit="s") - pd.Timestamp(c["dob"])).days / 365.25 if now else None
        return {
            "card_id": card_id,
            "masked": f"•••• {str(card_id)[-4:]}",
            "city": f"{c['city']}, {c['state']}",
            "job": c["job"],
            "age": round(age, 1) if age else None,
            "features": {f: _clean(vec[names.index(f)]) for f in CARD_FEATURES},
            "recent": recent,
        }

    def queue_view(self) -> list[dict]:
        with self.lock:
            items = sorted(self.queue.values(), key=lambda kv: kv[0], reverse=True)[:100]
            return [self._item_dict(i) | {"saving": s} for s, i in items]

    def resolve(self, txn_id: str, is_fraud: bool) -> None:
        with self.lock:
            entry = self.queue.pop(txn_id, None)
            if entry is None:
                raise KeyError(txn_id)
            if is_fraud:
                i = entry[1]
                self.scorer.chargeback(
                    {"reported_at": self.scorer.engine.now, "card_id": i.card_id,
                     "merchant_id": i.merchant_id, "category": i.category, "amount": i.amount}
                )  # fmt: skip
                self.stats["chargebacks"] += 1

    def score_manual(
        self,
        card_id: int,
        merchant_id: int,
        category: str,
        amount: float,
        away: bool,
        distance_km: float | None = None,
    ) -> dict:
        """Score and record a real transaction at the replay clock. `distance_km`
        places the merchant exactly (due north of home, as in what-if); otherwise
        `away` picks a random spot ~800 km away or around the corner."""
        c = self.customers.loc[card_id]
        rng = np.random.default_rng()
        offset = 7.0 if away else 0.1  # degrees: ~800 km away vs around the corner
        if distance_km is not None:
            lat, lon = float(c["home_lat"]) + distance_km / 111.2, float(c["home_lon"])
        else:
            lat = float(c["home_lat"] + rng.uniform(-offset, offset))
            lon = float(c["home_lon"] + rng.uniform(-offset, offset))
        with self.lock:
            now = self.scorer.engine.now
            txn = {
                "txn_id": uuid.uuid4().hex, "ts": now, "card_id": card_id, "merchant_id": merchant_id,
                "category": category, "amount": amount, "merch_lat": lat, "merch_lon": lon,
            }  # fmt: skip
            s = self.scorer.score(txn)
            item = self._record(s, txn, truth=None, manual=True)
        return self._item_dict(item)

    def options(self) -> dict:
        cards = self.customers.reset_index()
        return {
            "cards": [
                {
                    "card_id": int(r.card_id),
                    "label": f"•••• {str(r.card_id)[-4:]} · {r.city}, {r.state}",
                }
                for r in cards.itertuples(index=False)
            ],
            "merchants": [
                {"merchant_id": int(k), "name": v} for k, v in sorted(self.merchant_names.items())
            ],
            "categories": self.categories,
        }

    def monitoring(self) -> dict:
        """Score drift, as PSI and as alert volume: the share of transactions
        scored at or above each threshold, validation period vs live."""
        ref = self.scorer.bundle.reference_scores
        with self.lock:
            live = np.array(self.live_scores)
        thresholds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.9]
        out: dict = {"live_n": int(len(live)), "psi": None, "thresholds": thresholds}
        if ref is not None:
            out["reference"] = [float(np.mean(ref >= t)) for t in thresholds]
        if len(live):
            out["live"] = [float(np.mean(live >= t)) for t in thresholds]
        if ref is not None and len(live) >= 1_000:
            out["psi"] = psi(ref, live, edges=SCORE_EDGES)
            out["alert_ratio"] = alert_rate_ratio(ref, live, 0.01)
        backtest = settings.paths.reports / "backtest_monthly.csv"
        if backtest.exists():
            bt = pd.read_csv(backtest)
            out["backtest"] = {
                pol: grp.sort_values("month")[["month", "pr_auc"]].to_dict("records")
                for pol, grp in bt.groupby("policy")
            }
        return out


def _release_memory() -> None:
    """Hand freed buffers back to the OS so the resident size reflects live data."""
    import gc

    import pyarrow as pa

    gc.collect()
    pa.default_memory_pool().release_unused()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def _clean(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
