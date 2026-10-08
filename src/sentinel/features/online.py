"""Streaming feature engine: the serving-side compilation of a `FeatureSpec`.

Events must arrive in non-decreasing timestamp order. To honour the rule that
events sharing a timestamp never see each other, updates are buffered: an
event's contribution to the state is only applied once time moves past its
timestamp. Serving and the batch replay therefore produce exactly the same
features, and the parity test against the SQL compilation can demand equality.

State is kept per (stream, keys) group and per entity inside the group, in a
plain dict or in an `LRUCache` when memory has to be bounded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from sentinel.ds import LRUCache, SlidingWindowDistinct, SlidingWindowMax, SlidingWindowSum
from sentinel.features.spec import FeatureSpec, Stream

NAN = math.nan


class Event(NamedTuple):
    """One event as the engine sees it. Money is in integer cents."""

    ts: int
    amount: int
    card_id: int
    merchant_id: int
    category: str


_FIELD = {name: i for i, name in enumerate(Event._fields)}
_EMPTY = {"count": 0.0, "sum": 0.0, "distinct": 0.0}


class OutOfOrderEvent(ValueError):
    pass


class _EntityState:
    __slots__ = ("first_ts", "last_ts", "windows")

    def __init__(self, windows: list) -> None:
        self.windows = windows
        self.last_ts: int | None = None
        self.first_ts: int | None = None


@dataclass
class _Group:
    stream: Stream
    key_index: tuple[int, ...]
    slots: list[tuple[str, int | None, int | None]] = field(
        default_factory=list
    )  # kind, column, span
    reads: list[tuple[int, str, int | None]] = field(default_factory=list)  # out idx, op, slot
    states: dict | LRUCache = field(default_factory=dict)

    def slot(self, kind: str, column: str | None, span: int | None) -> int:
        desc = (kind, None if column is None else _FIELD[column], span)
        if desc not in self.slots:
            self.slots.append(desc)
        return self.slots.index(desc)

    def key(self, ev: tuple):
        idx = self.key_index
        if len(idx) == 1:
            return ev[idx[0]]
        return tuple(ev[i] for i in idx)

    def new_state(self) -> _EntityState:
        make = {"sum": SlidingWindowSum, "max": SlidingWindowMax, "distinct": SlidingWindowDistinct}
        return _EntityState([make[kind](span) for kind, _, span in self.slots])


class OnlineFeatureEngine:
    def __init__(self, spec: FeatureSpec, capacity: int | None = None) -> None:
        self.spec = spec
        self.names = spec.names
        groups: dict[tuple, _Group] = {}
        for i, agg in enumerate(spec.aggs):
            g = groups.get((agg.stream, agg.keys))
            if g is None:
                g = _Group(agg.stream, tuple(_FIELD[k] for k in agg.keys))
                if capacity is not None:
                    g.states = LRUCache(capacity)
                groups[(agg.stream, agg.keys)] = g
            if agg.op in ("count", "sum", "mean", "std"):
                # count/sum/mean/std share one integer-moment window on amount
                s = g.slot("sum", "amount", agg.span)
            elif agg.op in ("max", "distinct"):
                s = g.slot(agg.op, agg.column, agg.span)
            else:
                s = None
            g.reads.append((i, agg.op, s))
        self._groups = list(groups.values())
        self._by_stream = {st: [g for g in self._groups if g.stream == st] for st in ("txn", "cb")}
        self._now: int | None = None
        self._pending: list[tuple[Stream, tuple]] = []

    # -- public API -------------------------------------------------------

    def process(
        self, event: Event | Mapping, stream: Stream, compute: bool = True
    ) -> np.ndarray | None:
        """Feed one event. Returns the feature vector for a transaction, None for a chargeback."""
        ev = event if isinstance(event, tuple) else Event(**{k: event[k] for k in Event._fields})
        ts = ev[0]
        if self._now is not None and ts < self._now:
            raise OutOfOrderEvent(f"event at {ts} after {self._now}")
        if self._now is None or ts > self._now:
            self._flush()
            self._now = ts
        out = self._compute(ev) if stream == "txn" and compute else None
        self._pending.append((stream, ev))
        return out

    def warm(self, txns: pd.DataFrame, chargebacks: pd.DataFrame) -> int:
        """Replay history into the state without computing features (service start-up)."""
        n = 0
        for _, (stream, ev) in self._merge(txns, chargebacks):
            self.process(ev, stream, compute=False)
            n += 1
        return n

    def peek(self, event: Event) -> np.ndarray:
        """Features a transaction would get right now, without recording it.

        Only expired window entries are dropped (time never moves backwards),
        so this does not change any future feature value.
        """
        if self._now is not None and event.ts < self._now:
            raise OutOfOrderEvent(f"peek at {event.ts} before {self._now}")
        return self._compute(event)

    @property
    def now(self) -> int | None:
        return self._now

    def run(self, txns: pd.DataFrame, chargebacks: pd.DataFrame) -> pd.DataFrame:
        """Replay both streams in time order and return features for every transaction.

        `txns` needs txn_id, ts and the `Event` fields; `chargebacks` needs
        `reported_at` plus the reported transaction's `Event` fields.
        """
        n = len(txns)
        out = np.empty((n, len(self.names)))
        for i, (stream, ev) in self._merge(txns, chargebacks):
            vec = self.process(ev, stream)
            if vec is not None:
                out[i] = vec
        self._flush()
        df = pd.DataFrame(out, columns=self.names)
        df.insert(0, "txn_id", txns["txn_id"].to_numpy())
        return df

    # -- internals --------------------------------------------------------

    def _flush(self) -> None:
        for stream, ev in self._pending:
            self._update(ev, stream)
        self._pending.clear()

    def _compute(self, ev: tuple) -> np.ndarray:
        out = np.empty(len(self.names))
        ts = ev[0]
        for g in self._groups:
            st = g.states.get(g.key(ev))
            if st is None:
                for i, op, _ in g.reads:
                    out[i] = _EMPTY.get(op, NAN)
                continue
            windows = st.windows
            for w in windows:
                w.advance(ts)
            for i, op, s in g.reads:
                if op == "count":
                    out[i] = windows[s].count
                elif op == "sum":
                    out[i] = windows[s].total / 100
                elif op == "mean":
                    w = windows[s]
                    out[i] = w.total / w.count / 100 if w.count else NAN
                elif op == "std":
                    var = windows[s].var_pop()
                    out[i] = math.sqrt(max(var, 0.0)) / 100 if var is not None else NAN
                elif op == "max":
                    m = windows[s].max
                    out[i] = m / 100 if m is not None else NAN
                elif op == "distinct":
                    out[i] = windows[s].distinct
                elif op == "since_last":
                    out[i] = ts - st.last_ts if st.last_ts is not None else NAN
                elif op == "since_first":
                    out[i] = ts - st.first_ts if st.first_ts is not None else NAN
        return out

    def _update(self, ev: tuple, stream: Stream) -> None:
        ts = ev[0]
        for g in self._by_stream[stream]:
            key = g.key(ev)
            st = g.states.get(key)
            if st is None:
                st = g.new_state()
                if isinstance(g.states, LRUCache):
                    g.states.put(key, st)
                else:
                    g.states[key] = st
            for (_, col, _), w in zip(g.slots, st.windows, strict=True):
                w.advance(ts)
                w.add(ts, ev[col])
            st.last_ts = ts
            if st.first_ts is None:
                st.first_ts = ts

    @staticmethod
    def _merge(txns: pd.DataFrame, cbs: pd.DataFrame) -> Iterator[tuple[int, tuple[Stream, Event]]]:
        """Interleave the two streams by timestamp (ties: transactions first; order
        inside a timestamp does not matter because of the buffered updates)."""

        def events(df: pd.DataFrame, ts_col: str) -> Iterable[Event]:
            cols = [df[ts_col]] + [df[c] for c in Event._fields[1:]]
            return map(Event._make, zip(*(c.tolist() for c in cols), strict=True))

        ts = np.concatenate([txns["ts"].to_numpy(), cbs["reported_at"].to_numpy()])
        kind = np.concatenate([np.zeros(len(txns), np.int8), np.ones(len(cbs), np.int8)])
        order = np.lexsort((kind, ts))
        tx_events = list(events(txns, "ts"))
        cb_events = list(events(cbs, "reported_at"))
        n = len(txns)
        for j in order.tolist():
            if j < n:
                yield j, ("txn", tx_events[j])
            else:
                yield -1, ("cb", cb_events[j - n])
