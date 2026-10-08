"""Declarative feature definitions.

A feature is an aggregation over one of two event streams:

* ``txn`` - authorization requests, known the moment they happen;
* ``cb``  - chargebacks, which arrive days or weeks after the transaction they
  report and carry that transaction's card, merchant, category and amount.

Each `Agg` is compiled twice: to a PostgreSQL window query for training data
(`features.sql`) and to a streaming state machine for serving
(`features.online`). Both follow the same point-in-time rule:

    a feature for an event at time t only sees events with timestamp < t,
    and events that share a timestamp never see each other.

The second half matters because ~33k Sparkov events share a timestamp with
another event; letting them see each other would leak "future" data that a
real-time system would not have.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

Stream = Literal["txn", "cb"]
Op = Literal["count", "sum", "mean", "std", "max", "distinct", "since_last", "since_first"]

WINDOWS: dict[str, int] = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400, "30d": 30 * 86_400}
SQL_INTERVALS: dict[str, str] = {
    "1h": "1 hour",
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
}

_VALUE_OPS = {"sum", "mean", "std", "max", "distinct"}
_UNBOUNDED_ONLY = {"since_last", "since_first"}


@dataclass(frozen=True)
class Agg:
    stream: Stream
    keys: tuple[str, ...]
    op: Op
    column: str | None = None
    window: str | None = None  # None = all history

    def __post_init__(self) -> None:
        if (self.op in _VALUE_OPS) != (self.column is not None):
            raise ValueError(f"{self.op} {'needs' if self.op in _VALUE_OPS else 'takes no'} column")
        if self.op in _UNBOUNDED_ONLY and self.window is not None:
            raise ValueError(f"{self.op} is defined over all history only")
        if self.op == "distinct" and self.window is None:
            raise ValueError("distinct needs a bounded window")
        if self.window is not None and self.window not in WINDOWS:
            raise ValueError(f"unknown window {self.window}")
        if self.stream == "cb" and self.op not in {"count", "since_last"}:
            raise ValueError("chargeback stream supports count and since_last")

    @property
    def span(self) -> int | None:
        return None if self.window is None else WINDOWS[self.window]

    @property
    def name(self) -> str:
        entity = "_".join(k.removesuffix("_id") for k in self.keys) or "global"
        parts = [self.op if self.stream == "txn" else f"cb_{self.op}"]
        if self.column:
            parts.append(self.column.removesuffix("_id"))
        if self.op not in _UNBOUNDED_ONLY:
            parts.append(self.window or "all")
        return f"{entity}__{'_'.join(parts)}"


@dataclass(frozen=True)
class FeatureSpec:
    aggs: tuple[Agg, ...]

    def __post_init__(self) -> None:
        names = [a.name for a in self.aggs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature names")

    @cached_property
    def names(self) -> list[str]:
        return [a.name for a in self.aggs]

    @cached_property
    def version(self) -> str:
        """Content hash: changes whenever any definition changes."""
        return hashlib.sha1(repr(self.aggs).encode()).hexdigest()[:10]


def _card(op: Op, column: str | None = None, window: str | None = None) -> Agg:
    return Agg("txn", ("card_id",), op, column, window)


SPARKOV = FeatureSpec(
    (
        # Card velocity: how much and how often the card is being used right now.
        *(_card("count", window=w) for w in ("1h", "24h", "7d")),
        *(_card("sum", "amount", w) for w in ("1h", "24h", "7d")),
        _card("max", "amount", "24h"),
        _card("max", "amount", "7d"),
        _card("distinct", "merchant_id", "24h"),
        _card("distinct", "category", "24h"),
        _card("since_last"),
        # Card baseline: what normal looks like for this card.
        _card("count"),
        _card("mean", "amount"),
        _card("std", "amount"),
        _card("since_first"),
        Agg("txn", ("card_id", "category"), "count"),
        Agg("txn", ("card_id", "category"), "count", window="7d"),
        Agg("txn", ("card_id", "category"), "mean", "amount"),
        # Volume and recently reported fraud around the merchant and category.
        Agg("txn", ("merchant_id",), "count", window="30d"),
        Agg("cb", ("merchant_id",), "count", window="30d"),
        Agg("txn", ("category",), "count", window="30d"),
        Agg("cb", ("category",), "count", window="30d"),
        Agg("txn", (), "count", window="30d"),
        Agg("cb", (), "count", window="30d"),
        # The card's own reported fraud history.
        Agg("cb", ("card_id",), "count"),
        Agg("cb", ("card_id",), "since_last"),
    )
)
