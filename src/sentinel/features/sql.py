"""Compile a `FeatureSpec` to a single PostgreSQL query (the training path).

Point-in-time rule in SQL: every window frame ends at

    RANGE BETWEEN <span> PRECEDING AND '1 microsecond' PRECEDING

i.e. strictly before the current row's timestamp, so rows sharing a timestamp
never see each other. The obvious alternative, `EXCLUDE GROUP`, gives the same
answer but forces PostgreSQL to re-aggregate the whole frame for every row,
which is quadratic on the 30-day merchant/category/global windows. A frame that
simply ends earlier keeps the moving-aggregate fast path: COUNT and SUM(numeric)
have inverse transition functions, so each row is added and removed once.

Two things window functions cannot do are handled differently:

* COUNT(DISTINCT ...) is not allowed as a window function, so `distinct`
  features use a correlated subquery that walks the (card_id, ts) index.
* Standard deviation is computed from exact NUMERIC moments (n, sum, sum of
  squares) so that it matches the streaming engine's integer arithmetic.

Chargebacks are merged into the same event stream (`is_txn = false`) at their
`reported_at` time, so "chargebacks reported in the last 30 days" is a plain
window count rather than a range join.
"""

from __future__ import annotations

from sentinel.features.spec import SQL_INTERVALS, Agg, FeatureSpec


def _entity(agg: Agg) -> str:
    return "_".join(k.removesuffix("_id") for k in agg.keys) or "global"


def _window_name(agg: Agg) -> str:
    return f"w_{_entity(agg)}_{agg.window or 'all'}"


def _window_def(agg: Agg) -> str:
    start = "UNBOUNDED" if agg.window is None else f"INTERVAL '{SQL_INTERVALS[agg.window]}'"
    partition = f"PARTITION BY {', '.join(agg.keys)} " if agg.keys else ""
    return (
        f"{partition}ORDER BY ts "
        f"RANGE BETWEEN {start} PRECEDING AND INTERVAL '1 microsecond' PRECEDING"
    )


class _Primitives:
    """Deduplicated window aggregates computed in the inner query."""

    def __init__(self) -> None:
        self.exprs: dict[str, str] = {}

    def __call__(self, expr: str) -> str:
        if expr not in self.exprs:
            self.exprs[expr] = f"p{len(self.exprs)}"
        return f"a.{self.exprs[expr]}"


def compile_sql(
    spec: FeatureSpec, transactions: str = "transactions", chargebacks: str = "chargebacks"
) -> str:
    prim = _Primitives()
    windows: dict[str, str] = {}
    features: list[str] = []

    for agg in spec.aggs:
        flt = "is_txn" if agg.stream == "txn" else "NOT is_txn"
        w = _window_name(agg)
        windows[w] = _window_def(agg)

        def window_agg(fn: str, w: str = w, flt: str = flt) -> str:
            return prim(f"{fn} FILTER (WHERE {flt}) OVER {w}")

        if agg.op == "count":
            expr = f"{window_agg('COUNT(*)')}::float8"
        elif agg.op == "sum":
            expr = f"COALESCE({window_agg('SUM(amount)')}, 0)::float8"
        elif agg.op == "mean":
            n, s = window_agg("COUNT(*)"), window_agg("SUM(amount)")
            expr = f"({s} / NULLIF({n}, 0))::float8"
        elif agg.op == "std":
            n, s = window_agg("COUNT(*)"), window_agg("SUM(amount)")
            ss = window_agg("SUM(amount * amount)")
            # CASE, not NULLIF: PostgreSQL's GREATEST ignores NULLs and would turn
            # "no history" into a standard deviation of 0.
            expr = (
                f"(CASE WHEN {n} > 0 THEN "
                f"SQRT(GREATEST(({n} * {ss} - {s} * {s}) / ({n} * {n}), 0)) END)::float8"
            )
        elif agg.op == "max":
            expr = f"{window_agg('MAX(amount)')}::float8"
        elif agg.op == "since_last":
            expr = f"EXTRACT(EPOCH FROM a.ts - {window_agg('MAX(ts)')})::float8"
        elif agg.op == "since_first":
            expr = f"EXTRACT(EPOCH FROM a.ts - {window_agg('MIN(ts)')})::float8"
        elif agg.op == "distinct":
            match = " AND ".join(f"t2.{k} = a.{k}" for k in agg.keys)
            interval = SQL_INTERVALS[agg.window]
            expr = (
                f"(SELECT COUNT(DISTINCT t2.{agg.column}) FROM {transactions} t2 "
                f"WHERE {match} AND t2.ts >= a.ts - INTERVAL '{interval}' AND t2.ts < a.ts)::float8"
            )
        else:  # pragma: no cover - guarded by Agg.__post_init__
            raise ValueError(agg.op)
        features.append(f'{expr} AS "{agg.name}"')

    inner_cols = ",\n        ".join(f"{e} AS {alias}" for e, alias in prim.exprs.items())
    window_clause = ",\n        ".join(f"{name} AS ({d})" for name, d in windows.items())
    feature_cols = ",\n    ".join(features)
    return f"""
WITH ev AS (
    SELECT txn_id, ts, TRUE AS is_txn, card_id, merchant_id, category, amount
    FROM {transactions}
    UNION ALL
    SELECT t.txn_id, c.reported_at, FALSE, t.card_id, t.merchant_id, t.category, t.amount
    FROM {chargebacks} c
    JOIN {transactions} t ON t.txn_id = c.txn_id AND t.ts = c.txn_ts
),
a AS (
    SELECT txn_id, ts, is_txn, card_id, merchant_id, category,
        {inner_cols}
    FROM ev
    WINDOW
        {window_clause}
)
SELECT a.txn_id,
    EXTRACT(EPOCH FROM a.ts)::bigint AS ts,
    {feature_cols}
FROM a
WHERE a.is_txn
""".strip()
