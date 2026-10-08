"""Offline feature store: aggregate features materialized in PostgreSQL.

The table name carries the spec version hash, so changing any feature
definition produces a new table instead of silently mixing versions.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import psycopg
from psycopg import sql

from sentinel import db
from sentinel.config import settings
from sentinel.features.spec import SPARKOV, FeatureSpec
from sentinel.features.sql import compile_sql

log = logging.getLogger(__name__)


def table_name(spec: FeatureSpec) -> str:
    return f"offline_features_{spec.version}"


def materialize(conn: psycopg.Connection, spec: FeatureSpec = SPARKOV) -> float:
    """(Re)build the offline aggregate table. Returns the build time in seconds."""
    table = sql.Identifier(table_name(spec))
    conn.execute("SET LOCAL work_mem = '1GB'")
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
    t0 = time.perf_counter()
    conn.execute(sql.SQL("CREATE TABLE {} AS ").format(table) + sql.SQL(compile_sql(spec)))
    elapsed = time.perf_counter() - t0
    conn.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (txn_id)").format(table))
    conn.commit()
    log.info("built %s in %.1fs", table_name(spec), elapsed)
    return elapsed


def load_aggregates(spec: FeatureSpec = SPARKOV, refresh: bool = False) -> pd.DataFrame:
    """Aggregate features for every transaction (cached as Parquet)."""
    cache = settings.paths.processed / f"{table_name(spec)}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    with db.connect() as conn:
        df = db.read_frame(
            conn,
            sql.SQL("SELECT * FROM {} ORDER BY ts, txn_id").format(
                sql.Identifier(table_name(spec))
            ),
            dtype={"txn_id": str},
        )
    df["txn_id"] = df["txn_id"].str.replace("-", "", regex=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
