"""Thin PostgreSQL helpers built on psycopg 3.

Bulk movement in both directions goes through COPY, which is one to two orders
of magnitude faster than row-by-row INSERT/fetch for millions of rows.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from sentinel.config import settings

_CHUNK = 8 << 20


def enabled() -> bool:
    """False when the deployment has no database (SENTINEL_DB_URL=none)."""
    return settings.db_url.strip().lower() not in ("", "none")


def connect(url: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """Open a connection whose session time zone is UTC (fails after 5 s, never hangs)."""
    conninfo = make_conninfo(url or settings.db_url, options="-c timezone=UTC", connect_timeout=5)
    return psycopg.connect(conninfo, autocommit=autocommit)


def ensure_database(url: str | None = None) -> None:
    """Create the target database if it does not exist yet."""
    url = url or settings.db_url
    dbname = conninfo_to_dict(url)["dbname"]
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


def run_sql_file(conn: psycopg.Connection, path: Path) -> None:
    conn.execute(path.read_text(encoding="utf-8"))


def create_month_partitions(
    conn: psycopg.Connection, table: str, start: datetime, end: datetime
) -> list[str]:
    """Create one range partition per calendar month covering [start, end)."""
    created = []
    month = datetime(start.year, start.month, 1, tzinfo=start.tzinfo)
    while month < end:
        nxt = datetime(month.year + month.month // 12, month.month % 12 + 1, 1, tzinfo=month.tzinfo)
        name = f"{table}_{month:%Y_%m}"
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})"
            ).format(
                sql.Identifier(name),
                sql.Identifier(table),
                sql.Literal(month.isoformat()),
                sql.Literal(nxt.isoformat()),
            )
        )
        created.append(name)
        month = nxt
    conn.execute(
        sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF {} DEFAULT").format(
            sql.Identifier(f"{table}_default"), sql.Identifier(table)
        )
    )
    return created


def copy_from_frame(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> None:
    """Bulk-load a DataFrame into `table` using COPY ... FROM STDIN (CSV)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, date_format="%Y-%m-%d %H:%M:%S%z")
    data = buf.getvalue()
    stmt = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv)").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, df.columns))
    )
    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for i in range(0, len(data), _CHUNK):
            copy.write(data[i : i + _CHUNK])


def read_frame(
    conn: psycopg.Connection,
    query: str | sql.Composable,
    parse_dates: list[str] | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame:
    """Run `query` and return the result as a DataFrame via COPY ... TO STDOUT."""
    q = sql.SQL(query) if isinstance(query, str) else query
    stmt = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER)").format(q)
    out = io.BytesIO()
    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for block in copy:
            out.write(block)
    out.seek(0)
    df = pd.read_csv(out, dtype=dtype)
    for col in parse_dates or []:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df
