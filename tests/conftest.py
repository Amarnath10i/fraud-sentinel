import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from sentinel import db
from sentinel.config import settings


@pytest.fixture(scope="session")
def test_db_url():
    """A throwaway database next to the main one; tests marked `db` are skipped
    when PostgreSQL is not reachable."""
    base = os.environ.get("SENTINEL_TEST_DB_URL", settings.db_url)
    url = make_conninfo(base, dbname=conninfo_to_dict(base)["dbname"] + "_test")
    try:
        db.ensure_database(url)
    except psycopg.OperationalError as e:
        pytest.skip(f"PostgreSQL not available: {e}")
    with db.connect(url) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        db.run_sql_file(conn, settings.paths.sql / "schema.sql")
        conn.commit()
    return url
