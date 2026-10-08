"""The SQL compilation must produce exactly what the streaming engine produces."""

import numpy as np
import pandas as pd
import pytest

from sentinel import db
from sentinel.features.events import engine_frames
from sentinel.features.online import OnlineFeatureEngine
from sentinel.features.spec import SPARKOV
from sentinel.features.sql import compile_sql
from synthetic import make_stream

pytestmark = pytest.mark.db


def _load(url, tx, cb):
    customers = pd.DataFrame(
        {
            "card_id": sorted(tx["card_id"].unique()),
            "gender": "F",
            "dob": "1980-01-01",
            "job": "x",
            "city": "x",
            "state": "TX",
            "zip": "00000",
            "home_lat": 30.0,
            "home_lon": -97.0,
            "city_pop": 1000,
        }
    )
    merchants = pd.DataFrame({"merchant_id": sorted(tx["merchant_id"].unique())})
    merchants["name"] = "m" + merchants["merchant_id"].astype(str)
    with db.connect(url) as conn:
        conn.execute("TRUNCATE customers, merchants, transactions, chargebacks CASCADE")
        db.create_month_partitions(
            conn, "transactions", tx["ts"].min().to_pydatetime(), tx["ts"].max().to_pydatetime()
        )
        db.copy_from_frame(conn, "customers", customers)
        db.copy_from_frame(conn, "merchants", merchants)
        db.copy_from_frame(conn, "transactions", tx)
        db.copy_from_frame(conn, "chargebacks", cb)
        conn.commit()


@pytest.mark.parametrize("seed", [0, 1])
def test_sql_features_equal_streaming_features(test_db_url, seed):
    tx, cb = make_stream(n=400, seed=seed)
    _load(test_db_url, tx, cb)
    with db.connect(test_db_url) as conn:
        got = db.read_frame(conn, compile_sql(SPARKOV), dtype={"txn_id": str})
    got["txn_id"] = got["txn_id"].str.replace("-", "")
    got = got.set_index("txn_id")

    etx, ecb = engine_frames(tx, cb)
    expect = OnlineFeatureEngine(SPARKOV).run(etx, ecb).set_index("txn_id")

    assert len(got) == len(expect)
    got = got.loc[expect.index]
    np.testing.assert_array_equal(
        got["ts"].to_numpy(), etx.set_index("txn_id").loc[expect.index, "ts"]
    )
    for name in SPARKOV.names:
        np.testing.assert_allclose(
            got[name].to_numpy(), expect[name].to_numpy(), rtol=1e-12, err_msg=name
        )
