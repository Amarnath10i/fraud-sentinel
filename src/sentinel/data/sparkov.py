"""Load the Sparkov credit-card dataset into the normalized schema.

The Kaggle release (kartik2112/fraud-detection) ships as two flat CSVs with
customer attributes repeated on every row and the fraud label attached to the
transaction. Here it is split into the tables in `sql/schema.sql`:

* customer attributes go to `customers` (one row per card),
* merchant names get surrogate keys in `merchants`,
* the label moves out of `transactions` into `ground_truth` (evaluation only)
  and `chargebacks` (the delayed signal the system actually learns from).

Names and street addresses are dropped: nothing downstream needs them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel import db
from sentinel.config import ChargebackModel, settings

log = logging.getLogger(__name__)

RAW_FILES = ("fraudTrain.csv", "fraudTest.csv")
_RAW_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender", "city",
    "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num", "merch_lat",
    "merch_long", "is_fraud",
]  # fmt: skip


@dataclass
class Tables:
    customers: pd.DataFrame
    merchants: pd.DataFrame
    transactions: pd.DataFrame
    chargebacks: pd.DataFrame
    ground_truth: pd.DataFrame

    def items(self):
        # FK order: parents before children.
        return [
            ("customers", self.customers),
            ("merchants", self.merchants),
            ("transactions", self.transactions),
            ("chargebacks", self.chargebacks),
            ("ground_truth", self.ground_truth),
        ]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(raw_dir / name, usecols=_RAW_COLUMNS, dtype={"zip": str, "cc_num": "int64"})
        for name in RAW_FILES
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["trans_date_trans_time"]).dt.tz_localize("UTC")
    return raw.sort_values(["ts", "trans_num"], kind="stable").reset_index(drop=True)


def normalize(raw: pd.DataFrame, chargeback_model: ChargebackModel) -> Tables:
    customers = (
        raw.groupby("cc_num", sort=True)
        .first()
        .reset_index()
        .rename(columns={"cc_num": "card_id", "lat": "home_lat", "long": "home_lon"})
        [["card_id", "gender", "dob", "job", "city", "state", "zip", "home_lat", "home_lon", "city_pop"]]
    )  # fmt: skip
    if raw.groupby("cc_num")[["dob", "lat", "long", "job"]].nunique().max().max() != 1:
        raise ValueError("customer attributes are not constant per card")

    # Every merchant name carries a "fraud_" prefix from the data generator.
    names = raw["merchant"].str.removeprefix("fraud_")
    merchant_names = np.sort(names.unique())
    merchants = pd.DataFrame(
        {"merchant_id": np.arange(1, len(merchant_names) + 1), "name": merchant_names}
    )
    merchant_id = names.map(dict(zip(merchants["name"], merchants["merchant_id"], strict=True)))

    transactions = pd.DataFrame(
        {
            "txn_id": raw["trans_num"],
            "ts": raw["ts"],
            "card_id": raw["cc_num"],
            "merchant_id": merchant_id.astype("int32"),
            "category": raw["category"],
            "amount": raw["amt"].round(2),
            "merch_lat": raw["merch_lat"],
            "merch_lon": raw["merch_long"],
        }
    )
    ground_truth = pd.DataFrame(
        {"txn_id": raw["trans_num"], "is_fraud": raw["is_fraud"].astype(bool)}
    )
    chargebacks = simulate_chargebacks(transactions, ground_truth, chargeback_model)
    return Tables(customers, merchants, transactions, chargebacks, ground_truth)


def simulate_chargebacks(
    transactions: pd.DataFrame, ground_truth: pd.DataFrame, model: ChargebackModel
) -> pd.DataFrame:
    """Give every fraudulent transaction a report time ts + delay.

    Delays are log-normal (median `model.median_days`) and floored at
    `model.min_days`. Draws are made in txn_id order so the result does not
    depend on how the input happens to be sorted.
    """
    fraud = transactions.loc[ground_truth["is_fraud"].to_numpy(), ["txn_id", "ts"]]
    fraud = fraud.sort_values("txn_id", kind="stable")
    rng = np.random.default_rng(model.seed)
    days = model.median_days * np.exp(model.sigma * rng.standard_normal(len(fraud)))
    days = np.maximum(days, model.min_days)
    delay = pd.to_timedelta(np.round(days * 86400).astype("int64"), unit="s")
    out = pd.DataFrame(
        {
            "txn_id": fraud["txn_id"].to_numpy(),
            "txn_ts": fraud["ts"].to_numpy(),
            "reported_at": (fraud["ts"] + delay).to_numpy(),
        }
    )
    return out.sort_values("reported_at", kind="stable").reset_index(drop=True)


_DROP_ORDER = [
    "predictions",
    "model_registry",
    "ground_truth",
    "chargebacks",
    "transactions",
    "merchants",
    "customers",
]


def ingest(
    raw_dir: Path | None = None, *, reset: bool = True, write_parquet: bool = True
) -> Tables:
    raw_dir = raw_dir or settings.paths.raw_sparkov
    log.info("reading raw CSVs from %s", raw_dir)
    tables = normalize(load_raw(raw_dir), settings.chargebacks)
    tl = settings.timeline

    db.ensure_database()
    with db.connect() as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS " + ", ".join(_DROP_ORDER) + " CASCADE")
        db.run_sql_file(conn, settings.paths.sql / "schema.sql")
        db.create_month_partitions(conn, "transactions", tl.data_start, tl.data_end)
        for name, frame in tables.items():
            log.info("COPY %s (%d rows)", name, len(frame))
            db.copy_from_frame(conn, name, frame)
        conn.commit()
    with db.connect(autocommit=True) as conn:
        conn.execute("ANALYZE")

    if write_parquet:
        out = settings.paths.processed
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(out / f"{name}.parquet", index=False)
    return tables
