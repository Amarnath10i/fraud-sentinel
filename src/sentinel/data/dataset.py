"""Model-ready frames with point-in-time labels.

`load_frame()` joins transactions, customers, aggregate features, row features
and derived features into one table (one row per transaction).

Labels are never read from `ground_truth` for training. `observed_labels(as_of)`
reconstructs what the fraud team would actually know at `as_of`: a transaction
is fraud if a chargeback was reported by then, and otherwise assumed legitimate.
That assumption is only trusted for transactions older than the maturity
window, so `split()` only hands out training rows whose labels have matured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from sentinel.config import Timeline, settings
from sentinel.features.build import load_aggregates
from sentinel.features.events import epoch_seconds
from sentinel.features.rows import derived_features, row_features
from sentinel.features.spec import SPARKOV, FeatureSpec

# Aggregates kept out of the model: raw platform-wide volume trends with the
# calendar and is better used for monitoring than as a signal.
_EXCLUDED = {"global__count_30d", "global__cb_count_30d", "category__count_30d"}


def model_features(spec: FeatureSpec = SPARKOV) -> list[str]:
    rows = ["amount", "hour", "day_of_week", "is_night", "age_years", "gender_m",
            "log_city_pop", "dist_home_km", "category_code"]  # fmt: skip
    derived = ["amount_to_card_mean", "amount_zscore_card", "amount_to_card_max_7d",
               "amount_to_card_category_mean", "card_avg_amount_24h", "merchant_cb_rate_30d",
               "category_cb_rate_30d"]  # fmt: skip
    return rows + [n for n in spec.names if n not in _EXCLUDED] + derived


CATEGORICAL = ["category_code"]


def load_frame(spec: FeatureSpec = SPARKOV, refresh: bool = False) -> pd.DataFrame:
    p = settings.paths.processed
    cache = p / f"model_frame_{spec.version}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    tx = pd.read_parquet(p / "transactions.parquet")
    customers = pd.read_parquet(p / "customers.parquet")
    truth = pd.read_parquet(p / "ground_truth.parquet")
    cb = pd.read_parquet(p / "chargebacks.parquet")
    agg = load_aggregates(spec)

    tx = tx.merge(customers, on="card_id", how="left", validate="many_to_one")
    tx["txn_id"] = tx["txn_id"].astype(str)
    tx = tx.merge(agg.drop(columns="ts"), on="txn_id", how="inner", validate="one_to_one")
    tx = tx.merge(truth.astype({"txn_id": str}), on="txn_id", validate="one_to_one")
    reported = cb.astype({"txn_id": str}).set_index("txn_id")["reported_at"]
    tx["reported_at"] = tx["txn_id"].map(reported)

    ts = epoch_seconds(tx["ts"])
    dob_days = (pd.to_datetime(tx["dob"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    rows = row_features(
        {
            "ts": ts,
            "amount": tx["amount"].to_numpy(),
            "category": tx["category"].to_numpy(),
            "gender": tx["gender"].to_numpy(),
            "dob_days": dob_days,
            "city_pop": tx["city_pop"].to_numpy(),
            "home_lat": tx["home_lat"].to_numpy(),
            "home_lon": tx["home_lon"].to_numpy(),
            "merch_lat": tx["merch_lat"].to_numpy(),
            "merch_lon": tx["merch_lon"].to_numpy(),
        }
    )
    aggs = {n: tx[n].to_numpy(np.float64) for n in spec.names}
    derived = derived_features(rows["amount"], aggs)

    keep = tx[["txn_id", "ts", "card_id", "merchant_id", "category", "is_fraud", "reported_at"]]
    frame = pd.concat(
        [
            keep.reset_index(drop=True),
            pd.DataFrame(aggs),
            pd.DataFrame(rows),
            pd.DataFrame(derived),
        ],
        axis=1,
    )
    frame = frame.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)
    frame.to_parquet(cache, index=False)
    return frame


def observed_labels(frame: pd.DataFrame, as_of: datetime) -> np.ndarray:
    """1 if a chargeback had been reported by `as_of`, else 0."""
    reported = frame["reported_at"]
    return (reported.notna() & (reported <= pd.Timestamp(as_of))).to_numpy(np.int8)


@dataclass
class Split:
    name: str
    frame: pd.DataFrame
    y: np.ndarray  # labels the model is allowed to see (observed) or ground truth (test)
    label_source: str


def window(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    ts = frame["ts"]
    return frame[(ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))]


def split(frame: pd.DataFrame, tl: Timeline | None = None) -> dict[str, Split]:
    """Train / validation with labels as known at deploy time; test with ground truth."""
    tl = tl or settings.timeline
    train = window(frame, tl.data_start, tl.train_end)
    valid = window(frame, tl.train_end, tl.valid_end)
    test = window(frame, tl.deploy_at, tl.data_end)
    return {
        "train": Split(
            "train", train, observed_labels(train, tl.deploy_at), "chargebacks by deploy"
        ),
        "valid": Split(
            "valid", valid, observed_labels(valid, tl.deploy_at), "chargebacks by deploy"
        ),
        "test": Split("test", test, test["is_fraud"].to_numpy(np.int8), "ground truth"),
    }
