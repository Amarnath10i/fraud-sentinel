"""Turn stored tables into the event frames the streaming engine consumes."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def to_cents(amount: pd.Series) -> np.ndarray:
    return np.rint(amount.to_numpy(np.float64) * 100).astype(np.int64)


def engine_frames(
    transactions: pd.DataFrame, chargebacks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transactions and chargebacks with integer-second times and integer cents.

    Each chargeback carries the card, merchant, category and amount of the
    transaction it reports.
    """
    tx = pd.DataFrame(
        {
            "txn_id": transactions["txn_id"].astype(str).to_numpy(),
            "ts": epoch_seconds(transactions["ts"]),
            "amount": to_cents(transactions["amount"]),
            "card_id": transactions["card_id"].to_numpy(np.int64),
            "merchant_id": transactions["merchant_id"].to_numpy(np.int64),
            "category": transactions["category"].astype(str).to_numpy(),
        }
    )
    cb = pd.DataFrame(
        {
            "txn_id": chargebacks["txn_id"].astype(str).to_numpy(),
            "reported_at": epoch_seconds(chargebacks["reported_at"]),
        }
    ).merge(tx.drop(columns="ts"), on="txn_id", how="inner", validate="one_to_one")
    if len(cb) != len(chargebacks):
        raise ValueError("chargebacks reference unknown transactions")
    order = np.lexsort((tx["txn_id"].to_numpy(), tx["ts"].to_numpy()))
    return tx.iloc[order].reset_index(drop=True), cb.sort_values("reported_at", kind="stable")
