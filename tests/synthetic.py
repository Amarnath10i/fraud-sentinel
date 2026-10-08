"""Small synthetic transaction streams with the awkward cases built in:
equal timestamps (also within a card), events exactly on window edges and
chargebacks reported at the same second as other transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery_pos", "misc_net", "shopping_net"]


def make_stream(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01", tz="UTC")
    # gaps of 0 create ties; gaps of exactly 1h / 24h hit window boundaries
    gaps = rng.choice(
        [0, 1, 60, 600, 3_600, 86_400, 20_000], size=n, p=[0.15, 0.1, 0.15, 0.2, 0.15, 0.1, 0.15]
    )
    ts = start + pd.to_timedelta(np.cumsum(gaps), unit="s")
    tx = pd.DataFrame(
        {
            "txn_id": [f"{i:032x}" for i in range(n)],
            "ts": ts,
            "card_id": rng.integers(1, 6, n),
            "merchant_id": rng.integers(1, 5, n),
            "category": rng.choice(CATEGORIES, n),
            "amount": np.round(rng.lognormal(3.5, 1.2, n), 2).clip(1.0, 20_000),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_lon": rng.uniform(-120, -75, n),
        }
    )
    fraud = rng.random(n) < 0.15
    delays = rng.choice([1, 3_600, 86_400, 5 * 86_400], size=fraud.sum())
    cb = pd.DataFrame(
        {
            "txn_id": tx.loc[fraud, "txn_id"].to_numpy(),
            "txn_ts": tx.loc[fraud, "ts"].to_numpy(),
            "reported_at": (tx.loc[fraud, "ts"] + pd.to_timedelta(delays, unit="s")).to_numpy(),
        }
    )
    # snap some reports onto an existing transaction time to create cross-stream ties
    snap = rng.random(len(cb)) < 0.3
    for i in np.flatnonzero(snap):
        later = tx["ts"][tx["ts"] > cb.at[i, "txn_ts"]]
        if len(later):
            cb.at[i, "reported_at"] = later.iloc[rng.integers(0, min(5, len(later)))]
    return tx, cb
