"""The streaming engine must match a brute-force definition of every feature
and must never look into the future."""

import math

import numpy as np
import pandas as pd
import pytest

from sentinel.features.events import engine_frames
from sentinel.features.online import Event, OnlineFeatureEngine, OutOfOrderEvent
from sentinel.features.spec import SPARKOV, Agg, FeatureSpec
from synthetic import make_stream


def brute_force(spec: FeatureSpec, tx: pd.DataFrame, cb: pd.DataFrame) -> np.ndarray:
    """O(n^2) reference: recompute every feature from scratch for every transaction."""
    streams = {
        "txn": {c: tx[c].to_numpy() for c in tx.columns},
        "cb": {**{c: cb[c].to_numpy() for c in cb.columns}, "ts": cb["reported_at"].to_numpy()},
    }
    out = np.full((len(tx), len(spec.aggs)), np.nan)
    for i, row in enumerate(tx.itertuples(index=False)):
        t = row.ts
        for j, agg in enumerate(spec.aggs):
            s = streams[agg.stream]
            mask = s["ts"] < t
            if agg.span is not None:
                mask &= s["ts"] >= t - agg.span
            for k in agg.keys:
                mask &= s[k] == getattr(row, k)
            amounts = s["amount"][mask]
            n = len(amounts)
            if agg.op == "count":
                v = n
            elif agg.op == "sum":
                v = amounts.sum() / 100
            elif agg.op == "mean":
                v = amounts.sum() / n / 100 if n else math.nan
            elif agg.op == "std":
                v = amounts.std() / 100 if n else math.nan
            elif agg.op == "max":
                v = amounts.max() / 100 if n else math.nan
            elif agg.op == "distinct":
                v = len(set(s[agg.column][mask].tolist()))
            elif agg.op == "since_last":
                v = t - s["ts"][mask].max() if n else math.nan
            elif agg.op == "since_first":
                v = t - s["ts"][mask].min() if n else math.nan
            out[i, j] = v
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_matches_brute_force(seed):
    tx, cb = engine_frames(*make_stream(n=300, seed=seed))
    got = OnlineFeatureEngine(SPARKOV).run(tx, cb)
    expect = brute_force(SPARKOV, tx, cb)
    for j, name in enumerate(SPARKOV.names):
        np.testing.assert_allclose(got[name].to_numpy(), expect[:, j], rtol=1e-9, err_msg=name)


def test_features_never_depend_on_the_future():
    tx, cb = engine_frames(*make_stream(n=300, seed=4))
    full = OnlineFeatureEngine(SPARKOV).run(tx, cb).set_index("txn_id")
    for cut in np.quantile(tx["ts"], [0.25, 0.5, 0.8]).astype(np.int64):
        # features of anything up to the cut must not change when the rest is gone
        past_tx = tx[tx["ts"] <= cut]
        past_cb = cb[cb["reported_at"] <= cut]
        part = OnlineFeatureEngine(SPARKOV).run(past_tx, past_cb).set_index("txn_id")
        pd.testing.assert_frame_equal(part, full.loc[part.index])


def test_events_sharing_a_timestamp_do_not_see_each_other():
    spec = FeatureSpec((Agg("txn", ("card_id",), "count"),))
    eng = OnlineFeatureEngine(spec)
    a = Event(ts=100, amount=500, card_id=1, merchant_id=1, category="x")
    assert eng.process(a, "txn")[0] == 0
    assert eng.process(a._replace(amount=700), "txn")[0] == 0  # same second
    assert eng.process(a._replace(ts=101), "txn")[0] == 2


def test_out_of_order_events_are_rejected():
    eng = OnlineFeatureEngine(SPARKOV)
    e = Event(ts=100, amount=1, card_id=1, merchant_id=1, category="x")
    eng.process(e, "txn")
    with pytest.raises(OutOfOrderEvent):
        eng.process(e._replace(ts=99), "txn")


def test_bounded_state_matches_unbounded_when_capacity_suffices():
    tx, cb = engine_frames(*make_stream(n=200, seed=5))
    a = OnlineFeatureEngine(SPARKOV).run(tx, cb)
    b = OnlineFeatureEngine(SPARKOV, capacity=1_000).run(tx, cb)
    pd.testing.assert_frame_equal(a, b)
