"""API contract tests on a tiny model trained on synthetic data (no real artifacts needed)."""

import uuid

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinel.data.dataset import model_features
from sentinel.decision import CostModel
from sentinel.features.spec import SPARKOV
from sentinel.models.calibration import IsotonicCalibrator
from sentinel.serve.api import create_app
from sentinel.serve.bundle import Bundle
from sentinel.serve.scorer import Scorer
from synthetic import make_stream


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    feats = model_features()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2_000, len(feats)))
    X[:, feats.index("amount")] = rng.lognormal(4, 1, 2_000)
    y = (X[:, feats.index("amount")] > 300).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(X, y, feature_name=feats), 20
    )
    cal = IsotonicCalibrator().fit(booster.predict(X), y)
    bundle = Bundle("test-model", feats, SPARKOV.version, booster, cal, CostModel(), {})
    # round-trip through disk like a real deployment
    bundle = Bundle.load(bundle.save(tmp_path_factory.mktemp("models")))
    tx, cb = make_stream(n=200, seed=1)
    customers = pd.DataFrame(
        {
            "card_id": sorted(tx["card_id"].unique()),
            "gender": "F",
            "dob": "1980-01-01",
            "city_pop": 1000,
            "home_lat": 35.0,
            "home_lon": -100.0,
        }
    )
    s = Scorer(bundle, customers)
    s.warm(tx, cb, until=tx["ts"].iloc[-1])
    return s, tx["ts"].iloc[-1]


def _txn(ts, **kw):
    body = {
        "txn_id": uuid.uuid4().hex,  # every call is a new transaction unless told otherwise
        "ts": ts.isoformat(),
        "card_id": 1,
        "merchant_id": 2,
        "category": "misc_net",
        "amount": 25.0,
        "merch_lat": 35.1,
        "merch_lon": -100.2,
    }
    return body | kw


def test_score_endpoint_contract(scorer):
    s, last = scorer
    with TestClient(create_app(scorer=s, log_predictions=False)) as client:
        assert client.get("/health").json()["status"] == "ok"
        small = client.post("/v1/score", json=_txn(last + pd.Timedelta(minutes=1))).json()
        assert (
            0 <= small["p_fraud"] <= 1 and small["decision"] == "approve" and small["reasons"] == []
        )
        big = client.post(
            "/v1/score", json=_txn(last + pd.Timedelta(minutes=2), amount=5_000.0)
        ).json()
        assert big["decision"] in {"review", "decline"}
        assert big["reasons"] and big["reasons"][0]["feature"] == "amount"
        assert big["model_version"] == "test-model"


def test_errors_are_explicit(scorer):
    s, last = scorer
    with TestClient(create_app(scorer=s, log_predictions=False)) as client:
        t = last + pd.Timedelta(hours=1)
        assert client.post("/v1/score", json=_txn(t, card_id=999_999)).status_code == 404
        assert client.post("/v1/score", json=_txn(t)).status_code == 200
        assert client.post("/v1/score", json=_txn(t - pd.Timedelta(days=1))).status_code == 409
        assert client.post("/v1/score", json=_txn(t, amount=-5)).status_code == 422
        cb = {k: v for k, v in _txn(t).items() if k not in ("ts", "merch_lat", "merch_lon")}
        assert (
            client.post("/v1/chargebacks", json=cb | {"reported_at": t.isoformat()}).status_code
            == 204
        )


def test_online_scorer_equals_offline_pipeline(tmp_path):
    """Score a synthetic stream event by event and compare with the batch path:
    engine.run -> row_features -> derived_features -> model."""
    from sentinel.features.events import engine_frames, epoch_seconds
    from sentinel.features.online import OnlineFeatureEngine
    from sentinel.features.rows import derived_features, row_features

    feats = model_features()
    tx, cb = make_stream(n=300, seed=3)
    customers = pd.DataFrame(
        {
            "card_id": sorted(tx["card_id"].unique()),
            "gender": ["F", "M", "F", "M", "F"][: tx["card_id"].nunique()],
            "dob": "1975-05-05",
            "city_pop": 5000,
            "home_lat": 36.0,
            "home_lon": -99.0,
        }
    )
    # offline: batch features for every transaction
    etx, ecb = engine_frames(tx, cb)
    agg = OnlineFeatureEngine(SPARKOV).run(etx, ecb)
    merged = (
        tx.assign(txn_id=tx["txn_id"].astype(str))
        .merge(agg, on="txn_id")
        .merge(customers, on="card_id")
    )
    dob_days = (pd.to_datetime(merged["dob"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    rows = row_features(
        {
            "ts": epoch_seconds(merged["ts"]), "amount": merged["amount"].to_numpy(),
            "category": merged["category"].to_numpy(), "gender": merged["gender"].to_numpy(),
            "dob_days": dob_days, "city_pop": merged["city_pop"].to_numpy(),
            "home_lat": merged["home_lat"].to_numpy(), "home_lon": merged["home_lon"].to_numpy(),
            "merch_lat": merged["merch_lat"].to_numpy(), "merch_lon": merged["merch_lon"].to_numpy(),
        }
    )  # fmt: skip
    aggs = {n: merged[n].to_numpy(np.float64) for n in SPARKOV.names}
    values = rows | aggs | derived_features(rows["amount"], aggs)
    X = np.column_stack([values[f] for f in feats])

    rng = np.random.default_rng(1)
    y = (rng.random(len(X)) < 0.2).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(X, y, feature_name=feats), 30,
    )  # fmt: skip
    cal = IsotonicCalibrator().fit(booster.predict(X), y)
    bundle = Bundle("parity", feats, SPARKOV.version, booster, cal, CostModel(), {})
    offline = pd.Series(bundle.predict(X), index=merged["txn_id"])

    # online: one event at a time, chargebacks interleaved by time
    scorer = Scorer(bundle, customers)
    events = [(t, 0, r) for t, r in zip(tx["ts"], tx.to_dict("records"), strict=True)]
    cbr = cb.merge(tx[["txn_id", "card_id", "merchant_id", "category", "amount"]], on="txn_id")
    events += [(t, 1, r) for t, r in zip(cbr["reported_at"], cbr.to_dict("records"), strict=True)]
    online = {}
    for _, kind, r in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == 0:
            online[str(r["txn_id"])] = scorer.score(r).p_fraud
        else:
            scorer.chargeback(r)
    online = pd.Series(online)
    np.testing.assert_allclose(online.loc[offline.index].to_numpy(), offline.to_numpy(), atol=1e-12)


def test_retries_are_idempotent_and_do_not_inflate_velocity(scorer):
    from sentinel.features.online import Event

    s, last = scorer
    with TestClient(create_app(scorer=s, log_predictions=False)) as client:
        t = last + pd.Timedelta(hours=3)
        body = _txn(t, txn_id="ab" * 16, card_id=2, amount=40.0)
        first = client.post("/v1/score", json=body).json()
        # same transaction again, id written with dashes: a network retry
        retry = client.post(
            "/v1/score", json=body | {"txn_id": "abababab-abab-abab-abab-abababababab"}
        ).json()
        assert not first["duplicate"] and retry["duplicate"]
        assert (retry["p_fraud"], retry["decision"]) == (first["p_fraud"], first["decision"])
        # the next transaction on the card sees one earlier transaction in the last hour, not two
        later = int((t + pd.Timedelta(minutes=5)).timestamp())
        features = s.engine.process(Event(later, 1_000, 2, 1, "misc_net"), "txn")
        assert features[SPARKOV.names.index("card__count_1h")] == 1
