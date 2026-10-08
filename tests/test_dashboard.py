"""The dashboard routes, end to end, on a synthetic stream (no real artifacts)."""

import time
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinel.config import Paths, settings
from sentinel.data.dataset import model_features
from sentinel.decision import CostModel
from sentinel.features.spec import SPARKOV
from sentinel.models.calibration import IsotonicCalibrator
from sentinel.serve import api, demo
from sentinel.serve.bundle import Bundle
from sentinel.serve.scorer import Scorer
from synthetic import make_stream


@pytest.fixture()
def client(tmp_path, monkeypatch):
    tx, cb = make_stream(n=600, seed=2)
    split_at = tx["ts"].iloc[300]
    cards = sorted(tx["card_id"].unique())
    customers = pd.DataFrame(
        {
            "card_id": cards, "gender": "M", "dob": "1985-03-03", "job": "Engineer",
            "city": "Austin", "state": "TX", "zip": "73301", "home_lat": 30.0,
            "home_lon": -97.0, "city_pop": 900_000,
        }
    )  # fmt: skip
    merchants = pd.DataFrame({"merchant_id": sorted(tx["merchant_id"].unique())})
    merchants["name"] = "Shop " + merchants["merchant_id"].astype(str)
    truth = pd.DataFrame({"txn_id": tx["txn_id"], "is_fraud": tx["txn_id"].isin(cb["txn_id"])})

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for name, df in [("transactions", tx), ("chargebacks", cb), ("customers", customers),
                     ("merchants", merchants), ("ground_truth", truth)]:  # fmt: skip
        df.to_parquet(processed / f"{name}.parquet", index=False)
    test_settings = replace(
        settings, paths=Paths(data=tmp_path / "data", artifacts=tmp_path / "artifacts")
    )
    monkeypatch.setattr(demo, "settings", test_settings)
    monkeypatch.setattr(api, "settings", test_settings)
    monkeypatch.setenv("SENTINEL_WARM_UNTIL", split_at.isoformat())

    feats = model_features()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2_000, len(feats)))
    X[:, feats.index("amount")] = rng.lognormal(4, 1.2, 2_000)
    y = (X[:, feats.index("amount")] > 250).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(X, y, feature_name=feats), 20
    )
    cal = IsotonicCalibrator().fit(booster.predict(X), y)
    bundle = Bundle("dash-model", feats, SPARKOV.version, booster, cal, CostModel(), {},
                    reference_scores=cal.predict(booster.predict(X)))  # fmt: skip
    scorer = Scorer(bundle, customers)
    scorer.warm(tx, cb, until=split_at)
    snapshot = api.snapshot_path(api.warm_until())
    scorer.save_state(snapshot)

    with TestClient(api.create_app(scorer=scorer, log_predictions=False, demo=True)) as c:
        yield c, cards


def _wait_for(client, n):
    for _ in range(200):
        state = client.get("/v1/demo/state").json()
        if state["stats"]["transactions"] >= n or not state["running"]:
            return state
        time.sleep(0.05)
    raise AssertionError("replay did not progress")


def test_dashboard_page_and_replay(client):
    c, cards = client
    page = c.get("/")
    assert page.status_code == 200 and "fraud-sentinel" in page.text
    assert c.post("/v1/demo/control", json={"action": "start", "rate": 2000}).json()["running"]
    state = _wait_for(c, 300)
    assert state["stats"]["transactions"] == 300  # whole test half replayed, then it stops
    assert not state["running"] and state["progress"] == 1.0
    assert {i["outcome"] for i in state["items"]} <= {
        "ok", "friction", "false decline", "missed fraud", "fraud in review", "blocked fraud"
    }  # fmt: skip
    newer = c.get("/v1/demo/state", params={"after": state["items"][-1]["seq"]}).json()
    assert newer["items"] == []


def test_manual_scoring_card_view_queue_and_monitoring(client):
    c, cards = client
    big = c.post(
        "/v1/demo/score",
        json={"card_id": int(cards[0]), "merchant_id": 1, "category": "shopping_net",
              "amount": 9_000, "away": True},
    ).json()  # fmt: skip
    assert big["manual"] and big["decision"] in {"review", "decline"} and big["reasons"]
    card = c.get(f"/v1/demo/cards/{int(cards[0])}").json()
    assert card["recent"][0]["txn_id"] == big["txn_id"]
    assert "card__count_24h" in card["features"]
    assert c.get("/v1/demo/cards/123").status_code == 404

    queue = c.get("/v1/demo/queue").json()
    if big["decision"] == "review":
        assert queue[0]["txn_id"] == big["txn_id"]
        assert c.post(f"/v1/demo/queue/{big['txn_id']}", json={"fraud": True}).json()["ok"]
        assert (
            c.get(f"/v1/demo/cards/{int(cards[0])}").json()["features"]["card__cb_count_all"] >= 0
        )
    assert c.post("/v1/demo/queue/nope", json={"fraud": False}).status_code == 404

    m = c.get("/v1/demo/monitoring").json()
    assert len(m["thresholds"]) == len(m["reference"])
    assert c.post("/v1/demo/control", json={"action": "reset"}).status_code == 200
    assert c.get("/v1/demo/state").json()["stats"]["transactions"] == 0
