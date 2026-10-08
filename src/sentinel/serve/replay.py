"""Replay the deployment period through the online scorer.

Streams every test-period transaction and chargeback in time order through
`Scorer` (in-process), exactly as the API would see them, and checks three
things:

1. end-to-end parity: the served probability for each transaction equals the
   probability computed offline from the SQL feature table;
2. latency of the full online path per transaction (p50/p95/p99);
3. that every scored transaction landed in the PostgreSQL prediction log.

Optionally sends a sample through the real HTTP API as well (`--http N`).
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import numpy as np
import pandas as pd
import uvicorn

from sentinel import db
from sentinel.config import settings
from sentinel.data.dataset import load_frame, window
from sentinel.decision import ACTION_NAMES, CostModel, bayes_policy
from sentinel.experiments.common import write_report
from sentinel.serve.api import PredictionLog, build_scorer, create_app

log = logging.getLogger(__name__)


def _events(start: pd.Timestamp) -> list[tuple[str, dict]]:
    p = settings.paths.processed
    tx = pd.read_parquet(p / "transactions.parquet")
    cb = pd.read_parquet(p / "chargebacks.parquet")
    tx = tx[tx["ts"] >= start].assign(txn_id=lambda d: d["txn_id"].astype(str))
    cb = cb[cb["reported_at"] >= start].merge(
        pd.read_parquet(p / "transactions.parquet")[
            ["txn_id", "card_id", "merchant_id", "category", "amount"]
        ],
        on="txn_id",
    )
    events = [(t, 0, r) for t, r in zip(tx["ts"], tx.to_dict("records"), strict=True)]
    events += [(t, 1, r) for t, r in zip(cb["reported_at"], cb.to_dict("records"), strict=True)]
    events.sort(key=lambda e: (e[0], e[1]))
    return [("txn" if k == 0 else "cb", r) for _, k, r in events]


def _http_sample(events, n: int) -> np.ndarray:
    scorer = build_scorer()
    app = create_app(scorer=scorer, log_predictions=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    lat = []
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=10) as client:
        sent = 0
        for kind, r in events:
            if kind == "txn":
                body = {
                    k: r[k]
                    for k in (
                        "txn_id",
                        "card_id",
                        "merchant_id",
                        "category",
                        "amount",
                        "merch_lat",
                        "merch_lon",
                    )
                }
                body["ts"] = r["ts"].isoformat()
                t0 = time.perf_counter()
                client.post("/v1/score", json=body).raise_for_status()
                lat.append((time.perf_counter() - t0) * 1e3)
                sent += 1
                if sent >= n:
                    break
            else:
                body = {k: r[k] for k in ("txn_id", "card_id", "merchant_id", "category", "amount")}
                body["reported_at"] = r["reported_at"].isoformat()
                client.post("/v1/chargebacks", json=body).raise_for_status()
    server.should_exit = True
    thread.join(timeout=10)
    return np.array(lat)


def run(http: int = 0, log_to_db: bool = True) -> str:
    start = pd.Timestamp(settings.timeline.deploy_at)
    scorer = build_scorer(start)
    bundle = scorer.bundle
    events = _events(start)
    n_tx = sum(1 for k, _ in events if k == "txn")
    log.info("replaying %d events (%d transactions)", len(events), n_tx)

    with db.connect() as conn:
        conn.execute("DELETE FROM predictions WHERE model_version = %s", (bundle.version,))
        conn.commit()
    plog = PredictionLog(bundle.version) if log_to_db else None

    ids, probs, decisions, lat = [], [], [], []
    t0 = time.perf_counter()
    for kind, r in events:
        if kind == "txn":
            s = scorer.score(r)
            ids.append(s.txn_id)
            probs.append(s.p_fraud)
            decisions.append(s.decision)
            lat.append(s.latency_ms)
            if plog is not None:
                plog.put(s)
        else:
            scorer.chargeback(r)
    wall = time.perf_counter() - t0
    if plog is not None:
        plog.close()

    online = pd.DataFrame({"txn_id": ids, "p_online": probs, "decision": decisions})
    test = window(load_frame(), start, settings.timeline.data_end)
    offline = bundle.predict(test[bundle.features].to_numpy(np.float64))
    joined = online.merge(
        test[["txn_id", "is_fraud", "amount"]].assign(p_offline=offline), on="txn_id"
    )
    gap = np.abs(joined["p_online"] - joined["p_offline"])
    offline_decisions = np.array(ACTION_NAMES)[
        bayes_policy(joined["p_offline"], joined["amount"], CostModel())
    ]
    decision_mismatch = int((offline_decisions != joined["decision"].to_numpy()).sum())

    with db.connect() as conn:
        logged = conn.execute(
            "SELECT count(*) FROM predictions WHERE model_version = %s", (bundle.version,)
        ).fetchone()[0]
    lat = np.array(lat)
    counts = joined["decision"].value_counts()
    lines = [
        "# Online replay of the deployment period",
        "",
        f"Model `{bundle.version}`; state warmed with all events before {start:%Y-%m-%d %H:%M} UTC, "
        f"then {len(events):,} events ({n_tx:,} transactions, {len(events) - n_tx:,} chargebacks) "
        "streamed in time order through the in-process scorer.",
        "",
        "## End-to-end parity (served vs offline probability)",
        "",
        f"- transactions compared: {len(joined):,}",
        f"- max |p_online - p_offline|: {gap.max():.2e}",
        f"- decisions that differ: {decision_mismatch}",
        "",
        "## Latency of the online path (feature engine + model + calibration + policy + reasons)",
        "",
        "| p50 | p95 | p99 | max | throughput |",
        "|---|---|---|---|---|",
        f"| {np.percentile(lat, 50):.3f} ms | {np.percentile(lat, 95):.3f} ms | "
        f"{np.percentile(lat, 99):.3f} ms | {lat.max():.1f} ms | {n_tx / wall:,.0f} txn/s (single thread) |",
        "",
        "## Decisions",
        "",
        *[f"- {d}: {counts.get(d, 0):,}" for d in ACTION_NAMES],
        "",
        f"Prediction log: {logged:,} rows in PostgreSQL"
        + (f", {plog.dropped} dropped" if plog is not None else "")
        + ".",
        "",
    ]
    if http:
        h = _http_sample(events, http)
        lines += [
            f"## Over HTTP (first {len(h):,} transactions, FastAPI + uvicorn, localhost)",
            "",
            "| p50 | p95 | p99 |",
            "|---|---|---|",
            f"| {np.percentile(h, 50):.2f} ms | {np.percentile(h, 95):.2f} ms | {np.percentile(h, 99):.2f} ms |",
            "",
        ]
    text = "\n".join(lines)
    write_report("replay", text)
    return text
