"""HTTP API.

    POST /v1/score        score one authorization request
    POST /v1/chargebacks  report a chargeback (updates online state)
    GET  /v1/model        production model metadata
    GET  /health

Predictions are logged to PostgreSQL by a background thread in batches, so a
slow database never adds latency to the authorization path; if the queue
fills up, log rows are dropped and counted rather than blocking scoring.

Run: `uv run sentinel serve` (warms state up to deploy time, cached on disk).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentinel import db
from sentinel.config import settings
from sentinel.features.online import OutOfOrderEvent
from sentinel.features.spec import SPARKOV
from sentinel.serve.bundle import Bundle, production_path
from sentinel.serve.scorer import Scored, Scorer, UnknownCard

log = logging.getLogger(__name__)


class TransactionIn(BaseModel):
    txn_id: UUID
    ts: datetime
    card_id: int
    merchant_id: int
    category: str
    amount: float = Field(gt=0)
    merch_lat: float
    merch_lon: float


class ChargebackIn(BaseModel):
    txn_id: UUID
    reported_at: datetime
    card_id: int
    merchant_id: int
    category: str
    amount: float = Field(gt=0)


class Reason(BaseModel):
    feature: str
    value: float | None
    contribution: float


class ScoreOut(BaseModel):
    txn_id: UUID
    p_fraud: float
    decision: str
    reasons: list[Reason]
    model_version: str
    latency_ms: float


class PredictionLog:
    """Batching writer: put() never blocks the caller."""

    def __init__(self, model_version: str, batch: int = 2_000, interval: float = 1.0) -> None:
        self.model_version = model_version
        self.batch = batch
        self.interval = interval
        self.dropped = 0
        self.written = 0
        self._q: queue.Queue[Scored] = queue.Queue(maxsize=200_000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="prediction-log", daemon=True)
        self._thread.start()

    def put(self, s: Scored) -> None:
        try:
            self._q.put_nowait(s)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> list[Scored]:
        items = []
        try:
            items.append(self._q.get(timeout=self.interval))
            while len(items) < self.batch:
                items.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return items

    def _write(self, items: list[Scored]) -> None:
        rows = [
            (
                s.txn_id,
                self.model_version,
                s.p_fraud,
                s.decision,
                json.dumps(s.reasons),
                s.latency_ms,
            )
            for s in items
        ]
        with db.connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO predictions (txn_id, model_version, p_fraud, decision, reasons, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )
            conn.commit()
        self.written += len(rows)

    def _loop(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            items = self._drain()
            if items:
                try:
                    self._write(items)
                except Exception:  # logging must never take the service down
                    log.exception("failed to write %d predictions", len(items))
                    self.dropped += len(items)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


def warm_until(at: datetime | None = None) -> pd.Timestamp:
    return pd.Timestamp(at or os.environ.get("SENTINEL_WARM_UNTIL") or settings.timeline.deploy_at)


def snapshot_path(until: pd.Timestamp) -> Path:
    return settings.paths.artifacts / "state" / f"{SPARKOV.version}-{until:%Y%m%dT%H%M%S}.pkl"


def build_scorer(at: datetime | None = None) -> Scorer:
    """Production bundle + online state warmed from history (snapshot cached)."""
    bundle = Bundle.load(production_path())
    p = settings.paths.processed
    scorer = Scorer(bundle, pd.read_parquet(p / "customers.parquet"))
    until = warm_until(at)
    snapshot = snapshot_path(until)
    if snapshot.exists():
        scorer.load_state(snapshot)
    else:
        n = scorer.warm(
            pd.read_parquet(p / "transactions.parquet"),
            pd.read_parquet(p / "chargebacks.parquet"),
            until,
        )
        log.info("warmed online state with %d events up to %s", n, until)
        scorer.save_state(snapshot)
    return scorer


def create_app(scorer: Scorer | None = None, log_predictions: bool = True) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["scorer"] = scorer or build_scorer()
        state["log"] = PredictionLog(state["scorer"].bundle.version) if log_predictions else None
        yield
        if state["log"] is not None:
            state["log"].close()

    app = FastAPI(title="fraud-sentinel", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        s: Scorer = state["scorer"]
        return {"status": "ok", "model": s.bundle.version, "state_clock": s.engine.now}

    @app.get("/v1/model")
    def model() -> dict:
        b = state["scorer"].bundle
        return {
            "version": b.version,
            "feature_spec": b.feature_spec,
            "features": b.features,
            "meta": b.meta,
        }

    @app.post("/v1/score", response_model=ScoreOut)
    def score(txn: TransactionIn) -> ScoreOut:
        s: Scorer = state["scorer"]
        try:
            out = s.score(txn.model_dump())
        except UnknownCard as e:
            raise HTTPException(404, f"unknown card {e.args[0]}") from e
        except OutOfOrderEvent as e:
            raise HTTPException(409, str(e)) from e
        if state["log"] is not None:
            state["log"].put(out)
        return ScoreOut(
            txn_id=txn.txn_id, p_fraud=out.p_fraud, decision=out.decision,
            reasons=[Reason(**r) for r in out.reasons], model_version=s.bundle.version,
            latency_ms=out.latency_ms,
        )  # fmt: skip

    @app.post("/v1/chargebacks", status_code=204)
    def chargeback(cb: ChargebackIn) -> None:
        try:
            state["scorer"].chargeback(cb.model_dump())
        except OutOfOrderEvent as e:
            raise HTTPException(409, str(e)) from e

    return app
