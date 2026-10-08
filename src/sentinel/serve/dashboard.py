"""Dashboard routes: the single-page UI at / and the JSON it polls."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sentinel import db

STATIC = Path(__file__).parent / "static"


class Control(BaseModel):
    action: str = Field(pattern="^(start|pause|reset|rate)$")
    rate: float | None = None


class Resolve(BaseModel):
    fraud: bool


class ManualTxn(BaseModel):
    card_id: int
    merchant_id: int
    category: str
    amount: float = Field(gt=0, le=100_000)
    away: bool = False


def attach(app: FastAPI, state: dict) -> None:
    def demo():
        return state["demo"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/v1/demo/state")
    def demo_state(after: int = 0) -> dict:
        return demo().view(after)

    @app.post("/v1/demo/control")
    def demo_control(c: Control) -> dict:
        if c.action == "reset":
            demo().reset(state["snapshot"])
        else:
            demo().control(c.action if c.action != "rate" else "", c.rate)
        return {"running": demo().running, "rate": demo().rate}

    @app.get("/v1/demo/options")
    def demo_options() -> dict:
        return demo().options()

    @app.get("/v1/demo/cards/{card_id}")
    def demo_card(card_id: int) -> dict:
        try:
            return demo().card(card_id)
        except KeyError as e:
            raise HTTPException(404, f"unknown card {card_id}") from e

    @app.get("/v1/demo/queue")
    def demo_queue() -> list[dict]:
        return demo().queue_view()

    @app.post("/v1/demo/queue/{txn_id}")
    def demo_resolve(txn_id: str, r: Resolve) -> dict:
        try:
            demo().resolve(txn_id, r.fraud)
        except KeyError as e:
            raise HTTPException(404, "not in the review queue") from e
        return {"ok": True}

    @app.post("/v1/demo/score")
    def demo_score(t: ManualTxn) -> dict:
        try:
            return demo().score_manual(t.card_id, t.merchant_id, t.category, t.amount, t.away)
        except KeyError as e:
            raise HTTPException(404, f"unknown card {t.card_id}") from e

    @app.get("/v1/demo/monitoring")
    def demo_monitoring() -> dict:
        out = demo().monitoring()
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT version, created_at, stage, metrics FROM model_registry "
                    "ORDER BY created_at DESC LIMIT 10"
                ).fetchall()
            out["registry"] = [
                {"version": v, "created_at": c.isoformat(), "stage": s, "metrics": m}
                for v, c, s, m in rows
            ]
        except Exception:  # the dashboard should still render without the database
            out["registry"] = []
        return out
