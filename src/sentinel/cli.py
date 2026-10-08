"""Command-line entry point: `uv run sentinel <command>`.

Heavy imports happen inside each command so `sentinel --help` stays fast.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

Handler = Callable[[argparse.Namespace], None]
COMMANDS: dict[str, tuple[str, list[tuple[tuple[str, ...], dict[str, Any]]], Handler]] = {}


def arg(*flags: str, **kwargs: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    return flags, kwargs


def command(name: str, help_: str, *arguments: tuple[tuple[str, ...], dict[str, Any]]):
    def register(fn: Handler) -> Handler:
        COMMANDS[name] = (help_, list(arguments), fn)
        return fn

    return register


@command("ingest", "Load the Sparkov CSVs into PostgreSQL", arg("--no-reset", action="store_true"))
def _ingest(args: argparse.Namespace) -> None:
    from sentinel.data.sparkov import ingest

    tables = ingest(reset=not args.no_reset)
    for name, frame in tables.items():
        print(f"{name:14s} {len(frame):>10,d}")


@command("features", "Materialize the offline feature table in PostgreSQL")
def _features(args: argparse.Namespace) -> None:
    from sentinel import db
    from sentinel.features.build import load_aggregates, materialize

    with db.connect() as conn:
        seconds = materialize(conn)
    df = load_aggregates(refresh=True)
    print(f"{len(df):,} rows x {df.shape[1] - 2} aggregate features in {seconds:.1f}s")


@command("parity", "Replay all events through the streaming engine and diff against SQL")
def _parity(args: argparse.Namespace) -> None:
    from sentinel.config import settings
    from sentinel.features.parity import check

    report = check()
    out = settings.paths.reports / "parity.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_markdown(), encoding="utf-8")
    print(report.to_markdown())
    if not report.ok:
        raise SystemExit(1)


@command("compare", "Train every model, compare on the test period, run the ablation")
def _compare(args: argparse.Namespace) -> None:
    from sentinel.experiments.compare import run

    print(run())


@command("imbalance", "Compare class-imbalance strategies (weights, resampling, focal loss)")
def _imbalance(args: argparse.Namespace) -> None:
    from sentinel.experiments.imbalance import run

    print(run())


@command("decisions", "Cost out decision policies on the test period")
def _decisions(args: argparse.Namespace) -> None:
    from sentinel.experiments.decisions import run

    print(run())


@command("sketches", "Benchmark Count-Min Sketch against exact pair counts")
def _sketches(args: argparse.Namespace) -> None:
    from sentinel.experiments.sketches import run

    print(run())


@command("tune", "Optuna search with rolling-origin CV", arg("--trials", type=int, default=20))
def _tune(args: argparse.Namespace) -> None:
    from sentinel.experiments.tune import run

    print(run(args.trials))


@command("backtest", "Simulate 18 months of deployment under different retraining policies")
def _backtest(args: argparse.Namespace) -> None:
    from sentinel.experiments.backtest import run

    print(run())


@command("train", "Train, calibrate and register the production model")
def _train(args: argparse.Namespace) -> None:
    from sentinel.serve.train import train_production

    bundle = train_production()
    print(f"production model: {bundle.version}")


@command(
    "serve",
    "Run the scoring API",
    arg("--host", default="127.0.0.1"),
    arg("--port", type=int, default=8000),
    arg("--no-demo", action="store_true", help="API only, without the dashboard"),
)
def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from sentinel.serve.api import create_app

    uvicorn.run(create_app(demo=not args.no_demo), host=args.host, port=args.port)


@command(
    "replay",
    "Stream the test period through the online scorer and check parity/latency",
    arg("--http", type=int, default=0, help="also send the first N transactions over HTTP"),
)
def _replay(args: argparse.Namespace) -> None:
    from sentinel.serve.replay import run

    print(run(http=args.http))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="sentinel")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_, arguments, _) in COMMANDS.items():
        p = sub.add_parser(name, help=help_)
        for flags, kwargs in arguments:
            p.add_argument(*flags, **kwargs)
    args = parser.parse_args(argv)
    COMMANDS[args.command][2](args)


if __name__ == "__main__":
    main()
