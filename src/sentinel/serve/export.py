"""Package everything the public demo needs, without PostgreSQL or the full dataset.

The demo replays the deployment period, so it only needs:

* transactions from shortly before deploy time onwards (chargebacks reported
  during the replay refer back to transactions up to a few months earlier),
  with their ground truth for display;
* the online state snapshot at deploy time (all history before it, already
  folded into the streaming engine);
* the production model bundle and a snapshot of the registry.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from sentinel.config import settings
from sentinel.serve.api import snapshot_path, warm_until
from sentinel.serve.bundle import production_path, registry_rows

log = logging.getLogger(__name__)


def export_demo(out: Path, history_days: int = 120) -> dict[str, int]:
    p = settings.paths.processed
    start = pd.Timestamp(settings.timeline.deploy_at)

    tx = pd.read_parquet(p / "transactions.parquet")
    cb = pd.read_parquet(p / "chargebacks.parquet")
    cb = cb[cb["reported_at"] >= start]
    keep = (tx["ts"] >= start - pd.Timedelta(days=history_days)) | tx["txn_id"].isin(cb["txn_id"])
    tx = tx[keep]
    truth = pd.read_parquet(p / "ground_truth.parquet")
    truth = truth[truth["txn_id"].isin(tx["txn_id"])]

    processed = out / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frames = {
        "transactions": tx,
        "chargebacks": cb,
        "ground_truth": truth,
        "customers": pd.read_parquet(p / "customers.parquet"),
        "merchants": pd.read_parquet(p / "merchants.parquet"),
    }
    for name, df in frames.items():
        df.to_parquet(processed / f"{name}.parquet", index=False)

    artifacts = out / "artifacts"
    shutil.copytree(production_path(), artifacts / "models" / "production", dirs_exist_ok=True)
    snapshot = snapshot_path(warm_until())
    if not snapshot.exists():
        raise FileNotFoundError(f"{snapshot} missing: start `sentinel serve` once to build it")
    (artifacts / "state").mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, artifacts / "state" / snapshot.name)
    (artifacts / "registry.json").write_text(json.dumps(registry_rows(), indent=2, default=str))

    sizes = {str(f.relative_to(out)): f.stat().st_size for f in out.rglob("*") if f.is_file()}
    log.info("demo pack: %d files, %.1f MB", len(sizes), sum(sizes.values()) / 1e6)
    return sizes
