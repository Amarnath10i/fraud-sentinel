"""A deployable model: everything needed to turn a transaction into a decision.

A bundle is a directory with the LightGBM model, the isotonic calibrator, the
feature list, the feature-spec version it was trained against and the cost
model. Bundles are registered in PostgreSQL (`model_registry`), and promoting
one to production archives the previous one in the same transaction; a partial
unique index guarantees there is never more than one production model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from psycopg.types.json import Jsonb

from sentinel import db
from sentinel.config import settings
from sentinel.decision import CostModel
from sentinel.models.calibration import IsotonicCalibrator

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    version: str
    features: list[str]
    feature_spec: str
    booster: lgb.Booster
    calibrator: IsotonicCalibrator
    costs: CostModel
    meta: dict
    # calibrated validation scores: the reference distribution for score-drift monitoring
    reference_scores: np.ndarray | None = None

    def save(self, root: Path | None = None) -> Path:
        path = (root or settings.paths.artifacts / "models") / self.version
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        np.savez(path / "calibrator.npz", x=self.calibrator.x_, y=self.calibrator.y_)
        if self.reference_scores is not None:
            np.save(path / "reference_scores.npy", self.reference_scores)
        manifest = {
            "version": self.version,
            "features": self.features,
            "feature_spec": self.feature_spec,
            "costs": asdict(self.costs),
            "meta": self.meta,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> Bundle:
        manifest = json.loads((path / "manifest.json").read_text())
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        cal = IsotonicCalibrator()
        with np.load(path / "calibrator.npz") as z:
            cal.x_, cal.y_ = z["x"], z["y"]
        ref = path / "reference_scores.npy"
        return cls(
            version=manifest["version"],
            features=manifest["features"],
            feature_spec=manifest["feature_spec"],
            booster=booster,
            calibrator=cal,
            costs=CostModel(**manifest["costs"]),
            meta=manifest["meta"],
            reference_scores=np.load(ref) if ref.exists() else None,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calibrated fraud probability.

        Small batches predict on one thread: spinning up LightGBM's OpenMP pool
        for a single row costs far more than the prediction itself.
        """
        threads = 1 if len(X) < 1_000 else 0
        return self.calibrator.predict(self.booster.predict(X, num_threads=threads))


def new_version(prefix: str = "lgbm") -> str:
    return f"{prefix}-{datetime.now(UTC):%Y%m%d-%H%M%S}"


def register(bundle: Bundle, path: Path, promote: bool = True) -> None:
    m = bundle.meta
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO model_registry (version, algorithm, feature_set, train_start, train_end,
                label_cutoff, params, metrics, policy, artifact_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                bundle.version, "lightgbm+isotonic", bundle.feature_spec, m["train_start"],
                m["train_end"], m["label_cutoff"], Jsonb(m.get("params", {})),
                Jsonb(m.get("metrics", {})), Jsonb(asdict(bundle.costs)), str(path),
            ),
        )  # fmt: skip
        if promote:
            conn.execute("UPDATE model_registry SET stage = 'archived' WHERE stage = 'production'")
            conn.execute(
                "UPDATE model_registry SET stage = 'production' WHERE version = %s",
                (bundle.version,),
            )
        conn.commit()
    log.info("registered %s%s", bundle.version, " as production" if promote else "")


REGISTRY_COLUMNS = [
    "version", "created_at", "stage", "algorithm", "feature_set", "train_start", "train_end",
    "label_cutoff", "params", "metrics",
]  # fmt: skip


def registry_rows() -> list[dict]:
    """Model registry, newest first.

    Read from PostgreSQL; a deployment without a database (the public demo)
    ships `registry.json`, exported with `sentinel export-demo`, instead.
    """
    snapshot = settings.paths.artifacts / "registry.json"
    if not db.enabled():
        return json.loads(snapshot.read_text()) if snapshot.exists() else []
    try:
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(REGISTRY_COLUMNS)} FROM model_registry ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                c: (v.isoformat() if isinstance(v, datetime) else v)
                for c, v in zip(REGISTRY_COLUMNS, r, strict=True)
            }
            for r in rows
        ]
    except Exception:
        return json.loads(snapshot.read_text()) if snapshot.exists() else []


def production_path() -> Path:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT artifact_path FROM model_registry WHERE stage = 'production'"
        ).fetchone()
    if row is None:
        raise LookupError("no production model registered; run `sentinel train` first")
    return Path(row[0])
