"""Project-wide settings.

Everything time-related lives in `Timeline` so that every experiment uses the
same point-in-time boundaries. Paths and the database URL can be overridden
with environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


@dataclass(frozen=True)
class Paths:
    data: Path = Path(os.environ.get("SENTINEL_DATA_DIR", ROOT / "data"))
    artifacts: Path = Path(os.environ.get("SENTINEL_ARTIFACTS_DIR", ROOT / "artifacts"))
    reports: Path = ROOT / "reports"
    sql: Path = ROOT / "sql"

    @property
    def raw_sparkov(self) -> Path:
        return self.data / "raw" / "sparkov"

    @property
    def processed(self) -> Path:
        return self.data / "processed"


@dataclass(frozen=True)
class ChargebackModel:
    """How long it takes for a fraudulent transaction to be reported.

    Real chargebacks arrive days to weeks after the transaction. Delays are
    drawn from a log-normal distribution so that most reports come within a
    couple of weeks and a thin tail arrives after the maturity window, which
    is exactly the label noise a production system has to live with.
    """

    median_days: float = 12.0
    sigma: float = 0.7
    min_days: float = 1.0
    seed: int = 7


@dataclass(frozen=True)
class Timeline:
    """Point-in-time boundaries for the main experiment.

    The model is "deployed" at `deploy_at`, which is exactly where the original
    Kaggle test file begins. Only labels reported before `deploy_at` may be used
    for anything (training, early stopping, calibration, thresholds), and a
    transaction counts as a confirmed non-fraud only after `maturity` has passed
    without a chargeback.
    """

    data_start: datetime = _utc("2019-01-01")
    train_end: datetime = _utc("2020-02-01")
    deploy_at: datetime = _utc("2020-06-21T12:14:00")
    data_end: datetime = _utc("2021-01-01")
    maturity: timedelta = timedelta(days=60)

    @property
    def valid_end(self) -> datetime:
        """Last transaction time whose label has matured by deploy time."""
        return self.deploy_at - self.maturity


@dataclass(frozen=True)
class Settings:
    db_url: str = os.environ.get("SENTINEL_DB_URL", "postgresql://postgres@localhost:5432/sentinel")
    paths: Paths = field(default_factory=Paths)
    timeline: Timeline = field(default_factory=Timeline)
    chargebacks: ChargebackModel = field(default_factory=ChargebackModel)
    seed: int = 42


settings = Settings()
