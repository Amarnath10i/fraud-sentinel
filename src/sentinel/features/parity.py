"""Full-data check that offline (SQL) and online (streaming) features agree.

Training-serving skew is one of the most common ways a model that looks great
offline quietly underperforms in production. This replays every event through
the streaming engine and compares it with the offline table, value by value.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.config import settings
from sentinel.features.build import load_aggregates
from sentinel.features.events import engine_frames
from sentinel.features.online import OnlineFeatureEngine
from sentinel.features.spec import SPARKOV, FeatureSpec


@dataclass
class ParityReport:
    rows: int
    events: int
    engine_seconds: float
    table: pd.DataFrame

    @property
    def ok(self) -> bool:
        return bool((self.table["mismatches"] == 0).all())

    def to_markdown(self) -> str:
        rate = self.events / self.engine_seconds
        lines = [
            "# Offline/online feature parity",
            "",
            f"Feature spec `{SPARKOV.version}`: {len(self.table)} aggregate features, "
            f"{self.rows:,} transactions, {self.events:,} events (transactions + chargebacks).",
            "",
            f"Streaming replay: {self.engine_seconds:.0f}s ({rate:,.0f} events/s, single thread, pure Python).",
            "",
            f"Result: **{'all features identical' if self.ok else 'MISMATCHES FOUND'}** "
            "(relative tolerance 1e-9; NaN must match NaN).",
            "",
            self.table.to_markdown(index=False, floatfmt=".2e"),
            "",
        ]
        return "\n".join(lines)


def check(spec: FeatureSpec = SPARKOV, rtol: float = 1e-9) -> ParityReport:
    processed = settings.paths.processed
    transactions = pd.read_parquet(processed / "transactions.parquet")
    chargebacks = pd.read_parquet(processed / "chargebacks.parquet")
    tx, cb = engine_frames(transactions, chargebacks)

    t0 = time.perf_counter()
    online = OnlineFeatureEngine(spec).run(tx, cb).set_index("txn_id")
    seconds = time.perf_counter() - t0

    offline = load_aggregates(spec).set_index("txn_id").loc[online.index]
    rows = []
    for name in spec.names:
        a, b = offline[name].to_numpy(), online[name].to_numpy()
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        both = ~nan_a & ~nan_b
        diff = np.abs(a[both] - b[both])
        bad = (nan_a != nan_b).sum() + (diff > rtol * np.maximum(np.abs(a[both]), 1e-12)).sum()
        rows.append(
            {
                "feature": name,
                "null_rate": nan_a.mean(),
                "max_abs_diff": diff.max() if diff.size else 0.0,
                "mismatches": int(bad),
            }
        )
    return ParityReport(len(tx), len(tx) + len(cb), seconds, pd.DataFrame(rows))
