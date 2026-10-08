"""Exact vs approximate structures for (card, merchant) pairs.

"Is this the first time this card pays this merchant?" needs per-pair state.
Exact counting is fine for 1k cards, but pairs grow with cards x merchants, so
a large issuer would approximate. Two candidates, measured on the real stream:

* Count-Min Sketch - approximate counts with additive error eps * N;
* Bloom filter - approximate membership with false-positive rate p.

The question is membership ("count == 0?"), and the result shows why the
choice of structure matters more than its tuning.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sentinel.config import settings
from sentinel.ds import BloomFilter, CountMinSketch
from sentinel.experiments.common import write_report


def _dict_bytes(d: dict) -> int:
    return sys.getsizeof(d) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())


def run(delta: float = 0.01) -> str:
    tx = pd.read_parquet(
        settings.paths.processed / "transactions.parquet", columns=["ts", "card_id", "merchant_id"]
    ).sort_values("ts", kind="stable")
    keys = list(zip(tx["card_id"].tolist(), tx["merchant_id"].tolist(), strict=True))
    n = len(keys)

    exact: dict[tuple[int, int], int] = {}
    truly_new = np.empty(n, dtype=bool)
    for i, k in enumerate(keys):
        c = exact.get(k, 0)
        truly_new[i] = c == 0
        exact[k] = c + 1
    exact_mb = _dict_bytes(exact) / 1e6

    rows = [
        {
            "structure": "exact dict",
            "parameters": "-",
            "memory": f"{exact_mb:.1f} MB",
            "first-time pairs detected": "1.0000",
            "false 'new' flags": 0,
            "mean overcount": "0",
        }
    ]
    for eps in (1e-3, 1e-4, 1e-5):
        cms = CountMinSketch.from_error(eps, delta)
        flagged = np.empty(n, dtype=bool)
        for i, k in enumerate(keys):
            flagged[i] = cms.estimate(k) == 0
            cms.add(k)
        err = np.array([cms.estimate(k) - c for k, c in exact.items()])
        rows.append(
            {
                "structure": "Count-Min Sketch",
                "parameters": f"eps={eps:g}, {cms.width:,} x {cms.depth}",
                "memory": f"{cms.nbytes / 1e6:.2f} MB",
                "first-time pairs detected": f"{np.mean(flagged[truly_new]):.4f}",
                "false 'new' flags": int(np.sum(flagged & ~truly_new)),
                "mean overcount": f"{err.mean():.2f}",
            }
        )
    capacity = len(exact)
    for p in (0.01, 0.001):
        bf = BloomFilter.for_capacity(capacity, p)
        flagged = np.empty(n, dtype=bool)
        for i, k in enumerate(keys):
            flagged[i] = k not in bf
            bf.add(k)
        rows.append(
            {
                "structure": "Bloom filter",
                "parameters": f"p={p:g}, {bf.m:,} bits, k={bf.k}",
                "memory": f"{bf.nbytes / 1e6:.2f} MB",
                "first-time pairs detected": f"{np.mean(flagged[truly_new]):.4f}",
                "false 'new' flags": int(np.sum(flagged & ~truly_new)),
                "mean overcount": "n/a",
            }
        )

    text = "\n".join(
        [
            "# First-time (card, merchant) pairs: exact vs Count-Min vs Bloom",
            "",
            f"{n:,} transactions, {len(exact):,} distinct pairs, so {truly_new.mean():.1%} of "
            "transactions are a card's first payment to that merchant.",
            "",
            pd.DataFrame(rows).to_markdown(index=False),
            "",
            "Neither structure ever flags a known pair as new (no false negatives on "
            "membership). The difference is how many genuinely new pairs slip through:",
            "",
            "- Count-Min's error is additive, eps x N over the whole stream. With an average "
            f"of {n / len(exact):.1f} transactions per pair, a long tail of rare keys sits far "
            "below that error, so 'count == 0' is unreliable unless the sketch is almost as "
            "large as the exact table. It is the right tool for heavy hitters, not for novelty.",
            "- A Bloom filter answers membership directly: at 1% false positives it needs "
            "~9.6 bits per pair and catches ~99% of first-time pairs in a fraction of the "
            "exact table's memory.",
            "",
        ]
    )
    write_report("sketches", text)
    return text
