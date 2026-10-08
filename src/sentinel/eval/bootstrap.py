"""Cluster bootstrap confidence intervals.

Transactions are not independent: a compromised card produces a burst of
fraudulent transactions within a day or two. Resampling individual rows would
treat those as independent evidence and produce intervals that are too narrow,
so resampling is done over cards (all of a card's transactions move together).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MetricFn = Callable[[np.ndarray], dict[str, float]]  # receives row indices


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


class ClusterBootstrap:
    def __init__(self, clusters: np.ndarray, n_boot: int = 300, seed: int = 0) -> None:
        clusters = np.asarray(clusters)
        order = np.argsort(clusters, kind="mergesort")
        _, starts = np.unique(clusters[order], return_index=True)
        self._groups = np.split(order, starts[1:])
        self.n_boot = n_boot
        self.seed = seed

    def samples(self):
        rng = np.random.default_rng(self.seed)
        n = len(self._groups)
        for _ in range(self.n_boot):
            pick = rng.integers(0, n, n)
            yield np.concatenate([self._groups[i] for i in pick])

    def intervals(self, fn: MetricFn, n_rows: int, alpha: float = 0.05) -> dict[str, Interval]:
        point = fn(np.arange(n_rows))
        draws = [fn(idx) for idx in self.samples()]
        out = {}
        for k, v in point.items():
            vals = np.array([d[k] for d in draws], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
            out[k] = Interval(v, float(lo), float(hi))
        return out

    def paired_difference(
        self, fn_a: MetricFn, fn_b: MetricFn, n_rows: int, alpha: float = 0.05
    ) -> dict[str, tuple[Interval, float]]:
        """B minus A on the same resamples, with the share of resamples where B <= A."""
        pa, pb = fn_a(np.arange(n_rows)), fn_b(np.arange(n_rows))
        diffs: dict[str, list[float]] = {k: [] for k in pa}
        for idx in self.samples():
            a, b = fn_a(idx), fn_b(idx)
            for k in pa:
                diffs[k].append(b[k] - a[k])
        out = {}
        for k, vals in diffs.items():
            arr = np.array(vals)
            lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
            out[k] = (Interval(pb[k] - pa[k], float(lo), float(hi)), float(np.mean(arr <= 0)))
        return out
