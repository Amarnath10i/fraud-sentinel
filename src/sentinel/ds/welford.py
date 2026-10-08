"""Welford's online mean and variance, with Chan et al.'s parallel merge.

The textbook formula var = E[x^2] - E[x]^2 subtracts two large, nearly equal
numbers and loses most of its significant digits when the mean is large
compared with the spread. Welford updates the mean and the sum of squared
deviations (M2) incrementally and stays accurate in a single pass.
"""

from __future__ import annotations

import math


class Welford:
    __slots__ = ("m2", "mean", "n")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def merge(self, other: Welford) -> Welford:
        """Combine two accumulators as if one had seen both streams."""
        out = Welford()
        n = self.n + other.n
        if n == 0:
            return out
        delta = other.mean - self.mean
        out.n = n
        out.mean = self.mean + delta * other.n / n
        out.m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n
        return out

    @property
    def var_pop(self) -> float:
        return self.m2 / self.n if self.n else math.nan

    @property
    def var_sample(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else math.nan

    @property
    def std_pop(self) -> float:
        return math.sqrt(self.var_pop) if self.n else math.nan
