"""Time-based sliding windows over an event stream.

Every window answers questions about events with timestamp in [now - span, now).
The caller first `advance(now)` to evict expired events, reads the aggregate,
then `add(ts, value)` the current event. Each event is appended once and
evicted once, so all operations are O(1) amortized.

`span=None` means "all history": nothing is ever evicted and no events are kept.

Sums are maintained by adding on insert and subtracting on eviction. With
floats that slowly accumulates rounding error, so callers feed integers
(money in cents) and the sums stay exact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable


class SlidingWindowSum:
    """Count, sum and sum of squares of integer values in the window."""

    __slots__ = ("_events", "count", "span", "total", "total_sq")

    def __init__(self, span: int | None) -> None:
        self.span = span
        self._events: deque[tuple[int, int]] | None = deque() if span is not None else None
        self.count = 0
        self.total = 0
        self.total_sq = 0

    def advance(self, now: int) -> None:
        if self._events is None:
            return
        cutoff = now - self.span
        events = self._events
        while events and events[0][0] < cutoff:
            _, v = events.popleft()
            self.count -= 1
            self.total -= v
            self.total_sq -= v * v

    def add(self, ts: int, value: int) -> None:
        if self._events is not None:
            self._events.append((ts, value))
        self.count += 1
        self.total += value
        self.total_sq += value * value

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    def var_pop(self) -> float | None:
        """Population variance, computed exactly from integer moments."""
        n = self.count
        if n == 0:
            return None
        return (n * self.total_sq - self.total * self.total) / (n * n)


class SlidingWindowMax:
    """Maximum over the window using a monotonic deque.

    The deque holds (ts, value) with strictly decreasing values. A new value
    pops every smaller-or-equal value from the back: those can never be the
    maximum again because the new one is both larger and expires later.
    """

    __slots__ = ("_dq", "span")

    def __init__(self, span: int | None) -> None:
        self.span = span
        self._dq: deque[tuple[int, float]] = deque()

    def advance(self, now: int) -> None:
        if self.span is None:
            return
        cutoff = now - self.span
        dq = self._dq
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def add(self, ts: int, value: float) -> None:
        dq = self._dq
        while dq and dq[-1][1] <= value:
            dq.pop()
        dq.append((ts, value))

    @property
    def max(self) -> float | None:
        return self._dq[0][1] if self._dq else None


class SlidingWindowDistinct:
    """Exact number of distinct keys in the window (deque + reference counts)."""

    __slots__ = ("_counts", "_events", "span")

    def __init__(self, span: int | None) -> None:
        self.span = span
        self._events: deque[tuple[int, Hashable]] = deque()
        self._counts: dict[Hashable, int] = {}

    def advance(self, now: int) -> None:
        if self.span is None:
            return
        cutoff = now - self.span
        events, counts = self._events, self._counts
        while events and events[0][0] < cutoff:
            _, key = events.popleft()
            c = counts[key] - 1
            if c:
                counts[key] = c
            else:
                del counts[key]

    def add(self, ts: int, key: Hashable) -> None:
        if self.span is not None:
            self._events.append((ts, key))
        self._counts[key] = self._counts.get(key, 0) + 1

    @property
    def distinct(self) -> int:
        return len(self._counts)
