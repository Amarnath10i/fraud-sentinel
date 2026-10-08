"""Keep the k highest-priority items of a stream with a size-k min-heap.

Used for the fraud review queue: analysts can only look at k cases a day, so
the queue keeps the k cases with the highest expected loss. The heap root is
the weakest item currently kept, so a new item either beats it (replace root,
O(log k)) or is dropped in O(1).
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Generic, TypeVar

T = TypeVar("T")


class TopK(Generic[T]):
    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
        self._heap: list[tuple[float, int, T]] = []
        self._tiebreak = count()  # FIFO among equal priorities; never compares T

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, priority: float, item: T) -> T | None:
        """Offer an item. Returns whatever fell out of the top k, if anything."""
        if self.k == 0:
            return item
        entry = (priority, -next(self._tiebreak), item)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return None
        if entry > self._heap[0]:
            return heapq.heapreplace(self._heap, entry)[2]
        return item

    def items(self) -> list[tuple[float, T]]:
        """Kept items, highest priority first."""
        return [(p, x) for p, _, x in sorted(self._heap, reverse=True)]

    def threshold(self) -> float | None:
        """Lowest priority currently kept (the bar a new item must beat)."""
        return self._heap[0][0] if len(self._heap) == self.k and self.k else None
