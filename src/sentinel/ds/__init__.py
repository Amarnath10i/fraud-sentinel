"""Data structures behind the streaming feature engine.

All of them are written from scratch and tested against brute-force references
(see tests/test_ds.py).
"""

from sentinel.ds.lru import LRUCache
from sentinel.ds.sketch import CountMinSketch
from sentinel.ds.topk import TopK
from sentinel.ds.welford import Welford
from sentinel.ds.windows import SlidingWindowDistinct, SlidingWindowMax, SlidingWindowSum

__all__ = [
    "CountMinSketch",
    "LRUCache",
    "SlidingWindowDistinct",
    "SlidingWindowMax",
    "SlidingWindowSum",
    "TopK",
    "Welford",
]
