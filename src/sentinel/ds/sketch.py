"""Count-Min Sketch with conservative update.

Approximate frequency counts in fixed memory. For width w = ceil(e / eps) and
depth d = ceil(ln(1 / delta)), every estimate satisfies

    true <= estimate <= true + eps * N   with probability >= 1 - delta,

where N is the total count added. Conservative update only raises the counters
that are at the current minimum, which tightens estimates without breaking the
"never underestimates" guarantee.

Hashing uses blake2b so results are identical across processes (Python's
built-in `hash` of strings is salted per process).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

_MASK64 = (1 << 64) - 1


def _hash_pair(key: object) -> tuple[int, int]:
    digest = hashlib.blake2b(repr(key).encode(), digest_size=16).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little") | 1


class CountMinSketch:
    def __init__(self, width: int, depth: int, conservative: bool = True) -> None:
        self.width = width
        self.depth = depth
        self.conservative = conservative
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total = 0

    @classmethod
    def from_error(cls, eps: float, delta: float, conservative: bool = True) -> CountMinSketch:
        return cls(math.ceil(math.e / eps), math.ceil(math.log(1 / delta)), conservative)

    def _cells(self, key: object) -> np.ndarray:
        # Kirsch-Mitzenmacher: d hash functions from two, h_i = h1 + i * h2.
        h1, h2 = _hash_pair(key)
        return np.array(
            [((h1 + i * h2) & _MASK64) % self.width for i in range(self.depth)], dtype=np.int64
        )

    def add(self, key: object, count: int = 1) -> None:
        cols = self._cells(key)
        rows = np.arange(self.depth)
        self.total += count
        if self.conservative:
            target = self.table[rows, cols].min() + count
            self.table[rows, cols] = np.maximum(self.table[rows, cols], target)
        else:
            self.table[rows, cols] += count

    def estimate(self, key: object) -> int:
        return int(self.table[np.arange(self.depth), self._cells(key)].min())

    @property
    def nbytes(self) -> int:
        return self.table.nbytes


class BloomFilter:
    """Set membership in m bits with k hash functions.

    No false negatives; false-positive rate ~ (1 - e^(-k n / m))^k. Sized for
    n items at rate p: m = -n ln p / (ln 2)^2 bits and k = (m / n) ln 2.

    Count-Min answers "how many?" with an additive error of eps * N, which is
    useless for "zero or not?" on a long tail of rare keys. A Bloom filter
    answers exactly the membership question, in far less memory.
    """

    def __init__(self, m_bits: int, k: int) -> None:
        self.m = m_bits
        self.k = k
        self.bits = bytearray((m_bits + 7) // 8)

    @classmethod
    def for_capacity(cls, n: int, p: float) -> BloomFilter:
        m = math.ceil(-n * math.log(p) / math.log(2) ** 2)
        return cls(m, max(1, round(m / n * math.log(2))))

    def _positions(self, key: object):
        h1, h2 = _hash_pair(key)
        return (((h1 + i * h2) & _MASK64) % self.m for i in range(self.k))

    def add(self, key: object) -> None:
        for pos in self._positions(key):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, key: object) -> bool:
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(key))

    @property
    def nbytes(self) -> int:
        return len(self.bits)
