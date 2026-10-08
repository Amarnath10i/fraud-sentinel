"""Least-recently-used cache: hash map + doubly linked list.

The online feature engine keeps one state object per card, merchant, etc. In a
real deployment the number of entities is unbounded, so state lives in an LRU
with a fixed capacity: the hash map gives O(1) lookup, the linked list gives
O(1) "move to most recent" and "evict least recent".
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _Node:
    __slots__ = ("key", "next", "prev", "value")

    def __init__(self, key=None, value=None) -> None:
        self.key = key
        self.value = value
        self.prev: _Node = self
        self.next: _Node = self


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int, on_evict: Callable[[K, V], None] | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.on_evict = on_evict
        self.evictions = 0
        self._map: dict[K, _Node] = {}
        self._head = _Node()  # sentinel: head.next is most recent, head.prev least recent

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: object) -> bool:
        return key in self._map

    def __iter__(self) -> Iterator[K]:
        """Keys from most to least recently used."""
        node = self._head.next
        while node is not self._head:
            yield node.key
            node = node.next

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        head = self._head
        node.prev = head
        node.next = head.next
        head.next.prev = node
        head.next = node

    def get(self, key: K, default: V | None = None) -> V | None:
        node = self._map.get(key)
        if node is None:
            return default
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: K, value: V) -> None:
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._map) >= self.capacity:
            lru = self._head.prev
            self._unlink(lru)
            del self._map[lru.key]
            self.evictions += 1
            if self.on_evict is not None:
                self.on_evict(lru.key, lru.value)
        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        value = self.get(key)
        if value is None:
            value = factory()
            self.put(key, value)
        return value
