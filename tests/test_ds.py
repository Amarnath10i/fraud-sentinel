"""Property tests: every data structure must agree with a brute-force reference."""

import math
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.ds import (
    CountMinSketch,
    LRUCache,
    SlidingWindowDistinct,
    SlidingWindowMax,
    SlidingWindowSum,
    TopK,
    Welford,
)

# A stream of (time gap, value, key) with frequent equal timestamps.
streams = st.lists(
    st.tuples(st.integers(0, 50), st.integers(1, 10_000), st.integers(0, 6)),
    min_size=1,
    max_size=200,
)
spans = st.one_of(st.none(), st.integers(1, 120))


def _brute(history, now, span):
    lo = -math.inf if span is None else now - span
    return [(v, k) for ts, v, k in history if lo <= ts < now]


@settings(max_examples=300, deadline=None)
@given(streams, spans)
def test_sliding_windows_match_brute_force(stream, span):
    # Drive the windows the way the feature engine does: events that share a
    # timestamp are all read before any of them is added.
    s, m, d = SlidingWindowSum(span), SlidingWindowMax(span), SlidingWindowDistinct(span)
    history, now, groups = [], 0, {}
    for gap, value, key in stream:
        now += gap
        groups.setdefault(now, []).append((value, key))
    for now, group in groups.items():
        for w in (s, m, d):
            w.advance(now)
        expect = _brute(history, now, span)
        values = [v for v, _ in expect]
        assert s.count == len(values)
        assert s.total == sum(values)
        assert s.total_sq == sum(v * v for v in values)
        if values:
            mu = sum(values) / len(values)
            assert s.var_pop() == pytest.approx(sum((v - mu) ** 2 for v in values) / len(values))
        assert m.max == (max(values) if values else None)
        assert d.distinct == len({k for _, k in expect})
        for value, key in group:
            s.add(now, value)
            m.add(now, value)
            d.add(now, key)
            history.append((now, value, key))


def test_events_at_the_same_time_are_not_visible_until_time_moves():
    s = SlidingWindowSum(span=10)
    s.advance(100)
    s.add(100, 5)
    s.advance(100)  # same timestamp: the reader is expected to read *before* adding
    assert s.count == 1
    s.advance(110)
    assert s.count == 1  # ts=100 is still inside [100, 110)
    s.advance(111)
    assert s.count == 0


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=300), st.data())
def test_welford_matches_numpy_and_merge_is_exact(xs, data):
    w = Welford()
    for x in xs:
        w.add(x)
    assert w.mean == pytest.approx(np.mean(xs), rel=1e-9, abs=1e-6)
    assert w.var_pop == pytest.approx(np.var(xs), rel=1e-7, abs=1e-4)

    cut = data.draw(st.integers(0, len(xs)))
    a, b = Welford(), Welford()
    for x in xs[:cut]:
        a.add(x)
    for x in xs[cut:]:
        b.add(x)
    merged = a.merge(b)
    assert merged.n == w.n
    assert merged.mean == pytest.approx(w.mean, rel=1e-9, abs=1e-6)
    assert merged.var_pop == pytest.approx(w.var_pop, rel=1e-7, abs=1e-4)


def test_welford_is_stable_where_the_naive_formula_fails():
    xs = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]  # true variance 22.5
    w = Welford()
    for x in xs:
        w.add(x)
    naive = np.mean(np.square(xs)) - np.mean(xs) ** 2
    assert w.var_pop == pytest.approx(22.5, rel=1e-9)
    assert abs(naive - 22.5) > 1.0


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 8), st.lists(st.tuples(st.booleans(), st.integers(0, 15)), max_size=200))
def test_lru_matches_reference_model(capacity, ops):
    lru: LRUCache[int, int] = LRUCache(capacity)
    ref: list[int] = []  # most recent first
    values: dict[int, int] = {}
    for i, (is_put, key) in enumerate(ops):
        if is_put:
            lru.put(key, i)
            values[key] = i
            if key in ref:
                ref.remove(key)
            elif len(ref) == capacity:
                values.pop(ref.pop())
            ref.insert(0, key)
        else:
            got = lru.get(key)
            if key in ref:
                assert got == values[key]
                ref.remove(key)
                ref.insert(0, key)
            else:
                assert got is None
        assert list(lru) == ref


def test_lru_calls_eviction_hook():
    evicted = []
    lru = LRUCache(2, on_evict=lambda k, v: evicted.append(k))
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")
    lru.put("c", 3)
    assert evicted == ["b"] and lru.evictions == 1


def test_count_min_never_underestimates_and_respects_error_bound():
    rng = random.Random(0)
    keys = [rng.randrange(2_000) for _ in range(20_000)]
    eps, delta = 0.002, 0.01
    cms = CountMinSketch.from_error(eps, delta)
    truth: dict[int, int] = {}
    for k in keys:
        cms.add(k)
        truth[k] = truth.get(k, 0) + 1
    errors = np.array([cms.estimate(k) - c for k, c in truth.items()])
    assert (errors >= 0).all()
    assert np.mean(errors <= eps * len(keys)) >= 1 - delta


def test_conservative_update_is_tighter():
    rng = random.Random(1)
    keys = [rng.randrange(500) for _ in range(5_000)]
    plain, cons = CountMinSketch(64, 3, conservative=False), CountMinSketch(64, 3)
    for k in keys:
        plain.add(k)
        cons.add(k)
    err = lambda s: sum(s.estimate(k) - keys.count(k) for k in set(keys))  # noqa: E731
    assert err(cons) < err(plain)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10), st.lists(st.integers(0, 50), max_size=100))
def test_topk_keeps_the_k_largest(k, priorities):
    top: TopK[int] = TopK(k)
    for i, p in enumerate(priorities):
        top.push(p, i)
    kept = [p for p, _ in top.items()]
    assert kept == sorted(priorities, reverse=True)[:k]
