"""
Persistent indicator/feature cache -- keyed by (data, indicator, params).

Search Lab, Evolution Lab, and Iterative Refinement all re-backtest the
SAME underlying market data hundreds to tens of thousands of times per run
(one Stage 1 candidate, one GA/surrogate generation member, one walk-forward
fold, one robustness neighbor...). A huge fraction of those candidates
share the exact same indicator + period + column combination -- e.g. every
`mtf_pullback` candidate in a parameter grid still computes EMA(20) the
same way, and a GA/surrogate search re-proposes similar gene values
generation after generation. Recomputing EMA/RSI/ATR/etc. from scratch for
every single one of those is pure waste.

This module is a small process-local cache (module-level dict) sitting in
front of app.strategy.indicators.build_indicator_series. It is
DELIBERATELY process-local, not cross-process/shared/disk-backed:

  - Search Lab / Evolution Lab workers already load the market data ONCE
    per worker process (see app.search.batch_runner._init_worker /
    app.evolution.engine._evo_init_worker) and keep it fixed for the
    worker's entire lifetime -- so a process-local cache already covers
    every candidate that worker will ever backtest in this run.
  - Avoids any inter-process serialization cost (pickling indicator
    Series across a process pool would likely cost MORE than just
    recomputing them).
  - Avoids any staleness risk from a shared/disk cache outliving the
    dataset it was computed from.

Cache key includes a cheap fingerprint of the DataFrame (id() + length +
first/last timestamp) in addition to the indicator/period/column/lookback,
so if a worker process is ever reused across two different datasets (a
different Python object happening to receive the same `id()` after the
first one is garbage collected -- rare but possible), a shape/timestamp
mismatch still forces a fresh computation instead of silently returning
another dataset's indicator values.
"""
from __future__ import annotations

import threading
from typing import Callable

import pandas as pd

# Simple cap so a very long-running Evolution Lab session (thousands of
# generations) can't let this grow unbounded. On cap, the whole cache is
# cleared rather than doing LRU bookkeeping -- cheap, and a cleared cache
# just means the next few candidates recompute once, not a correctness issue.
_MAX_ENTRIES = 20_000

_CACHE: dict[tuple, pd.Series] = {}
_LOCK = threading.Lock()
_HITS = 0
_MISSES = 0


def _frame_fingerprint(frame: pd.DataFrame) -> tuple:
    try:
        n = len(frame)
        if n and "timestamp" in frame.columns:
            first_ts = frame["timestamp"].iloc[0]
            last_ts = frame["timestamp"].iloc[-1]
        else:
            first_ts = last_ts = None
        return (id(frame), n, str(first_ts), str(last_ts))
    except Exception:  # noqa: BLE001 -- fingerprinting must never crash a backtest
        return (id(frame), len(frame) if frame is not None else 0, None, None)


def get_or_compute(
    frame: pd.DataFrame,
    kind: str,
    period: int,
    column: str,
    lookback: int | None,
    compute_fn: Callable[[], pd.Series],
) -> pd.Series:
    """Returns compute_fn()'s result, from cache if this exact
    (frame, kind, period, column, lookback) combination was already
    computed once by this process. Always returns a fresh `.copy()` so a
    caller mutating the returned Series in place can never corrupt what
    other candidates read from the cache."""
    key = (_frame_fingerprint(frame), kind, int(period), column, lookback)

    global _HITS, _MISSES
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        with _LOCK:
            _HITS += 1
        return cached.copy()

    result = compute_fn()

    with _LOCK:
        _MISSES += 1
        if len(_CACHE) >= _MAX_ENTRIES:
            _CACHE.clear()
        _CACHE[key] = result
    return result.copy()


def clear() -> None:
    """Drops every cached series. Call when switching to a genuinely new
    dataset within the same long-lived process (e.g. multi-instrument
    search re-using a worker pool across instruments)."""
    global _HITS, _MISSES
    with _LOCK:
        _CACHE.clear()
        _HITS = 0
        _MISSES = 0


def stats() -> dict:
    """Hit/miss counters -- surfaced in Search Lab / Evolution Lab progress
    logs so Owen can see the cache is actually doing something, not just
    trust that it is."""
    with _LOCK:
        total = _HITS + _MISSES
        hit_rate = (_HITS / total) if total else 0.0
        return {"entries": len(_CACHE), "hits": _HITS, "misses": _MISSES, "hit_rate": hit_rate}
