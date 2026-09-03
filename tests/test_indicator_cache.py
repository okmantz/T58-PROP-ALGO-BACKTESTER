"""Tests for app.strategy.indicator_cache and its wiring into
app.strategy.indicators.build_indicator_series."""
from __future__ import annotations

import pandas as pd

from app.strategy import indicator_cache
from app.strategy.indicators import build_indicator_series


def _df(n=300):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(range(n), dtype=float) + 100.0
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 100.0,
    })


def test_cache_returns_equal_series_on_repeat_call():
    indicator_cache.clear()
    df = _df()
    first = build_indicator_series(df, "ema", period=20, column="close")
    second = build_indicator_series(df, "ema", period=20, column="close")
    pd.testing.assert_series_equal(first, second)


def test_cache_records_a_hit_on_repeat_call():
    indicator_cache.clear()
    df = _df()
    build_indicator_series(df, "rsi", period=14, column="close")
    before = indicator_cache.stats()
    build_indicator_series(df, "rsi", period=14, column="close")
    after = indicator_cache.stats()
    assert after["hits"] == before["hits"] + 1


def test_different_periods_are_cached_separately():
    indicator_cache.clear()
    df = _df()
    ema_20 = build_indicator_series(df, "ema", period=20, column="close")
    ema_50 = build_indicator_series(df, "ema", period=50, column="close")
    assert not ema_20.equals(ema_50)


def test_mutating_returned_series_does_not_corrupt_the_cache():
    indicator_cache.clear()
    df = _df()
    first = build_indicator_series(df, "sma", period=10, column="close")
    first.iloc[:] = -999.0  # mutate the caller's own copy
    second = build_indicator_series(df, "sma", period=10, column="close")
    assert not (second == -999.0).all()


def test_different_dataframes_do_not_share_cache_entries():
    indicator_cache.clear()
    df_a = _df(n=200)
    df_b = _df(n=400)  # different length -> different fingerprint
    a = build_indicator_series(df_a, "ema", period=20, column="close")
    b = build_indicator_series(df_b, "ema", period=20, column="close")
    assert len(a) != len(b)


def test_clear_resets_stats_and_entries():
    df = _df()
    build_indicator_series(df, "atr", period=14, column="close")
    indicator_cache.clear()
    stats = indicator_cache.stats()
    assert stats == {"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
