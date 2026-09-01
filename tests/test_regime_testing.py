"""Tests for app.validation.regime_testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy
from app.validation.regime_testing import run_regime_test


def _two_regime_df(n=3000, seed=5):
    """Alternating calm/volatile 500-bar blocks -- enough data that each
    volatility tercile should find at least one qualifying contiguous
    segment."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        vol = 0.00003 if (i // 500) % 2 == 0 else 0.00020
        step = rng.normal(0, vol)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, vol / 2))
        l = min(o, c) - abs(rng.normal(0, vol / 2))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 20, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "risk_management": {"stop_type": "fixed", "stop_value": 20, "target_type": "fixed", "target_value": 40},
    }


def test_regime_test_returns_expected_bucket_count():
    df = _two_regime_df()
    result = run_regime_test(
        df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_regimes=3, min_segment_bars=80,
    )
    assert result is not None
    assert result.n_buckets == 3
    assert 0.0 <= result.regime_stability_pct <= 100.0
    assert result.n_profitable_buckets == sum(1 for b in result.buckets if b.is_profitable)


def test_regime_test_returns_none_on_too_little_data():
    df = _two_regime_df(n=60)
    result = run_regime_test(df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_regimes=3)
    assert result is None


def test_bucket_trade_counts_are_consistent_with_stability_flag():
    df = _two_regime_df()
    result = run_regime_test(
        df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_regimes=2, min_segment_bars=80,
    )
    assert result is not None
    assert result.is_regime_stable == (result.n_profitable_buckets >= max(1, result.n_buckets - 1))
