"""Tests for app.strategy.regime_router."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.regime_router import RegimeRouterStrategy
from app.validation.regime_matrix import label_regimes


def _trending_df(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.5, n))
    price = 1900 + drift + noise
    high = price + np.abs(rng.normal(0.3, 0.15, n))
    low = price - np.abs(rng.normal(0.3, 0.15, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": high, "low": low, "close": price, "volume": 100.0,
    })


def _always_long_config():
    """Constant long signal -- ATR-based risk mgmt so it produces a
    per-bar stop/target distance series."""
    return {
        "long_entry": "close > 0",
        "risk_management": {
            "stop_type": "atr", "stop_value": 1.0, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 2.0, "target_atr_period": 14,
        },
    }


def _always_short_config():
    return {
        "short_entry": "close > 0",
        "risk_management": {
            "stop_type": "atr", "stop_value": 1.5, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 2.5, "target_atr_period": 14,
        },
    }


def _always_long_pips_config():
    """Constant long signal, scalar-pips risk mgmt (no distance series) --
    exercises the router's pips-to-distance fallback conversion."""
    return {
        "long_entry": "close > 0",
        "risk_management": {"stop_type": "fixed", "stop_value": 20, "target_type": "fixed", "target_value": 40},
    }


def test_rejects_unknown_dimension():
    with pytest.raises(StrategyError):
        RegimeRouterStrategy("not_a_dimension", {"x": ManualStrategy(_always_long_config())})


def test_rejects_empty_mapping():
    with pytest.raises(StrategyError):
        RegimeRouterStrategy("environment", {})


def test_signal_only_active_within_assigned_regime():
    df = _trending_df()
    router = RegimeRouterStrategy(
        regime_dimension="environment",
        strategies_by_regime={
            "trending": ManualStrategy(_always_long_config()),
            "ranging": ManualStrategy(_always_short_config()),
        },
    )
    result = router.generate(df)
    labels, _ = label_regimes(df)
    env = labels["environment"]

    trending_signals = result.signals[env == "trending"]
    ranging_signals = result.signals[env == "ranging"]
    other_signals = result.signals[~env.isin(["trending", "ranging"])]

    assert (trending_signals == 1.0).all()
    assert (ranging_signals == -1.0).all()
    assert (other_signals == 0.0).all()


def test_unassigned_regime_bars_stay_flat_not_defaulted():
    df = _trending_df()
    router = RegimeRouterStrategy(
        regime_dimension="environment",
        strategies_by_regime={"trending": ManualStrategy(_always_long_config())},
    )
    result = router.generate(df)
    labels, _ = label_regimes(df)
    non_trending = result.signals[labels["environment"] != "trending"]
    assert (non_trending == 0.0).all()


def test_combines_per_bar_atr_distance_series_from_each_sub_strategy():
    df = _trending_df()
    router = RegimeRouterStrategy(
        regime_dimension="environment",
        strategies_by_regime={
            "trending": ManualStrategy(_always_long_config()),
            "ranging": ManualStrategy(_always_short_config()),
        },
    )
    result = router.generate(df)
    assert result.stop_loss_distance is not None
    assert result.take_profit_distance is not None
    labels, _ = label_regimes(df)
    active_mask = labels["environment"].isin(["trending", "ranging"])
    # Every bar actually routed to a sub-strategy must have a real (non-NaN)
    # stop/target distance -- the two ATR-based sub-strategies always set one.
    assert result.stop_loss_distance[active_mask].notna().all()
    assert result.take_profit_distance[active_mask].notna().all()


def test_scalar_pips_substrategy_converted_to_distance_series():
    df = _trending_df()
    pip_size = 0.01
    router = RegimeRouterStrategy(
        regime_dimension="environment",
        strategies_by_regime={"trending": ManualStrategy(_always_long_pips_config())},
        pip_size=pip_size,
    )
    result = router.generate(df)
    labels, _ = label_regimes(df)
    mask = (labels["environment"] == "trending").fillna(False)
    if mask.any():
        assert result.stop_loss_distance[mask].notna().all()
        assert np.isclose(result.stop_loss_distance[mask].iloc[0], 20 * pip_size)


def test_regime_router_runs_end_to_end_through_the_backtest_engine():
    df = _trending_df()
    router = RegimeRouterStrategy(
        regime_dimension="environment",
        strategies_by_regime={
            "trending": ManualStrategy(_always_long_config()),
            "ranging": ManualStrategy(_always_short_config()),
        },
    )
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    bt = run_backtest(df, router, risk)
    assert bt.trades  # real, backtestable trades came out the other end


def test_thresholds_can_be_reused_for_forward_consistency():
    df = _trending_df()
    _, thresholds = label_regimes(df)
    tail = df.iloc[-500:].reset_index(drop=True)
    router = RegimeRouterStrategy(
        regime_dimension="volatility",
        strategies_by_regime={"extreme": ManualStrategy(_always_long_config())},
        thresholds=thresholds,
    )
    result = router.generate(tail)
    assert len(result.signals) == len(tail)
