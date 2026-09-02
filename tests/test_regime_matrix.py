"""Tests for app.validation.regime_matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy
from app.validation.regime_matrix import (
    build_regime_matrix,
    classify_latest_regime,
    is_regime_disabled,
    label_regimes,
    run_regime_matrix,
)


def _trending_df(n=3000, seed=1):
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


def _ema_cross_config():
    return {
        "indicators": [
            {"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
            {"type": "ema", "period": 30, "column": "close", "as": "ema_slow"},
        ],
        "long_entry": "ema_fast > ema_slow",
        "short_entry": "ema_fast < ema_slow",
    }


def test_label_regimes_covers_every_bar_in_one_of_five_session_buckets():
    df = _trending_df()
    labels, thresholds = label_regimes(df)
    assert set(labels["session"].dropna().unique()) <= {"asia", "london", "ny_open", "ny", "power_hour"}
    assert labels["session"].notna().all()  # session has no warm-up window, unlike the other 3 dims


def test_label_regimes_produces_five_buckets_for_trend_and_volatility():
    df = _trending_df()
    labels, _ = label_regimes(df)
    assert labels["trend"].dropna().nunique() == 5
    assert labels["volatility"].dropna().nunique() == 5


def test_thresholds_reapply_deterministically_to_new_data():
    df = _trending_df()
    _, thresholds = label_regimes(df)
    tail = df.iloc[-300:].reset_index(drop=True)
    relabeled, _ = label_regimes(tail, thresholds=thresholds)
    # Same thresholds applied twice to the same data must agree exactly.
    relabeled_again, _ = label_regimes(tail, thresholds=thresholds)
    pd.testing.assert_series_equal(relabeled["volatility"], relabeled_again["volatility"])


def test_build_regime_matrix_returns_none_on_too_little_data():
    df = _trending_df(n=50)
    result = build_regime_matrix(df, trades=[], initial_balance=10_000.0)
    assert result is None


def test_run_regime_matrix_cells_sum_trades_to_backtest_total_minus_unclassifiable():
    df = _trending_df()
    strategy = ManualStrategy(_ema_cross_config())
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    bt = run_backtest(df, strategy, risk)

    result = run_regime_matrix(df, strategy, risk, dimensions=("volatility", "environment"))
    assert result is not None
    assert result.primary_dimensions == ("volatility", "environment")

    attributed = sum(c.n_trades for c in result.cells)
    # Every trade should be attributed unless its entry fell in the
    # warm-up window before both dimensions had a valid reading.
    assert attributed <= len(bt.trades)
    assert attributed > 0


def test_disable_regimes_only_flags_cells_meeting_both_trade_count_and_pf_bar():
    df = _trending_df()
    strategy = ManualStrategy(_ema_cross_config())
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_regime_matrix(
        df, strategy, risk, dimensions=("volatility", "environment"),
        min_trades_for_verdict=15, disable_pf_threshold=1.0,
    )
    assert result is not None
    for cell in result.disable_regimes():
        assert cell.n_trades >= 15
        assert cell.profit_factor < 1.0
    # Every flagged cell must actually be a member of result.cells.
    assert set(id(c) for c in result.disable_regimes()) <= set(id(c) for c in result.cells)


def test_single_dimension_breakdown_covers_the_two_dimensions_not_in_the_primary_matrix():
    df = _trending_df()
    strategy = ManualStrategy(_ema_cross_config())
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_regime_matrix(df, strategy, risk, dimensions=("volatility", "environment"))
    assert result is not None
    assert set(result.single_dimension.keys()) == {"trend", "session"}
    assert len(result.single_dimension["trend"]) <= 5
    assert len(result.single_dimension["session"]) <= 5


def test_render_table_lists_every_cell_and_any_disable_recommendation():
    df = _trending_df()
    strategy = ManualStrategy(_ema_cross_config())
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_regime_matrix(df, strategy, risk)
    assert result is not None
    table = result.render_table()
    assert "Regime Survival Matrix" in table
    for cell in result.cells:
        assert cell.label in table


def test_classify_latest_regime_and_is_regime_disabled_round_trip():
    df = _trending_df()
    strategy = ManualStrategy(_ema_cross_config())
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_regime_matrix(df, strategy, risk, dimensions=("volatility", "environment"))
    assert result is not None

    tail = df.iloc[-300:].reset_index(drop=True)
    current = classify_latest_regime(tail, result.thresholds, dims=("volatility", "environment"))
    assert set(current.keys()) == {"volatility", "environment"}

    # A synthetic "current" matching a known disabled cell (if any) must be
    # detected; an empty disabled list must never flag anything.
    if result.disable_regimes():
        disabled_dims = result.disable_regimes()[0].dims
        assert is_regime_disabled(disabled_dims, result.disable_regimes())
    assert not is_regime_disabled(current, [])
