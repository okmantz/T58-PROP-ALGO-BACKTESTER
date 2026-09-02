"""Tests for app.optimize.risk_sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.optimize.risk_sweep import DEFAULT_RISK_VALUES, run_risk_sweep
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _trending_df(n=2500, seed=4):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 60, n)
    noise = np.cumsum(rng.normal(0, 0.6, n))
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
        "long_exit": "ema_fast < ema_slow",
        "risk_management": {"stop_type": "fixed", "stop_value": 20, "target_type": "fixed", "target_value": 40},
    }


def _rules():
    return PropRules(account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5, max_drawdown_pct=10)


def _fast_survival_cfg():
    from app.prop.survival_engine import PropSurvivalConfig
    return PropSurvivalConfig(n_simulations=200, life_simulations=50)


def test_risk_sweep_produces_one_point_per_risk_level():
    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_risk_sweep(
        df, lambda: ManualStrategy(_ema_cross_config()), risk, _rules(),
        risk_values=[0.25, 0.5, 1.0], survival_cfg=_fast_survival_cfg(),
    )
    assert len(result.points) == 3
    assert [p.risk_value for p in result.points] == [0.25, 0.5, 1.0]


def test_risk_sweep_picks_the_highest_survival_score_as_best():
    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_risk_sweep(
        df, lambda: ManualStrategy(_ema_cross_config()), risk, _rules(),
        risk_values=[0.25, 0.5, 1.0], survival_cfg=_fast_survival_cfg(),
    )
    assert result.best_point is not None
    assert result.best_point.prop_survival_score == max(p.prop_survival_score for p in result.points)


def test_risk_sweep_defaults_match_masterclass_list():
    assert DEFAULT_RISK_VALUES == [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]


def test_risk_sweep_rejects_empty_risk_values():
    import pytest
    df = _trending_df(n=200)
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    with pytest.raises(ValueError):
        run_risk_sweep(df, lambda: ManualStrategy(_ema_cross_config()), risk, _rules(), risk_values=[])


def test_risk_sweep_skips_levels_that_produce_no_trades_instead_of_crashing():
    df = _trending_df()
    no_signal_config = {"long_entry": "close > 999999999"}
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_risk_sweep(
        df, lambda: ManualStrategy(no_signal_config), risk, _rules(),
        risk_values=[0.25, 0.5], survival_cfg=_fast_survival_cfg(),
    )
    assert result.points == []
    assert result.best_point is None
    assert result.notes


def test_render_table_includes_best_marker_and_headline():
    df = _trending_df()
    risk = RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)
    result = run_risk_sweep(
        df, lambda: ManualStrategy(_ema_cross_config()), risk, _rules(),
        risk_values=[0.25, 1.0], survival_cfg=_fast_survival_cfg(),
    )
    table = result.render_table()
    assert "Risk Sweep" in table
    assert "Best risk level for this strategy/data" in table
