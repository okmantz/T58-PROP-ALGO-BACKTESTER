"""Tests for the cost-stress penalty (app.optimize.refinement.apply_cost_stress_penalty
and its wiring into _evaluate / RefinementConfig / the walk-forward-aware GA)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import (
    RefinementConfig, _evaluate, _stressed_risk_config, apply_cost_stress_penalty,
)
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _synthetic_df(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.03, 0.4, n))
    close = 100 + drift
    high = close + rng.random(n) * 0.3
    low = close - rng.random(n) * 0.3
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


_TREND_FOLLOWER = {
    "name": "EMA cross (trend follower for cost-stress testing)",
    "indicators": [
        {"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
        {"type": "ema", "period": 30, "column": "close", "as": "ema_slow"},
    ],
    "long_entry": "ema_fast > ema_slow",
    "long_exit": "ema_fast < ema_slow",
    "short_entry": "ema_fast < ema_slow",
    "short_exit": "ema_fast > ema_slow",
    "stop_loss_pips": 300,
    "take_profit_pips": 600,
}


# ---------------------------------------------------------------------------
# apply_cost_stress_penalty -- pure function
# ---------------------------------------------------------------------------

def test_zero_weight_returns_nominal_unchanged():
    assert apply_cost_stress_penalty(10.0, -5.0, 0.0) == 10.0


def test_nonpositive_nominal_returned_unchanged():
    assert apply_cost_stress_penalty(0.0, -100.0, 1.0) == 0.0
    assert apply_cost_stress_penalty(-3.0, 50.0, 1.0) == -3.0


def test_nonfinite_nominal_returned_unchanged():
    assert apply_cost_stress_penalty(float("-inf"), 5.0, 1.0) == float("-inf")


def test_no_degradation_when_stressed_equals_nominal():
    assert apply_cost_stress_penalty(10.0, 10.0, 1.0) == pytest.approx(10.0)


def test_stressed_outperformance_is_not_rewarded():
    # Stressed run scoring slightly ABOVE nominal (pure noise) must not
    # increase fitness above nominal -- degradation is clamped at >= 0.
    assert apply_cost_stress_penalty(10.0, 12.0, 1.0) == pytest.approx(10.0)


def test_full_erosion_at_weight_1_drives_fitness_to_zero():
    assert apply_cost_stress_penalty(10.0, float("-inf"), 1.0) == pytest.approx(0.0)


def test_partial_degradation_scales_with_weight():
    # nominal=10, stressed=5 -> degradation fraction = 0.5
    assert apply_cost_stress_penalty(10.0, 5.0, 1.0) == pytest.approx(5.0)
    assert apply_cost_stress_penalty(10.0, 5.0, 0.5) == pytest.approx(7.5)
    assert apply_cost_stress_penalty(10.0, 5.0, 0.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _stressed_risk_config
# ---------------------------------------------------------------------------

def test_stressed_risk_config_scales_only_cost_fields():
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.5, spread_pips=1.0, slippage_pips=0.5,
                       commission_per_trade=2.0, max_trades_per_day=7)
    stressed = _stressed_risk_config(risk, 3.0)
    assert stressed.spread_pips == pytest.approx(3.0)
    assert stressed.slippage_pips == pytest.approx(1.5)
    assert stressed.commission_per_trade == pytest.approx(6.0)
    # everything else must be untouched
    assert stressed.initial_balance == risk.initial_balance
    assert stressed.risk_value == risk.risk_value
    assert stressed.max_trades_per_day == risk.max_trades_per_day


# ---------------------------------------------------------------------------
# RefinementConfig defaults / validation
# ---------------------------------------------------------------------------

def test_refinement_config_cost_stress_defaults_on():
    cfg = RefinementConfig()
    assert cfg.cost_stress_enabled is True
    assert cfg.cost_stress_multiplier >= 1.0
    assert 0.0 <= cfg.cost_stress_penalty_weight <= 1.0


def test_refinement_config_clamps_out_of_range_values():
    cfg = RefinementConfig(cost_stress_multiplier=0.1, cost_stress_penalty_weight=5.0)
    assert cfg.cost_stress_multiplier >= 1.0
    assert cfg.cost_stress_penalty_weight == 1.0


# ---------------------------------------------------------------------------
# _evaluate wiring: stressing a real backtest can only pull fitness down
# ---------------------------------------------------------------------------

def test_evaluate_with_cost_stress_never_exceeds_nominal_fitness():
    df = _synthetic_df()
    strategy = ManualStrategy(_TREND_FOLLOWER)
    # Nonzero baseline cost is required -- multiplying a 0-cost RiskConfig
    # by anything is still 0-cost, so stress would (correctly) do nothing.
    risk = RiskConfig(spread_pips=1.0, slippage_pips=0.5, commission_per_trade=1.0)
    prop_rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=100, random_seed=1)

    fitness_no_stress, *_ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=None, cost_stress_penalty_weight=0.0,
    )
    fitness_with_stress, *_ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=5.0, cost_stress_penalty_weight=1.0,
    )
    assert math.isfinite(fitness_no_stress)
    assert fitness_with_stress <= fitness_no_stress + 1e-6


def test_evaluate_reported_statistics_are_always_nominal_not_stressed():
    """The scalar fitness is cost-stress-adjusted, but the statistics/
    prop_summary/mc_summary returned alongside it must always describe the
    NOMINAL run -- reports should never silently show stressed-cost numbers."""
    df = _synthetic_df()
    strategy = ManualStrategy(_TREND_FOLLOWER)
    risk = RiskConfig(spread_pips=1.0, slippage_pips=0.5, commission_per_trade=1.0)
    prop_rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=100, random_seed=1)

    _, stats_nominal, _, _, _, _, _ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=None, cost_stress_penalty_weight=0.0,
    )
    _, stats_stressed_call, _, _, _, _, _ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=5.0, cost_stress_penalty_weight=1.0,
    )
    assert stats_nominal["net_profit"] == pytest.approx(stats_stressed_call["net_profit"])


def test_zero_weight_makes_evaluate_stress_a_no_op():
    df = _synthetic_df()
    strategy = ManualStrategy(_TREND_FOLLOWER)
    risk = RiskConfig(spread_pips=1.0, slippage_pips=0.5, commission_per_trade=1.0)
    prop_rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=100, random_seed=1)

    fitness_a, *_ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=None, cost_stress_penalty_weight=0.0,
    )
    fitness_b, *_ = _evaluate(
        df, strategy, risk, prop_rules, mc_cfg, "net_profit",
        cost_stress_multiplier=5.0, cost_stress_penalty_weight=0.0,
    )
    assert fitness_a == pytest.approx(fitness_b)
