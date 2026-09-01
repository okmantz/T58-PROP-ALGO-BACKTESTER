"""Tests for app.prop.survival_engine."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.prop.simulator import PropRules
from app.prop.survival_engine import (
    PropSurvivalConfig, ResetEconomics, run_prop_survival_analysis, simulate_reset_chain,
)
from app.prop.simulator import precompute_day_structure


def _mock_trades(n=250, seed=1, mean=25.0, std=180.0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    trades = []
    for i in range(n):
        t = base + pd.Timedelta(days=i // 3)
        pnl = float(rng.normal(mean, std))
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _rules(**overrides):
    base = dict(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=3, consistency_rule_pct=30, payout_frequency_days=14,
    )
    base.update(overrides)
    return PropRules(**base)


def test_survival_analysis_runs_and_produces_bounded_probabilities():
    trades = _mock_trades(300)
    cfg = PropSurvivalConfig(n_simulations=200, life_simulations=100, random_seed=1)
    result = run_prop_survival_analysis(trades, _rules(), cfg)

    for pct in (
        result.evaluation.probability_pass_evaluation,
        result.evaluation.probability_hit_daily_loss,
        result.evaluation.probability_hit_max_drawdown,
        result.funded.probability_first_payout,
        result.funded.probability_second_payout,
        result.funded.probability_third_payout,
        result.reset_economics.probability_net_positive_after_resets,
        result.prop_survival_score,
    ):
        assert 0.0 <= pct <= 100.0


def test_payout_probabilities_are_monotonically_non_increasing():
    """You can't have a HIGHER probability of a 3rd payout than a 2nd, or
    a 2nd than a 1st -- each requires the ones before it."""
    trades = _mock_trades(400, mean=40.0, std=150.0)
    cfg = PropSurvivalConfig(n_simulations=500, life_simulations=50, random_seed=2)
    result = run_prop_survival_analysis(trades, _rules(payout_frequency_days=5), cfg)
    f = result.funded
    assert f.probability_first_payout >= f.probability_second_payout >= f.probability_third_payout


def test_reset_economics_more_attempts_increases_cost_and_eventual_survival():
    """More attempts monotonically costs more in expected fees (each extra
    attempt is a real, non-negotiable fee) -- and, since each attempt is
    an independent chance to eventually survive to the end of the data,
    the probability of burning through every attempt without ever
    surviving can only go down (never up) as more attempts are allowed.
    Note this does NOT mean expected net profit always improves with more
    attempts -- for a weak/negative-edge strategy, extra resets can easily
    bleed more in fees than they recover, which is itself the whole point
    of modeling reset economics explicitly rather than assuming "more
    tries = better."."""
    trades = _mock_trades(300, mean=15.0, std=200.0)
    econ_low = ResetEconomics(evaluation_fee=100, reset_fee=50, profit_split_pct=80, max_attempts=1)
    econ_high = ResetEconomics(evaluation_fee=100, reset_fee=50, profit_split_pct=80, max_attempts=4)
    cfg_low = PropSurvivalConfig(n_simulations=100, life_simulations=400, random_seed=5, reset_economics=econ_low)
    cfg_high = PropSurvivalConfig(n_simulations=100, life_simulations=400, random_seed=5, reset_economics=econ_high)
    result_low = run_prop_survival_analysis(trades, _rules(), cfg_low)
    result_high = run_prop_survival_analysis(trades, _rules(), cfg_high)

    assert result_high.reset_economics.expected_fees_paid >= result_low.reset_economics.expected_fees_paid
    assert result_high.reset_economics.probability_exhausts_resets_without_profit <= \
        result_low.reset_economics.probability_exhausts_resets_without_profit + 1e-6


def test_zero_trades_raises():
    with pytest.raises(ValueError):
        run_prop_survival_analysis([], PropRules(), PropSurvivalConfig(n_simulations=10, life_simulations=10))


def test_simulate_reset_chain_costs_at_least_one_fee():
    trades = _mock_trades(100, mean=-50.0, std=50.0)  # a losing strategy
    base_pnls = np.array([t.pnl for t in trades])
    base_dates = [pd.Timestamp(t.entry_time).normalize() for t in trades]
    day_structure = precompute_day_structure(base_dates)
    rng = np.random.default_rng(3)
    reset = ResetEconomics(evaluation_fee=200, reset_fee=100, profit_split_pct=80, max_attempts=2)

    class _Cfg:
        method = "bootstrap"
        block_size = 5

    life = simulate_reset_chain(base_pnls, base_dates, _rules(), day_structure, reset, _Cfg(), 0.0, rng)
    assert life["fees_paid"] >= reset.evaluation_fee
    assert life["attempts_used"] >= 1
    assert math.isfinite(life["net_profit"])
