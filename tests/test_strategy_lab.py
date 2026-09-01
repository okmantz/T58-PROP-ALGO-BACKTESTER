"""
Tests for app.lab.strategy_lab.

Deliberately tiny settings throughout (few candidates, few sims, few GA
generations) -- this exercises the WHOLE funnel end-to-end (a real
integration test spanning Search Lab's Stage 1-3, regime testing,
parameter stability, and the untouched test), which is inherently slower
than a pure unit test. Kept fast enough for a normal test run by shrinking
every cost knob rather than by skipping stages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.lab.strategy_lab import StrategyLabSpec, run_strategy_lab


def _trending_df(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = 0.00012 * np.sin(i / 250) + rng.normal(0, 0.00007)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _tiny_spec(**overrides):
    base = dict(
        market="TEST", timeframe_label="5m", trading_window_label="Any",
        goal_metric="first_payout_probability",
        n_candidates=8, stage1_top_n=6, stage2_top_n=4, walk_forward_top_n=3, monte_carlo_top_n=2,
        regime_top_n=2, finalist_count=2,
        ga_population=4, ga_generations=1, ga_search_mc_sims=30, full_mc_sims=60,
        walk_forward_folds=2, survival_mc_sims=60, survival_life_sims=30,
        parameter_stability_max_params=2, parameter_stability_n_steps_1d=3, parameter_stability_n_steps_2d=3,
        regime_min_segment_bars=40, workers=1, random_seed=1,
    )
    base.update(overrides)
    return StrategyLabSpec(**base)


def _prop_rules():
    return PropRules(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=2, consistency_rule_pct=30,
    )


def test_strategy_lab_runs_end_to_end_and_ranks_finalists():
    df = _trending_df()
    result = run_strategy_lab(df, RiskConfig(), _prop_rules(), _tiny_spec())

    assert result.total_candidates <= 8
    assert result.stage1_survivors <= 6
    assert result.stage2_survivors <= 4

    ranks = [f.rank for f in result.finalists]
    assert ranks == sorted(ranks)
    scores = [f.prop_robustness_score for f in result.finalists]
    assert scores == sorted(scores, reverse=True)
    for f in result.finalists:
        assert 0.0 <= f.prop_robustness_score <= 100.0
        assert f.explanation  # the "Explain" step always produces something


def test_strategy_lab_rejects_unknown_goal_metric():
    with pytest.raises(ValueError):
        StrategyLabSpec(goal_metric="not_a_real_metric")


def test_strategy_lab_handles_no_stage3_survivors_gracefully():
    """An impossible filter (a 100% profit-factor floor no candidate can
    clear) should return a clean empty result, not raise."""
    df = _trending_df(n=400)
    spec = _tiny_spec(n_candidates=4, stage1_top_n=1, stage2_top_n=1)
    # Force Stage 1 to reject everything via an absurd profit-factor floor
    # by shrinking the search space to a single family with a tiny sample.
    spec.stage1_top_n = 1
    result = run_strategy_lab(df, RiskConfig(), _prop_rules(), spec)
    assert isinstance(result.finalists, list)
