"""Tests for app.validation.parameter_robustness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation.parameter_robustness import compute_parameter_robustness


def _trending_df(n=1200, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
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


def _fixed_config():
    """A strategy with no tunable numeric knobs at all (fixed conditions,
    no indicators/risk gene the discovery machinery picks up) -- exercises
    the 'nothing to sweep' error path."""
    return {
        "name": "always flat",
        "long_entry": "1 > 2",
        "long_exit": "1 > 2",
    }


def _prop_rules():
    return PropRules(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=1, consistency_rule_pct=None,
    )


def test_compute_parameter_robustness_runs_and_scores_0_to_100():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = compute_parameter_robustness(
        df, strategy, RiskConfig(), _prop_rules(), MonteCarloConfig(n_simulations=80, random_seed=1),
        max_params=3, n_steps_1d=5, n_steps_2d=4, n_heatmap_pairs=1,
    )
    assert 0.0 <= result.parameter_robustness_score <= 100.0
    assert result.n_parameters_checked >= 1
    assert len(result.pair_heatmaps) == 1
    heatmap = result.pair_heatmaps[0]
    assert len(heatmap.rows) == len(heatmap.heatmap.a_values) * len(heatmap.heatmap.b_values)
    for row in heatmap.rows:
        assert 0.0 <= row.pass_pct <= 100.0 or not (row.pass_pct == row.pass_pct)  # allow NaN-safe


def test_heatmap_rows_match_grid_shape_and_labels():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = compute_parameter_robustness(
        df, strategy, RiskConfig(), _prop_rules(), MonteCarloConfig(n_simulations=50, random_seed=2),
        max_params=2, n_steps_1d=4, n_steps_2d=3, n_heatmap_pairs=1,
    )
    heatmap = result.pair_heatmaps[0]
    a_label = heatmap.heatmap.gene_a_label
    b_label = heatmap.heatmap.gene_b_label
    assert all(r.param_a_label == a_label and r.param_b_label == b_label for r in heatmap.rows)


def test_no_tunable_parameters_raises():
    df = _trending_df(n=200)
    strategy = ManualStrategy(_fixed_config())
    with pytest.raises(RefinementError):
        compute_parameter_robustness(
            df, strategy, RiskConfig(), _prop_rules(), MonteCarloConfig(n_simulations=20, random_seed=1),
        )
