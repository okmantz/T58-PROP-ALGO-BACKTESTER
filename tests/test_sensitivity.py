import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation.sensitivity import (
    compute_1d_sensitivity,
    compute_2d_heatmap,
    list_tunable_parameters,
)


def _trending_df(n=1500, seed=5, drift=0.00015):
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
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def _flat_config():
    return {"name": "no params", "long_entry": "close > 0", "long_exit": "close < 0",
            "short_entry": "close < -1", "short_exit": "close > 1"}


def test_list_tunable_parameters():
    strategy = ManualStrategy(_sma_config())
    labels = list_tunable_parameters(strategy)
    assert any("period" in l for l in labels)


def test_compute_1d_sensitivity_basic():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)

    results = compute_1d_sensitivity(df, strategy, risk, rules, mc_cfg, metric="profit_factor", n_steps=5)
    assert len(results) > 0
    for r in results:
        assert len(r.values) == len(r.metric_values)
        assert isinstance(r.cliff_detected, bool)


def test_compute_2d_heatmap_basic():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    labels = list_tunable_parameters(strategy)
    period_labels = [l for l in labels if "period" in l]
    assert len(period_labels) >= 2

    result = compute_2d_heatmap(
        df, strategy, risk, rules, mc_cfg,
        gene_label_a=period_labels[0], gene_label_b=period_labels[1],
        metric="net_profit", n_steps=4,
    )
    assert len(result.grid) == len(result.a_values)
    assert all(len(row) == len(result.b_values) for row in result.grid)


def test_sensitivity_raises_without_tunable_params():
    df = _trending_df(n=500)
    strategy = ManualStrategy(_flat_config())
    with pytest.raises(RefinementError):
        compute_1d_sensitivity(df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=10))
