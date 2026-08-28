import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _trending_df(n=2400, seed=3, drift=0.00015):
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


def test_run_walkforward_aware_refinement_basic():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=30)

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
    )

    assert result.n_folds >= 2
    assert result.best is not None
    assert result.best.config is not None
    assert len(result.leaderboard) == refine_cfg.population_size
    assert len(result.generation_history) == refine_cfg.generations + 1


def test_run_walkforward_aware_refinement_no_params_raises():
    df = _trending_df(n=500)
    strategy = ManualStrategy(_flat_config())
    with pytest.raises(RefinementError):
        run_walkforward_aware_refinement(df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=10), n_folds=3)


def test_run_walkforward_aware_refinement_insufficient_data_raises():
    df = _trending_df(n=30)
    strategy = ManualStrategy(_sma_config())
    with pytest.raises(RefinementError):
        run_walkforward_aware_refinement(df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=10), n_folds=5)
