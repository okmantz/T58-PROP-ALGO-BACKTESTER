import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation.walk_forward_opt import build_folds, run_walk_forward_optimization


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


def test_build_folds_rolling_and_anchored():
    df = _trending_df(n=1200)
    rolling = build_folds(df, n_folds=4, window_mode="rolling")
    anchored = build_folds(df, n_folds=4, window_mode="anchored")
    assert len(rolling) >= 3
    assert len(anchored) >= 3
    # anchored train windows must be non-decreasing in size
    sizes = [len(f.train_df) for f in anchored]
    assert sizes == sorted(sizes)
    # folds must be chronological and test comes after train
    for f in rolling + anchored:
        assert pd.Timestamp(f.train_period[1]) <= pd.Timestamp(f.test_period[0]) or f.train_period[1] <= f.test_period[0]


def test_build_folds_insufficient_data_returns_empty():
    df = _trending_df(n=30)
    assert build_folds(df, n_folds=5) == []


def test_run_walk_forward_optimization_chains_oos_trades():
    df = _trending_df(n=2400)
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=100)
    refine_cfg = RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=50)

    result = run_walk_forward_optimization(
        df, strategy, risk, rules, mc_cfg,
        n_folds=3, window_mode="rolling", refine_cfg=refine_cfg, random_seed=1,
    )

    assert len(result.folds) >= 2
    assert result.combined_statistics.total_trades == len(result.combined_trades)
    # combined equity curve should be chronologically ordered
    if len(result.combined_equity_curve):
        ts = pd.to_datetime(result.combined_equity_curve["timestamp"])
        assert list(ts) == sorted(ts)
    summary = result.to_summary_dict()
    assert summary["n_folds_completed"] == len(result.folds)


def test_run_walk_forward_optimization_anchored_mode():
    df = _trending_df(n=2000, seed=7)
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=30)

    result = run_walk_forward_optimization(
        df, strategy, risk, rules, mc_cfg,
        n_folds=3, window_mode="anchored", refine_cfg=refine_cfg, random_seed=2,
    )
    assert result.window_mode == "anchored"
    assert len(result.folds) >= 2


def test_run_walk_forward_optimization_raises_on_too_little_data():
    df = _trending_df(n=40)
    strategy = ManualStrategy(_sma_config())
    with pytest.raises(RefinementError):
        run_walk_forward_optimization(df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=10), n_folds=5)
