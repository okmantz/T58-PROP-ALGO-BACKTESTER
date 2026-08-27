"""Tests for app.search.robustness."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.prop.simulator import PropRules
from app.search.robustness import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    parameter_neighborhood_robustness,
    run_walk_forward,
)
from app.strategy.manual import ManualStrategy


def _trending_df(n=1500, seed=3, drift=0.00015):
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


# ---------------------------------------------------------------------------
# Deflated / Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_grows_with_trial_count():
    trial_sharpes = list(np.random.default_rng(0).normal(0, 0.5, 200))
    small = expected_max_sharpe(trial_sharpes, n_trials=5)
    large = expected_max_sharpe(trial_sharpes, n_trials=5000)
    assert large > small >= 0.0


def test_expected_max_sharpe_zero_for_single_trial_or_no_spread():
    assert expected_max_sharpe([1.0], n_trials=1) == 0.0
    assert expected_max_sharpe([1.0, 1.0, 1.0], n_trials=3) == 0.0  # zero std dev


def test_deflated_sharpe_same_observed_value_scores_lower_with_more_trials():
    """The exact same raw Sharpe should look LESS significant the more
    candidates it was picked as the best of -- this is the entire point of
    the correction, so it's the single most important behavior to pin down."""
    trial_sharpes = list(np.random.default_rng(1).normal(0.2, 0.6, 500))
    few = deflated_sharpe_ratio(2.0, trial_sharpes, n_trials=5, n_trade_returns=200)
    many = deflated_sharpe_ratio(2.0, trial_sharpes, n_trials=5000, n_trade_returns=200)
    assert many.probabilistic_sharpe <= few.probabilistic_sharpe
    assert many.benchmark_sharpe >= few.benchmark_sharpe


def test_deflated_sharpe_probability_in_valid_range():
    trial_sharpes = list(np.random.default_rng(2).normal(0, 0.4, 100))
    result = deflated_sharpe_ratio(1.5, trial_sharpes, n_trials=100, n_trade_returns=150)
    assert 0.0 <= result.probabilistic_sharpe <= 1.0
    assert result.is_significant == (result.probabilistic_sharpe >= result.significance_threshold)


def test_deflated_sharpe_handles_nonfinite_observed_sharpe():
    result = deflated_sharpe_ratio(float("nan"), [0.1, 0.2, 0.3], n_trials=3, n_trade_returns=50)
    assert math.isfinite(result.probabilistic_sharpe)
    assert result.observed_sharpe == 0.0


def test_deflated_sharpe_handles_tiny_trade_counts_without_crashing():
    result = deflated_sharpe_ratio(1.0, [0.1, 0.2], n_trials=2, n_trade_returns=1)
    assert math.isfinite(result.probabilistic_sharpe)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def test_walk_forward_returns_none_on_too_little_data():
    df = _trending_df(n=50)
    result = run_walk_forward(df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_folds=4)
    assert result is None


def test_walk_forward_runs_expected_fold_count_on_sufficient_data():
    df = _trending_df(n=3000)
    result = run_walk_forward(df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_folds=3)
    assert result is not None
    assert result.n_folds <= 3
    assert len(result.folds) == result.n_folds
    for fold in result.folds:
        assert fold.train_bars > 0
        # folds must be chronological and non-overlapping (train comes strictly before test)
        assert fold.train_period[1] <= fold.test_period[0]


def test_walk_forward_efficiency_is_stable_flag_matches_threshold():
    df = _trending_df(n=3000)
    result = run_walk_forward(
        df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_folds=3, stability_threshold=-999,
    )
    assert result.is_stable is True  # threshold so low nothing can fail it

    result2 = run_walk_forward(
        df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_folds=3, stability_threshold=999,
    )
    assert result2.is_stable is False  # threshold so high nothing can pass it


def test_walk_forward_builds_a_fresh_strategy_instance_each_call():
    calls = []

    def builder():
        calls.append(1)
        return ManualStrategy(_sma_config())

    df = _trending_df(n=2000)
    run_walk_forward(df, builder, RiskConfig(), n_folds=2)
    # 2 folds x (train + test) = 4 fresh instances, never one reused instance
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness
# ---------------------------------------------------------------------------

def _manual_spec(config=None):
    return {"source_type": "manual", "config": config or _sma_config()}


_PYTHON_SRC = '''STRATEGY_NAME = "Test EMA Cross"
EMA_FAST = 5
EMA_SLOW = 15
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    sig = (fast > slow).astype(int) - (fast < slow).astype(int)
    return sig
'''

_PINESCRIPT_SRC = '''//@version=5
strategy("Test", overlay=true)
fastLen = input.int(5, "Fast Length")
slowLen = input.int(15, "Slow Length")
fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
longCondition = ta.crossover(fast, slow)
shortCondition = ta.crossunder(fast, slow)
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
// T58_SL_PIPS=20
// T58_TP_PIPS=40
'''

_MQL5_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 15, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) { trade.Buy(0.1, _Symbol); }
   if (fastMA < slowMA) { trade.Sell(0.1, _Symbol); }
   // T58_SL_PIPS=20
   // T58_TP_PIPS=40
}
'''


def test_robustness_returns_none_when_no_tunable_parameters():
    spec = {"source_type": "manual", "config": {"name": "no params", "long_entry": "close > 0"}}
    df = _trending_df(n=500)
    result = parameter_neighborhood_robustness(
        spec, df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20), n_neighbors=3,
    )
    assert result is None


def test_robustness_runs_requested_neighbor_count():
    df = _trending_df(n=2000)
    result = parameter_neighborhood_robustness(
        _manual_spec(), df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=4, seed=1,
    )
    assert result is not None
    assert result.n_neighbors_tested == 4
    assert len(result.neighbor_fitnesses) == 4
    assert result.is_stable == (result.stability_ratio >= result.stability_threshold)


def test_robustness_is_reproducible_with_same_seed():
    df = _trending_df(n=2000)
    r1 = parameter_neighborhood_robustness(
        _manual_spec(), df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=3, seed=7,
    )
    r2 = parameter_neighborhood_robustness(
        _manual_spec(), df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=3, seed=7,
    )
    assert r1.neighbor_fitnesses == r2.neighbor_fitnesses


def test_robustness_works_for_python_candidate(tmp_path):
    path = tmp_path / "strat.py"
    path.write_text(_PYTHON_SRC, encoding="utf-8")
    spec = {"source_type": "python", "code_text": _PYTHON_SRC, "code_extension": ".py"}
    df = _trending_df(n=2000)
    result = parameter_neighborhood_robustness(
        spec, df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=3, seed=1, tmp_dir=tmp_path / "scratch",
    )
    assert result is not None
    assert result.n_neighbors_tested == 3


def test_robustness_works_for_pinescript_candidate():
    spec = {"source_type": "pinescript", "code_text": _PINESCRIPT_SRC, "code_extension": ".pine"}
    df = _trending_df(n=2000)
    result = parameter_neighborhood_robustness(
        spec, df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=3, seed=1,
    )
    assert result is not None
    assert result.n_neighbors_tested == 3


def test_robustness_works_for_mql5_candidate():
    spec = {"source_type": "mql5", "code_text": _MQL5_SRC, "code_extension": ".mq5"}
    df = _trending_df(n=2000)
    result = parameter_neighborhood_robustness(
        spec, df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_neighbors=3, seed=1,
    )
    assert result is not None
    assert result.n_neighbors_tested == 3


def test_robustness_python_candidate_requires_tmp_dir(tmp_path):
    spec = {"source_type": "python", "code_text": _PYTHON_SRC, "code_extension": ".py"}
    df = _trending_df(n=500)
    with pytest.raises(Exception):
        parameter_neighborhood_robustness(
            spec, df, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=10),
            n_neighbors=2, seed=1, tmp_dir=None,
        )
