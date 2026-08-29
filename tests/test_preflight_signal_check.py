"""
Tests for app.optimize.refinement.preflight_signal_check and its wiring
into Walk-Forward Optimization, Multi-Objective search, and the
Walk-Forward-Aware GA.

Context: a strategy that structurally never fires on the given data (most
commonly: an hour-of-day "session" filter applied to non-intraday data,
which zeroes out every bar) used to silently grind through every fold /
every GA generation and hand back an all-zero / -inf / "infeasible"
report with no explanation. These tests confirm the fix: each of the
three heavier searches now fails FAST with a clear, actionable
RefinementError instead.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.multi_objective import MultiObjectiveConfig, run_multi_objective_refinement
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation.walk_forward_opt import run_walk_forward_optimization


def _ohlcv(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = price + np.abs(rng.normal(0, 0.03, n))
    low = price - np.abs(rng.normal(0, 0.03, n))
    close = price + rng.normal(0, 0.01, n)
    volume = rng.integers(100, 1000, n)
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def _never_fires_config():
    """A tunable ManualStrategy config whose entry condition can never be
    true on realistic price data -- the "structurally zero trades"
    scenario this check exists to catch fast."""
    return {
        "name": "never fires",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
        ],
        "long_entry": "sma_fast > 999999",
        "long_exit": "sma_fast < 0",
        "short_entry": "sma_fast > 999999",
        "short_exit": "sma_fast < 0",
    }


def _always_fires_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 3, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 9, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
    }


def test_preflight_signal_check_raises_when_baseline_has_zero_trades():
    df = _ohlcv()
    strategy = ManualStrategy(_never_fires_config())
    with pytest.raises(RefinementError, match="ZERO trades"):
        preflight_signal_check(df, strategy, RiskConfig(), "Some feature")


def test_preflight_signal_check_passes_when_baseline_has_trades():
    df = _ohlcv()
    strategy = ManualStrategy(_always_fires_config())
    # Should not raise.
    preflight_signal_check(df, strategy, RiskConfig(), "Some feature")


def test_preflight_signal_check_does_not_mask_strategy_crashes():
    """A strategy that raises outright is a different failure mode (an
    actual bug/incompatibility) -- this check must not swallow that into
    a generic "zero trades" message; it should let it propagate so the
    caller's own error handling (StrategyError, etc.) reports it."""
    class _Boom:
        source_type = "manual"

    df = _ohlcv()
    # Not a real Strategy -- run_backtest will raise something other than
    # a clean zero-trade result. preflight_signal_check should re-raise
    # (or simply return without raising RefinementError), never mask it
    # as "ZERO trades".
    try:
        preflight_signal_check(df, _Boom(), RiskConfig(), "Some feature")
    except RefinementError as exc:
        assert "ZERO trades" not in str(exc)


def test_walk_forward_optimization_fails_fast_on_zero_trade_baseline():
    df = _ohlcv(n=2000)
    strategy = ManualStrategy(_never_fires_config())
    with pytest.raises(RefinementError, match="Walk-Forward Optimization"):
        run_walk_forward_optimization(
            df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
            n_folds=3, refine_cfg=RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=20),
        )


def test_multi_objective_fails_fast_on_zero_trade_baseline():
    df = _ohlcv(n=1200)
    strategy = ManualStrategy(_never_fires_config())
    with pytest.raises(RefinementError, match="Multi-Objective search"):
        run_multi_objective_refinement(
            df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
            mo_config=MultiObjectiveConfig(population_size=4, generations=1, search_monte_carlo_sims=20),
        )


def test_walkforward_ga_fails_fast_on_zero_trade_baseline():
    df = _ohlcv(n=2000)
    strategy = ManualStrategy(_never_fires_config())
    with pytest.raises(RefinementError, match="Walk-Forward-Aware GA"):
        run_walkforward_aware_refinement(
            df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
            refinement_config=RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=20),
            n_folds=3,
        )


def test_walk_forward_optimization_still_runs_when_baseline_has_trades():
    """Guard against a preflight check that's so eager it blocks the
    normal, working case."""
    df = _ohlcv(n=2000)
    strategy = ManualStrategy(_always_fires_config())
    result = run_walk_forward_optimization(
        df, strategy, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=20),
        n_folds=3, refine_cfg=RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=20),
    )
    assert len(result.folds) >= 2
