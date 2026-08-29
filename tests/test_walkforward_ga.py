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


def test_ai_suggest_cb_seeds_the_population():
    """A working ai_suggest_cb should get its genomes evaluated as part of
    generation 0's population, not silently ignored."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=1, search_monte_carlo_sims=30)

    calls = []

    def fake_ai_suggest(genes):
        calls.append(len(genes))
        return [[g.base_value for g in genes]]  # trivially valid: the baseline genome again

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
        ai_suggest_cb=fake_ai_suggest,
    )
    assert len(calls) >= 1
    assert result.best is not None


def test_ai_suggest_cb_exception_does_not_break_the_search():
    """A misbehaving/unreachable AI callback must be treated exactly like
    AI assist being off -- never allowed to crash or block the GA."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=1, search_monte_carlo_sims=30)

    def broken_ai_suggest(genes):
        raise RuntimeError("Ollama is not running")

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
        ai_suggest_cb=broken_ai_suggest,
    )
    assert result.best is not None
    assert len(result.leaderboard) == refine_cfg.population_size


def test_ai_suggest_cb_with_wrong_length_genome_is_ignored():
    """A genome that doesn't match the discovered gene count (e.g. a
    hallucinated response) must be silently skipped, not passed to the
    backtester with mismatched parameters."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=1, search_monte_carlo_sims=30)

    def wrong_length_ai_suggest(genes):
        return [[1.0, 2.0, 3.0, 4.0, 5.0]]  # deliberately wrong length

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
        ai_suggest_cb=wrong_length_ai_suggest,
    )
    assert result.best is not None


def test_ai_suggest_cb_two_argument_form_receives_population_snapshot():
    """A callback declaring a second parameter should receive the prior
    generation's [(genome, fitness), ...] snapshot -- empty on generation
    0, non-empty (and the right size) on every generation after."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=30)

    seen_population_sizes = []

    def population_aware_ai_suggest(genes, population):
        seen_population_sizes.append(len(population))
        return [[g.base_value for g in genes]]

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
        ai_suggest_cb=population_aware_ai_suggest,
    )
    assert result.best is not None
    # Generation 0 call gets an empty snapshot; every later generation's
    # call gets the full prior population.
    assert seen_population_sizes[0] == 0
    assert all(size == refine_cfg.population_size for size in seen_population_sizes[1:])
    assert len(seen_population_sizes) >= 2


def test_ai_suggest_cb_single_argument_form_still_works_unchanged():
    """Existing one-argument callbacks (no population parameter) must
    keep working exactly as before -- detected via signature inspection,
    not a breaking API change."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=30)

    calls = []

    def one_arg_ai_suggest(genes):
        calls.append(len(genes))
        return [[g.base_value for g in genes]]

    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3,
        ai_suggest_cb=one_arg_ai_suggest,
    )
    assert result.best is not None
    assert len(calls) >= 2
