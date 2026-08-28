import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.multi_objective import (
    MultiObjectiveConfig,
    _dominates,
    fast_non_dominated_sort,
    run_multi_objective_refinement,
)
from app.optimize.parameter_space import RefinementError
from app.prop.simulator import PropRules
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
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_dominates_basic():
    assert _dominates((2.0, 2.0), (1.0, 1.0))
    assert _dominates((2.0, 1.0), (1.0, 1.0))
    assert not _dominates((1.0, 1.0), (1.0, 1.0))
    assert not _dominates((1.0, 2.0), (2.0, 1.0))


def test_fast_non_dominated_sort_simple():
    class C:
        def __init__(self, sv):
            self._sort_values = sv
            self.rank = -1

    pop = [C((3, 3)), C((1, 1)), C((2, 2)), C((3, 1)), C((1, 3))]
    fronts = fast_non_dominated_sort(pop)
    assert 0 in fronts[0]  # (3,3) dominates everything, must be in front 0
    assert all(isinstance(f, list) for f in fronts)


def test_rejects_single_objective():
    with pytest.raises(RefinementError):
        MultiObjectiveConfig(objectives=["sharpe_ratio"])


def test_rejects_unknown_objective():
    with pytest.raises(RefinementError):
        MultiObjectiveConfig(objectives=["sharpe_ratio", "not_a_real_metric"])


def test_run_multi_objective_refinement_produces_pareto_front():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    mo_cfg = MultiObjectiveConfig(
        objectives=["sharpe_ratio", "max_drawdown_pct"],
        population_size=8, generations=2, search_monte_carlo_sims=30, random_seed=1,
    )

    result = run_multi_objective_refinement(df, strategy, risk, rules, mc_cfg, mo_cfg)

    assert len(result.pareto_front) >= 1
    assert len(result.pareto_front) <= mo_cfg.population_size
    # no candidate in the front should dominate another candidate in the front
    front_vals = [c._sort_values for c in result.pareto_front]
    for i, a in enumerate(front_vals):
        for j, b in enumerate(front_vals):
            if i != j:
                assert not _dominates(a, b)
    for c in result.pareto_front:
        assert c.config is not None
