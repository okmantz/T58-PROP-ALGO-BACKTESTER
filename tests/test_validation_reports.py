import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.multi_objective import MultiObjectiveConfig, run_multi_objective_refinement
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, run_portfolio_backtest
from app.prop.simulator import PropRules
from app.reports.validation_reports import (
    generate_cpcv_report,
    generate_multi_objective_report,
    generate_pbo_report,
    generate_portfolio_report,
    generate_sensitivity_report,
    generate_walk_forward_report,
    generate_walkforward_ga_report,
)
from app.strategy.manual import ManualStrategy
from app.validation.cpcv import compute_pbo, run_cpcv
from app.validation.sensitivity import compute_1d_sensitivity, compute_2d_heatmap
from app.validation.walk_forward_opt import run_walk_forward_optimization


def _trending_df(n=2400, seed=3, drift=0.00015, start_price=1.10):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = start_price
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


def _sma_config(fast=5, slow=15):
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_generate_walk_forward_report(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk, rules = RiskConfig(), PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=50)
    refine_cfg = RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=30)
    result = run_walk_forward_optimization(df, strategy, risk, rules, mc_cfg, n_folds=3, refine_cfg=refine_cfg, random_seed=1)
    paths = generate_walk_forward_report(tmp_path, result)
    assert paths["json"].exists() and paths["html"].exists()
    assert "<svg" in paths["html"].read_text()


def test_generate_cpcv_and_pbo_reports(tmp_path):
    df = _trending_df(n=3000, seed=5)
    risk = RiskConfig()
    cpcv_result = run_cpcv(df, lambda: ManualStrategy(_sma_config()), risk, n_groups=6, n_test_groups=2, max_paths=6)
    paths = generate_cpcv_report(tmp_path, cpcv_result)
    assert paths["json"].exists() and paths["html"].exists()

    specs = [{"source_type": "manual", "config": _sma_config(5, 15)}, {"source_type": "manual", "config": _sma_config(8, 21)}]
    pbo_result = compute_pbo(df, specs, risk, n_groups=6, n_test_groups=2, max_paths=6)
    pbo_paths = generate_pbo_report(tmp_path, pbo_result)
    assert pbo_paths["json"].exists() and pbo_paths["html"].exists()


def test_generate_sensitivity_report(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk, rules = RiskConfig(), PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    sweeps = compute_1d_sensitivity(df, strategy, risk, rules, mc_cfg, n_steps=5)
    from app.validation.sensitivity import list_tunable_parameters
    labels = [l for l in list_tunable_parameters(strategy) if "period" in l]
    heatmap = compute_2d_heatmap(df, strategy, risk, rules, mc_cfg, labels[0], labels[1], n_steps=4)
    paths = generate_sensitivity_report(tmp_path, sweeps, heatmap)
    assert paths["json"].exists() and paths["html"].exists()
    assert "<svg" in paths["html"].read_text()


def test_generate_portfolio_report(tmp_path):
    df_a = _trending_df(seed=1)
    df_b = _trending_df(seed=99, drift=-0.0001, start_price=2000.0)
    legs = [
        InstrumentLeg(name="EURUSD", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="XAUUSD", df=df_b, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
    ]
    result = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=50_000))
    paths = generate_portfolio_report(tmp_path, result)
    assert paths["json"].exists() and paths["html"].exists()


def test_generate_multi_objective_report(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk, rules = RiskConfig(), PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    mo_cfg = MultiObjectiveConfig(objectives=["sharpe_ratio", "max_drawdown_pct"], population_size=8, generations=2, search_monte_carlo_sims=20, random_seed=1)
    result = run_multi_objective_refinement(df, strategy, risk, rules, mc_cfg, mo_cfg)
    paths = generate_multi_objective_report(tmp_path, result)
    assert paths["json"].exists() and paths["html"].exists()


def test_generate_walkforward_ga_report(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk, rules = RiskConfig(), PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(population_size=6, generations=2, search_monte_carlo_sims=20)
    result = run_walkforward_aware_refinement(df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3)
    paths = generate_walkforward_ga_report(tmp_path, result)
    assert paths["json"].exists() and paths["html"].exists()
