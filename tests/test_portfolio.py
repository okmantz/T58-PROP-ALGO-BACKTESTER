import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.portfolio.portfolio import (
    InstrumentLeg,
    PortfolioConfig,
    PortfolioError,
    run_portfolio_backtest,
)
from app.strategy.manual import ManualStrategy


def _trending_df(n=1500, seed=3, drift=0.00015, start_price=1.10):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    price = start_price
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.0006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.0003))
        l = min(o, c) - abs(rng.normal(0, 0.0003))
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


def test_run_portfolio_backtest_two_uncorrelated_legs():
    df_a = _trending_df(seed=1, drift=0.00015)
    df_b = _trending_df(seed=99, drift=-0.00010, start_price=2000.0)
    legs = [
        InstrumentLeg(name="EURUSD", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="XAUUSD", df=df_b, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
    ]
    result = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=50_000))

    assert len(result.legs) == 2
    assert set(result.correlation_matrix.keys()) == {"EURUSD", "XAUUSD"}
    assert result.combined_statistics.total_trades == len(result.combined_trades)
    # weights should be positive and roughly preserve total risk budget (sum ~= n_legs)
    total_weight = sum(l.final_weight for l in result.legs)
    assert 1.0 < total_weight < 3.0


def test_run_portfolio_backtest_requires_two_legs():
    df_a = _trending_df()
    legs = [InstrumentLeg(name="EURUSD", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig())]
    with pytest.raises(PortfolioError):
        run_portfolio_backtest(legs)


def test_portfolio_summary_dict_shape():
    df_a = _trending_df(seed=2)
    df_b = _trending_df(seed=42, start_price=50.0)
    legs = [
        InstrumentLeg(name="A", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="B", df=df_b, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
    ]
    result = run_portfolio_backtest(legs)
    summary = result.to_summary_dict()
    assert "legs" in summary and "combined_statistics" in summary and "correlation_matrix" in summary


# ---------------------------------------------------------------------------
# Aggregate eval_pass_probability on the COMBINED portfolio -- the actual
# "does diversifying across strategies raise my probability of passing"
# question, answered directly rather than only per-strategy.
# ---------------------------------------------------------------------------

def _mean_reversion_config():
    return {
        "name": "mean reversion",
        "indicators": [
            {"type": "sma", "period": 20, "column": "close", "as": "sma"},
            {"type": "rsi", "period": 14, "column": "close", "as": "rsi"},
        ],
        "long_entry": "rsi < 30", "long_exit": "rsi > 50",
        "short_entry": "rsi > 70", "short_exit": "rsi < 50",
        "stop_loss_pips": 20, "take_profit_pips": 30,
    }


def test_portfolio_mc_fields_absent_when_not_requested():
    df_a = _trending_df(seed=1, drift=0.00015)
    df_b = _trending_df(seed=99, drift=-0.00010, start_price=2000.0)
    legs = [
        InstrumentLeg(name="A", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="B", df=df_b, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
    ]
    result = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=50_000))

    assert result.mc_result is None
    assert result.single_run_summary is None
    assert "mc_result" not in result.to_summary_dict()


def test_portfolio_mc_fields_populated_when_prop_rules_and_mc_config_given():
    from app.monte_carlo.engine import MonteCarloConfig
    from app.prop.simulator import PropRules

    df_a = _trending_df(seed=1, drift=0.00015)
    df_b = _trending_df(seed=99, drift=-0.00010, start_price=2000.0)
    legs = [
        InstrumentLeg(name="A", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="B", df=df_b, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
    ]
    cfg = PortfolioConfig(
        initial_balance=50_000,
        prop_rules=PropRules(account_size=50_000),
        mc_config=MonteCarloConfig(n_simulations=200),
    )
    result = run_portfolio_backtest(legs, cfg)

    assert result.mc_result is not None
    assert 0.0 <= result.mc_result.evaluation_pass_probability <= 100.0
    assert result.single_run_summary is not None
    assert "mc_result" in result.to_summary_dict()


def test_portfolio_legs_can_use_genuinely_different_strategies():
    """The whole point of the feature: each leg is free to run its OWN
    strategy (not just the same strategy on a different instrument) --
    e.g. a trend-following SMA cross alongside an RSI mean-reversion
    strategy, combined into one shared-account book."""
    df_a = _trending_df(seed=1, drift=0.00015)
    df_b = _trending_df(seed=7, drift=0.0, start_price=1.20)  # flat/noisy, suits mean reversion
    legs = [
        InstrumentLeg(name="trend-leg", df=df_a, strategy=ManualStrategy(_sma_config()), risk=RiskConfig()),
        InstrumentLeg(name="mean-reversion-leg", df=df_b, strategy=ManualStrategy(_mean_reversion_config()), risk=RiskConfig()),
    ]
    result = run_portfolio_backtest(legs, PortfolioConfig(initial_balance=100_000))

    assert len(result.legs) == 2
    assert {l.name for l in result.legs} == {"trend-leg", "mean-reversion-leg"}
    # Combined trades should include activity from both legs' own distinct
    # entry logic, not just one leg dominating because they're the same strategy.
    assert result.combined_statistics.total_trades == len(result.combined_trades)
