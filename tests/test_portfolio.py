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
