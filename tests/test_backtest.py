import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy


def _trending_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        drift = 0.00003 + rng.normal(0, 0.00004)
        o = price
        c = o + drift
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_strategy():
    return ManualStrategy({
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
    })


def test_backtest_runs_and_produces_trades():
    df = _trending_df()
    risk = RiskConfig(initial_balance=10000, pip_size=0.0001)
    result = run_backtest(df, _sma_strategy(), risk)
    assert result.statistics.total_trades >= 1
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve["equity"].iloc[0] > 0


def test_backtest_statistics_fields_present():
    df = _trending_df()
    risk = RiskConfig(initial_balance=10000, pip_size=0.0001)
    result = run_backtest(df, _sma_strategy(), risk)
    stats = result.statistics.to_dict()
    for field in ["net_profit", "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct"]:
        assert field in stats


def test_backtest_no_signals_produces_no_trades():
    df = _trending_df(n=50)
    risk = RiskConfig(initial_balance=10000)
    strat = ManualStrategy({
        "name": "never",
        "long_entry": "close > 999999",
    })
    result = run_backtest(df, strat, risk)
    assert result.statistics.total_trades == 0
    assert result.equity_curve["equity"].iloc[-1] == risk.initial_balance
