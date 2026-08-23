import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_concentration_stats
from app.strategy.manual import ManualStrategy


def _trending_df(n=600, seed=1):
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
        "stop_loss_pips": 15,
        "take_profit_pips": 30,
    })


def test_trades_carry_initial_risk_and_it_drives_average_r():
    df = _trending_df()
    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=0.0001)
    result = run_backtest(df, _sma_strategy(), risk)
    assert len(result.trades) > 0
    # Every trade should have a positive initial_risk on record now.
    assert all(t.initial_risk is not None and t.initial_risk > 0 for t in result.trades)
    # average_r should be finite and derived from real R multiples, not the
    # old realized-loss approximation.
    assert np.isfinite(result.statistics.average_r)


def test_concentration_stats_flag_single_trade_dependence():
    df = _trending_df()
    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=0.0001)
    result = run_backtest(df, _sma_strategy(), risk)
    stats = compute_concentration_stats(result.trades)
    assert "net_profit_excluding_best_trade" in stats
    assert "net_profit_excluding_best_day" in stats
    # Removing the best trade can never increase net profit.
    net_profit = sum(t.pnl for t in result.trades)
    assert stats["net_profit_excluding_best_trade"] <= net_profit + 1e-9


def test_concentration_stats_empty_trades_is_safe():
    stats = compute_concentration_stats([])
    assert stats["best_trade_pnl"] == 0.0
    assert stats["net_profit_excluding_best_trade"] == 0.0


def test_holdout_comparison_splits_chronologically_and_runs_independently():
    df = _trending_df(n=1000)
    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=0.0001)
    holdout = run_holdout_comparison(df, _sma_strategy(), risk, holdout_frac=0.2)

    assert holdout["in_sample_bars"] + holdout["holdout_bars"] == len(df)
    assert holdout["holdout_bars"] == int(len(df) * 0.2)
    assert holdout["in_sample_statistics"] is not None
    assert holdout["holdout_statistics"] is not None
    # The holdout period must start strictly after the in-sample period ends.
    in_end = pd.Timestamp(holdout["in_sample_period"][1])
    out_start = pd.Timestamp(holdout["holdout_period"][0])
    assert out_start > in_end


def test_holdout_comparison_handles_too_little_data_gracefully():
    df = _trending_df(n=2)
    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=0.0001)
    # Should not raise even on a pathologically short dataframe.
    holdout = run_holdout_comparison(df, _sma_strategy(), risk, holdout_frac=0.2)
    assert holdout["in_sample_bars"] + holdout["holdout_bars"] == len(df)
