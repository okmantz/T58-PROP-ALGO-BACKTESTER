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


def test_daily_loss_limit_force_closes_on_intrabar_floating_loss():
    """
    A long trade whose bar dips deep underwater intrabar (breaching the
    daily loss floor) but recovers by the bar's close must still be
    force-closed -- a real funded account would have been auto-liquidated
    the moment floating equity crossed the daily floor, regardless of
    where the bar eventually closes. Checking only realized end-of-bar
    P&L would incorrectly let this trade ride.
    """
    from app.backtest.execution import run_execution

    ts = pd.date_range("2024-01-01 09:00", periods=4, freq="15min")
    # Bar 0: enter long at close via signal.
    # Bar 1: gaps down hard intrabar (low far below entry) then recovers
    #        to close near flat -- floating loss at the low must breach
    #        the daily limit even though the close looks fine.
    rows = [
        (ts[0], 100.0, 100.2, 99.9, 100.0, 100.0),
        (ts[1], 100.0, 100.1, 90.0, 99.9, 100.0),   # deep intrabar wick down
        (ts[2], 99.9, 100.5, 99.8, 100.4, 100.0),
        (ts[3], 100.4, 100.6, 100.3, 100.5, 100.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 1, 1, 1])

    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=1.0,
        pip_size=1.0, daily_loss_limit_pct=5.0,  # 5% of 10,000 = $500 floor
    )
    trades, equity_df = run_execution(
        df, signals, risk, stop_loss_pips=None, take_profit_pips=None,
    )

    # Sizing at 1% risk with the engine's 1%-of-price fallback stop
    # (no stop defined) means the position is large enough that the
    # bar-1 wick (10 points against a ~1-point fallback stop) is a
    # catastrophic move well past the $500 daily floor.
    reasons = [t.exit_reason for t in trades]
    assert "daily_loss_limit_forced_close" in reasons
    forced = next(t for t in trades if t.exit_reason == "daily_loss_limit_forced_close")
    # Forced close must happen on bar 1 (the wick bar), not later.
    assert forced.exit_time == ts[1]


def test_equity_curve_reflects_mark_to_market_not_only_realized():
    """
    The equity curve must move while a trade is open and price is moving
    against/for it, not just jump at trade close -- otherwise drawdown
    statistics silently ignore intrabar/multi-bar floating losses.
    """
    from app.backtest.execution import run_execution

    ts = pd.date_range("2024-01-01 09:00", periods=3, freq="15min")
    rows = [
        (ts[0], 100.0, 100.1, 99.9, 100.0, 100.0),
        (ts[1], 100.0, 100.1, 99.9, 95.0, 100.0),   # large adverse close, trade stays open
        (ts[2], 95.0, 95.1, 94.9, 95.0, 100.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 1, 1])

    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=1.0)
    trades, equity_df = run_execution(
        df, signals, risk, stop_loss_pips=None, take_profit_pips=None,
    )
    # Equity at bar 1 (still open, deep in floating loss) must already be
    # below initial balance -- not flat at initial balance because
    # nothing has "realized" yet.
    equity_bar1 = equity_df.iloc[1]["equity"]
    assert equity_bar1 < risk.initial_balance


def test_eod_drawdown_mode_survives_intraday_dip_that_recovers_by_close():
    """
    In 'eod' drawdown_check_mode, a day with two trades -- a big intraday
    loss followed by a recovery, netting a SMALL loss for the day overall
    -- must NOT fail on the first (losing) trade the way 'intrabar' mode
    would. Only the day's final cumulative P&L matters.
    """
    from app.prop.simulator import PropRules, simulate_account

    rules_intrabar = PropRules(
        account_size=100_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=50.0,
        consistency_rule_pct=None, min_trading_days=1, drawdown_check_mode="intrabar",
    )
    rules_eod = PropRules(
        account_size=100_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=50.0,
        consistency_rule_pct=None, min_trading_days=1, drawdown_check_mode="eod",
    )
    # Same day: first trade loses $6,000 (> 5% of 100k = $5,000 -- breaches
    # intrabar), second trade (same day) recovers $5,500, net day = -$500.
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")]
    pnls = [-6000.0, 5500.0]

    result_intrabar = simulate_account(pnls, dates, rules_intrabar)
    result_eod = simulate_account(pnls, dates, rules_eod)

    assert result_intrabar.failed and result_intrabar.failure_reason == "daily_loss_limit"
    assert not result_eod.failed
