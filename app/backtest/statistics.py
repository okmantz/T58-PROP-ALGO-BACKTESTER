"""
Backtest statistics.

Computes the full statistics set required by the product spec: returns,
win/loss, risk, strategy-quality, and risk-adjusted metrics, from a trade
list and an equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from app.backtest.execution import Trade


@dataclass
class BacktestStatistics:
    net_profit: float
    gross_profit: float
    gross_loss: float
    return_pct: float
    average_trade: float

    win_rate: float
    loss_rate: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float

    max_drawdown: float
    max_drawdown_pct: float
    average_drawdown_pct: float
    max_daily_drawdown_pct: float
    max_weekly_drawdown_pct: float
    max_losing_streak: int
    max_winning_streak: int

    profit_factor: float
    expectancy: float
    average_r: float
    risk_reward: float

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    total_trades: int

    def to_dict(self) -> dict:
        return asdict(self)


def _max_streak(bools: list[bool]) -> int:
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return dd.fillna(0.0)


def _periodic_max_drawdown(equity_df: pd.DataFrame, freq: str) -> float:
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    worst = 0.0
    for _, grp in df.resample(freq):
        if grp.empty:
            continue
        dd = _drawdown_series(grp["equity"])
        worst = min(worst, dd.min())
    return abs(worst) * 100.0


def compute_statistics(
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    initial_balance: float,
    bars_per_year: float = 252 * 78,  # rough default for intraday FX; overridable
) -> BacktestStatistics:
    if not trades:
        return BacktestStatistics(
            net_profit=0, gross_profit=0, gross_loss=0, return_pct=0, average_trade=0,
            win_rate=0, loss_rate=0, average_winner=0, average_loser=0,
            largest_winner=0, largest_loser=0,
            max_drawdown=0, max_drawdown_pct=0, average_drawdown_pct=0,
            max_daily_drawdown_pct=0, max_weekly_drawdown_pct=0,
            max_losing_streak=0, max_winning_streak=0,
            profit_factor=0, expectancy=0, average_r=0, risk_reward=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=0,
        )

    pnls_raw = np.array([t.pnl for t in trades])
    finite_mask = np.isfinite(pnls_raw)
    n_bad = int((~finite_mask).sum())
    pnls = pnls_raw[finite_mask]
    if n_bad:
        # A non-finite trade pnl indicates a data or configuration anomaly
        # (e.g. an indicator warm-up period, a near-zero ATR stop distance,
        # or similar edge case). Excluding it keeps the rest of the report
        # trustworthy instead of letting one bad value poison every
        # aggregate statistic (which otherwise silently turns into NaN).
        import warnings
        warnings.warn(
            f"Excluded {n_bad} trade(s) with a non-finite P&L from the backtest "
            "statistics. Check your risk settings (in particular pip size vs. "
            "the instrument's actual price scale, and any ATR-multiple stop) "
            "if this number is large.",
            RuntimeWarning,
        )
    if not len(pnls):
        return BacktestStatistics(
            net_profit=0, gross_profit=0, gross_loss=0, return_pct=0, average_trade=0,
            win_rate=0, loss_rate=0, average_winner=0, average_loser=0,
            largest_winner=0, largest_loser=0,
            max_drawdown=0, max_drawdown_pct=0, average_drawdown_pct=0,
            max_daily_drawdown_pct=0, max_weekly_drawdown_pct=0,
            max_losing_streak=0, max_winning_streak=0,
            profit_factor=0, expectancy=0, average_r=0, risk_reward=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=len(trades),
        )
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    net_profit = float(pnls.sum())
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    return_pct = (net_profit / initial_balance) * 100 if initial_balance else 0.0
    average_trade = float(pnls.mean())

    win_rate = len(wins) / len(pnls) * 100
    loss_rate = len(losses) / len(pnls) * 100
    average_winner = float(wins.mean()) if len(wins) else 0.0
    average_loser = float(losses.mean()) if len(losses) else 0.0
    largest_winner = float(wins.max()) if len(wins) else 0.0
    largest_loser = float(losses.min()) if len(losses) else 0.0

    equity = equity_curve["equity"]
    running_max = equity.cummax()
    dd_abs = equity - running_max
    max_drawdown = float(dd_abs.min())
    dd_pct_series = _drawdown_series(equity)
    max_drawdown_pct = float(abs(dd_pct_series.min()) * 100)
    average_drawdown_pct = float(abs(dd_pct_series[dd_pct_series < 0].mean()) * 100) if (dd_pct_series < 0).any() else 0.0
    max_daily_dd = _periodic_max_drawdown(equity_curve, "1D")
    max_weekly_dd = _periodic_max_drawdown(equity_curve, "1W")

    win_streak = _max_streak(list(pnls > 0))
    loss_streak = _max_streak(list(pnls <= 0))

    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy = float(average_trade)
    # Average R approximated as average pnl / average risked amount per trade (risk-normalized return)
    risk_per_trade = [abs(t.pnl) for t in trades if t.pnl <= 0]
    avg_risk = float(np.mean(risk_per_trade)) if risk_per_trade else (abs(average_loser) or 1.0)
    average_r = float(average_trade / avg_risk) if avg_risk else 0.0
    risk_reward = float(abs(average_winner / average_loser)) if average_loser != 0 else float("inf") if average_winner > 0 else 0.0

    # Risk-adjusted ratios computed on per-trade returns (simple, MVP-appropriate approach)
    trade_returns = pnls / initial_balance if initial_balance else pnls
    mean_ret = trade_returns.mean()
    std_ret = trade_returns.std(ddof=1) if len(trade_returns) > 1 else 0.0
    downside = trade_returns[trade_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

    sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(len(trade_returns))) if std_ret else 0.0
    sortino_ratio = float((mean_ret / downside_std) * np.sqrt(len(trade_returns))) if downside_std else 0.0
    calmar_ratio = float(return_pct / max_drawdown_pct) if max_drawdown_pct else 0.0

    return BacktestStatistics(
        net_profit=net_profit, gross_profit=gross_profit, gross_loss=gross_loss,
        return_pct=return_pct, average_trade=average_trade,
        win_rate=win_rate, loss_rate=loss_rate,
        average_winner=average_winner, average_loser=average_loser,
        largest_winner=largest_winner, largest_loser=largest_loser,
        max_drawdown=max_drawdown, max_drawdown_pct=max_drawdown_pct,
        average_drawdown_pct=average_drawdown_pct,
        max_daily_drawdown_pct=max_daily_dd, max_weekly_drawdown_pct=max_weekly_dd,
        max_losing_streak=loss_streak, max_winning_streak=win_streak,
        profit_factor=profit_factor, expectancy=expectancy,
        average_r=average_r, risk_reward=risk_reward,
        sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio, calmar_ratio=calmar_ratio,
        total_trades=len(trades),
    )


def compute_cost_ladder(trades: list, rungs_pct: list[float] | None = None) -> list[dict]:
    """
    Re-costs the SAME historical trade sequence at increasing round-turn
    friction levels and reports net profit / profit factor at each rung.

    This is the single most emphasized practice in serious strategy
    validation: a "real" edge should survive some added friction, and
    should die gracefully (not from 3.0 to 0.0 profit factor between two
    adjacent rungs) as costs rise. An edge that only exists at 0% added
    cost is not an edge you can trade — it's the cost model doing the
    lying for you.

    rungs_pct: extra ROUND-TURN cost, as a fraction of notional (entry
    price x size), applied on top of whatever commission/spread/slippage
    the backtest already modeled. Defaults to 0%, 0.05%, 0.10%, 0.25% per
    side (i.e. the exact ladder the falsification-kit methodology uses).
    """
    if rungs_pct is None:
        rungs_pct = [0.0, 0.0005, 0.0010, 0.0025]

    notionals = np.array([abs(t.entry_price * t.size) for t in trades]) if trades else np.array([])
    base_pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    finite_mask = np.isfinite(base_pnls)
    notionals = notionals[finite_mask]
    base_pnls = base_pnls[finite_mask]

    ladder = []
    for rung in rungs_pct:
        extra_cost = notionals * rung
        pnls = base_pnls - extra_cost
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss = float(pnls[pnls <= 0].sum())
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (float("inf") if gross_profit > 0 else 0.0)
        ladder.append({
            "extra_cost_pct_per_trade": rung * 100,
            "net_profit": float(pnls.sum()) if len(pnls) else 0.0,
            "profit_factor": profit_factor,
            "win_rate": float((pnls > 0).sum() / len(pnls) * 100) if len(pnls) else 0.0,
        })
    return ladder
