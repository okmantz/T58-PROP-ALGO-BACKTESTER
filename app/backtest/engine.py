"""
Backtest engine.

Orchestrates: Dataset + Strategy + Risk/Execution Configuration
           -> Trade List + Equity Curve + Statistics
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig
from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.strategy.base import Strategy, StrategyResult


@dataclass
class BacktestResult:
    strategy_name: str
    trades: list[Trade]
    equity_curve: pd.DataFrame
    statistics: BacktestStatistics
    initial_balance: float


def run_holdout_comparison(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    holdout_frac: float = 0.2,
) -> dict:
    """
    Chronological in-sample / out-of-sample split, run ONCE.

    Splits df at a single time point (default: the last 20% of bars becomes
    the holdout), runs the identical strategy + risk config independently on
    each half, and returns both statistics sets side by side. Each half is
    passed to the strategy as its own standalone DataFrame -- higher-timeframe
    context the strategy derives internally (e.g. resampling to 1H/4H) is
    therefore also correctly confined to that half, with no leakage of
    future (holdout-period) price action into the in-sample run or vice
    versa.

    This is deliberately NOT a walk-forward optimizer and does not re-tune
    any parameters between segments -- it answers a narrower, falsification-
    style question: "does this exact strategy, as written, keep working on
    data it has never touched?" A strategy whose edge is real should degrade
    gracefully, not evaporate or invert, on the holdout segment.
    """
    n = len(df)
    split_idx = int(n * (1 - holdout_frac))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else n

    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)

    in_sample_result = run_backtest(in_sample_df, strategy, risk) if len(in_sample_df) else None
    holdout_result = run_backtest(holdout_df, strategy, risk) if len(holdout_df) else None

    return {
        "holdout_frac": holdout_frac,
        "in_sample_period": (
            (str(in_sample_df["timestamp"].iloc[0]), str(in_sample_df["timestamp"].iloc[-1]))
            if len(in_sample_df) else (None, None)
        ),
        "holdout_period": (
            (str(holdout_df["timestamp"].iloc[0]), str(holdout_df["timestamp"].iloc[-1]))
            if len(holdout_df) else (None, None)
        ),
        "in_sample_bars": len(in_sample_df),
        "holdout_bars": len(holdout_df),
        "in_sample_statistics": in_sample_result.statistics.to_dict() if in_sample_result else None,
        "holdout_statistics": holdout_result.statistics.to_dict() if holdout_result else None,
    }


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    adaptive_risk: AdaptiveRiskConfig | None = None,
) -> BacktestResult:
    """
    df: standardized OHLCV DataFrame (see app.data.importer)
    strategy: any Strategy subclass instance (manual/python/pinescript/mql5)
    risk: RiskConfig describing sizing, costs, and execution assumptions
    adaptive_risk: optional declarative money-management overlay (see
        app.backtest.adaptive_risk) -- None/omitted runs exactly as before
        this parameter existed.
    """
    strat_result: StrategyResult = strategy.generate(df)

    trades, equity_curve = run_execution(
        df=df,
        signals=strat_result.signals,
        risk=risk,
        stop_loss_pips=strat_result.stop_loss_pips,
        take_profit_pips=strat_result.take_profit_pips,
        stop_loss_distance=strat_result.stop_loss_distance,
        take_profit_distance=strat_result.take_profit_distance,
        trailing_stop_distance=strat_result.trailing_stop_distance,
        breakeven_trigger_r=strat_result.breakeven_trigger_r,
        adaptive_risk=adaptive_risk,
    )

    stats = compute_statistics(trades, equity_curve, initial_balance=risk.initial_balance)

    return BacktestResult(
        strategy_name=strat_result.name,
        trades=trades,
        equity_curve=equity_curve,
        statistics=stats,
        initial_balance=risk.initial_balance,
    )
