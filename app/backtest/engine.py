"""
Backtest engine.

Orchestrates: Dataset + Strategy + Risk/Execution Configuration
           -> Trade List + Equity Curve + Statistics
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskConfig) -> BacktestResult:
    """
    df: standardized OHLCV DataFrame (see app.data.importer)
    strategy: any Strategy subclass instance (manual/python/pinescript/mql5)
    risk: RiskConfig describing sizing, costs, and execution assumptions
    """
    strat_result: StrategyResult = strategy.generate(df)

    trades, equity_curve = run_execution(
        df=df,
        signals=strat_result.signals,
        risk=risk,
        stop_loss_pips=strat_result.stop_loss_pips,
        take_profit_pips=strat_result.take_profit_pips,
    )

    stats = compute_statistics(trades, equity_curve, initial_balance=risk.initial_balance)

    return BacktestResult(
        strategy_name=strat_result.name,
        trades=trades,
        equity_curve=equity_curve,
        statistics=stats,
        initial_balance=risk.initial_balance,
    )
