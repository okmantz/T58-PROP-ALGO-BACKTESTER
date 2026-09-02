"""
Multi-asset / multi-strategy portfolio backtesting with correlation-aware
position sizing.

This app's engine (app.backtest.engine.run_backtest) is deliberately
single-instrument, single-open-position -- that's the right MVP shape for
validating one strategy on one instrument, and this module does not
rewrite it. Instead, a portfolio here is built in two honest, explicit
passes:

  Pass 1 (measure): run every instrument/strategy "leg" independently at
    its OWN nominal risk setting, to get each leg's own trade list and
    equity curve.

  Pass 2 (correlate + re-size): compute the correlation matrix of each
    leg's daily returns from Pass 1, derive a correlation-aware risk
    weight per leg (legs that move with the rest of the book get sized
    down; legs that diversify it keep more of their nominal risk), and
    re-run every leg with that adjusted RiskConfig.risk_value.

  Combine: every leg's final trades are merged into ONE chronological
  trade sequence and walked forward as a single shared account balance,
  which is what "trading a portfolio out of one prop account" actually
  means.

Known, explicitly-scoped limitations (an honest MVP, not a full
multi-asset event-driven engine):
  - The correlation-based re-weighting is a single static pass over the
    whole backtest window, not a rolling/intra-backtest rebalance. A
    strategy whose correlation structure changes materially over the
    test period will not have that captured.
  - Combining legs by chronological trade CLOSE time correctly aggregates
    realized P&L into one equity curve, but does not model shared margin
    or a portfolio-level max-concurrent-exposure cap across instruments
    that happen to have overlapping open trades. For most prop-firm
    single-account use cases (one balance, one drawdown floor, trading
    several instruments with the same account) this is the right model;
    for a margin-constrained multi-position book it understates risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.execution import Trade
from app.backtest.risk import RiskConfig
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy


class PortfolioError(Exception):
    """Raised when a portfolio backtest cannot proceed."""


@dataclass
class InstrumentLeg:
    name: str
    df: pd.DataFrame
    strategy: Strategy
    risk: RiskConfig
    weight: float = 1.0  # nominal (pre-correlation) relative risk weight


@dataclass
class PortfolioConfig:
    initial_balance: float = 100_000.0
    correlation_penalty_strength: float = 0.6   # 0 = ignore correlation, 1 = full inverse-correlation re-weighting
    max_instrument_weight_frac: float = 0.5      # cap on any one leg's share of total portfolio risk budget
    min_weight_frac: float = 0.15                # floor on any one leg's share of its OWN nominal weight
    # Optional: when both are supplied, run_portfolio_backtest also runs a
    # Monte Carlo simulation on the COMBINED (shared-account) trade
    # sequence and reports the portfolio's own eval_pass_probability --
    # answering "if I fund ONE account trading this whole book of
    # strategies, what's the probability of passing?" directly, rather
    # than only ever reporting that number per individual strategy.
    # Running several genuinely uncorrelated modest-edge strategies as one
    # portfolio raises this number more reliably than pushing any single
    # strategy's own pass probability higher, because it smooths out the
    # daily-loss/drawdown spikes any one strategy's losing streak would
    # otherwise cause alone. None (either) = skip the Monte Carlo step
    # entirely (old behavior, unchanged).
    prop_rules: PropRules | None = None
    mc_config: MonteCarloConfig | None = None


@dataclass
class LegResult:
    name: str
    nominal_weight: float
    avg_correlation_with_others: float
    final_weight: float
    risk_value_scale: float
    trade_count: int
    net_profit: float
    statistics: dict


@dataclass
class PortfolioResult:
    legs: list  # list[LegResult]
    correlation_matrix: dict  # {leg_name: {leg_name: corr}}
    combined_trades: list  # list[Trade], chronological
    combined_equity_curve: pd.DataFrame
    combined_statistics: BacktestStatistics
    diversification_ratio: float | None  # weighted-avg individual vol / portfolio vol; >1 means diversification helped
    warnings: list = field(default_factory=list)
    # Populated only when PortfolioConfig.prop_rules + mc_config were both
    # supplied -- the portfolio's OWN aggregate probability of clearing
    # the eval, as one shared account trading every leg together. None
    # otherwise (Monte Carlo wasn't requested).
    mc_result: object | None = None          # MonteCarloResult | None
    single_run_summary: dict | None = None   # summarize_single_run() on the one real combined trade sequence

    def to_summary_dict(self) -> dict:
        d = {
            "legs": [l.__dict__ for l in self.legs],
            "correlation_matrix": self.correlation_matrix,
            "combined_statistics": self.combined_statistics.to_dict(),
            "diversification_ratio": self.diversification_ratio,
            "warnings": self.warnings,
        }
        if self.mc_result is not None:
            d["mc_result"] = self.mc_result.to_dict()
            d["single_run_summary"] = self.single_run_summary
        return d


def _daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    ec = equity_curve.copy()
    ec["timestamp"] = pd.to_datetime(ec["timestamp"])
    ec = ec.set_index("timestamp").sort_index()
    daily = ec["equity"].resample("1D").last().ffill()
    return daily.pct_change().dropna()


def _rebuild_equity_curve(trades: list[Trade], initial_balance: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame({"timestamp": [], "equity": []})
    ordered = sorted(trades, key=lambda t: t.exit_time)
    equity = initial_balance
    rows = [{"timestamp": ordered[0].entry_time, "equity": initial_balance}]
    for t in ordered:
        equity += t.pnl if math.isfinite(t.pnl) else 0.0
        rows.append({"timestamp": t.exit_time, "equity": equity})
    return pd.DataFrame(rows)


def run_portfolio_backtest(
    legs: list[InstrumentLeg],
    config: PortfolioConfig | None = None,
) -> PortfolioResult:
    if len(legs) < 2:
        raise PortfolioError("Portfolio backtesting requires at least 2 instrument/strategy legs.")

    cfg = config or PortfolioConfig()
    warnings: list[str] = []

    # --- Pass 1: measure each leg independently at its nominal risk ---
    baseline_results = {}
    for leg in legs:
        bt = run_backtest(leg.df, leg.strategy, leg.risk)
        baseline_results[leg.name] = bt
        if not bt.trades:
            warnings.append(f"Leg '{leg.name}' produced zero trades at its nominal risk setting.")

    # --- Correlation matrix from Pass-1 daily returns ---
    returns_by_leg = {leg.name: _daily_returns(baseline_results[leg.name].equity_curve) for leg in legs}
    names = [leg.name for leg in legs]
    returns_df = pd.DataFrame(returns_by_leg).dropna(how="all")
    if len(returns_df) >= 3:
        corr_df = returns_df.corr().fillna(0.0)
    else:
        warnings.append(
            "Not enough overlapping daily return data to compute a reliable correlation "
            "matrix; treating all legs as uncorrelated (no re-weighting applied)."
        )
        corr_df = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

    correlation_matrix = {a: {b: float(corr_df.loc[a, b]) if a in corr_df.index and b in corr_df.columns else 0.0
                               for b in names} for a in names}

    avg_corr = {}
    for name in names:
        others = [correlation_matrix[name][other] for other in names if other != name]
        avg_corr[name] = float(np.mean(others)) if others else 0.0

    # --- Correlation-aware weight adjustment ---
    n = len(legs)
    raw_weights = {}
    for leg in legs:
        penalty = max(0.0, min(avg_corr[leg.name], 1.0)) * cfg.correlation_penalty_strength
        raw = leg.weight * (1.0 - penalty)
        floor = leg.weight * cfg.min_weight_frac
        raw_weights[leg.name] = max(raw, floor)

    total_raw = sum(raw_weights.values()) or 1.0
    scale = n / total_raw
    final_weights = {name: w * scale for name, w in raw_weights.items()}

    # Cap any one leg's share of the total portfolio risk budget, then renormalize once.
    cap = cfg.max_instrument_weight_frac * n
    capped = {name: min(w, cap) for name, w in final_weights.items()}
    total_capped = sum(capped.values()) or 1.0
    final_weights = {name: w * (n / total_capped) for name, w in capped.items()}

    # --- Pass 2: re-run each leg with its correlation-adjusted risk_value ---
    leg_results: list[LegResult] = []
    all_final_trades: list[Trade] = []
    for leg in legs:
        weight = final_weights[leg.name]
        nominal_weight = leg.weight
        scale_factor = weight / nominal_weight if nominal_weight else weight
        adjusted_risk = RiskConfig(
            initial_balance=leg.risk.initial_balance,
            risk_mode=leg.risk.risk_mode,
            risk_value=leg.risk.risk_value * scale_factor,
            max_trades_per_day=leg.risk.max_trades_per_day,
            commission_per_trade=leg.risk.commission_per_trade,
            slippage_pips=leg.risk.slippage_pips,
            spread_pips=leg.risk.spread_pips,
            pip_size=leg.risk.pip_size,
            max_position_size=leg.risk.max_position_size,
            daily_loss_limit_pct=leg.risk.daily_loss_limit_pct,
        )
        bt = run_backtest(leg.df, leg.strategy, adjusted_risk)
        all_final_trades.extend(bt.trades)
        leg_results.append(LegResult(
            name=leg.name,
            nominal_weight=nominal_weight,
            avg_correlation_with_others=avg_corr[leg.name],
            final_weight=weight,
            risk_value_scale=scale_factor,
            trade_count=len(bt.trades),
            net_profit=bt.statistics.net_profit,
            statistics=bt.statistics.to_dict(),
        ))

    combined_equity = _rebuild_equity_curve(all_final_trades, cfg.initial_balance)
    combined_stats = compute_statistics(all_final_trades, combined_equity, initial_balance=cfg.initial_balance) \
        if len(combined_equity) else compute_statistics([], pd.DataFrame({"timestamp": [], "equity": []}), cfg.initial_balance)

    # Diversification ratio: weighted-average of each leg's OWN volatility,
    # vs. the realized volatility of the combined portfolio equity curve.
    diversification_ratio = None
    try:
        combined_returns = _daily_returns(combined_equity)
        portfolio_vol = float(combined_returns.std())
        indiv_vols = {name: float(returns_by_leg[name].std()) if len(returns_by_leg[name]) > 1 else 0.0 for name in names}
        weighted_avg_vol = sum(final_weights[name] * indiv_vols[name] for name in names) / max(sum(final_weights.values()), 1e-9)
        if portfolio_vol and math.isfinite(portfolio_vol) and portfolio_vol > 0:
            diversification_ratio = float(weighted_avg_vol / portfolio_vol)
    except Exception:
        diversification_ratio = None

    return PortfolioResult(
        legs=leg_results,
        correlation_matrix=correlation_matrix,
        combined_trades=all_final_trades,
        combined_equity_curve=combined_equity,
        combined_statistics=combined_stats,
        diversification_ratio=diversification_ratio,
        warnings=warnings,
        **_combined_mc_fields(all_final_trades, cfg),
    )


def _combined_mc_fields(all_final_trades: list[Trade], cfg: PortfolioConfig) -> dict:
    """Runs Monte Carlo on the portfolio's combined trade sequence against
    ONE shared prop account, when both prop_rules and mc_config were
    supplied -- see PortfolioConfig.prop_rules's docstring for why this
    is the number that actually answers "does diversifying across
    several strategies raise my probability of passing". Returns an
    empty dict (mc_result/single_run_summary both stay None) if either
    config piece is missing or there are no combined trades to score."""
    if cfg.prop_rules is None or cfg.mc_config is None or not all_final_trades:
        return {}
    pnls = [t.pnl for t in all_final_trades]
    dates = [t.entry_time for t in all_final_trades]
    single_run = simulate_account(pnls, dates, cfg.prop_rules)
    mc = run_monte_carlo(all_final_trades, cfg.prop_rules, cfg.mc_config)
    return {"mc_result": mc, "single_run_summary": summarize_single_run(single_run)}
