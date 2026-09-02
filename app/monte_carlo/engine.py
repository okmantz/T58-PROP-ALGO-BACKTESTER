"""
Monte Carlo Prop Simulation.

The primary feature of the application. Instead of asking only "was the
strategy profitable historically?", this engine asks: "if I ran this
strategy through thousands of simulated prop accounts with different
possible trade sequences, how often would I pass and actually get paid?"

Each simulation resamples the historical trade P&L sequence (shuffle /
bootstrap / block-bootstrap for loss-streak stress, with optional slippage
stress) and re-runs it through the exact same prop-rule account simulator
used for the single historical run (app.prop.simulator.simulate_account),
so results are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account


@dataclass
class MonteCarloConfig:
    n_simulations: int = 10_000
    method: str = "bootstrap"        # "shuffle" | "bootstrap" | "block_bootstrap"
    block_size: int = 5              # used when method == "block_bootstrap"
    slippage_stress_pct: float = 0.0  # extra % cost applied to every trade
    random_seed: int | None = 42


@dataclass
class MonteCarloResult:
    n_simulations: int
    evaluation_pass_probability: float
    first_payout_probability: float
    failure_before_payout_probability: float
    multiple_payout_probability: float

    median_days_to_pass: float | None
    median_days_to_first_payout: float | None
    average_days_to_first_payout: float | None

    median_return_pct: float
    mean_return_pct: float
    expected_payout: float
    median_payout: float
    total_simulated_withdrawals: float

    median_drawdown_pct: float
    p95_drawdown_pct: float
    worst_drawdown_pct: float
    risk_of_ruin_pct: float
    median_max_losing_streak: float
    worst_max_losing_streak: int

    return_percentiles: dict = field(default_factory=dict)      # {5,25,50,75,95: pct}
    drawdown_percentiles: dict = field(default_factory=dict)
    days_to_payout_distribution: list = field(default_factory=list)
    # Full per-simulation distributions, kept for charting (e.g. a return
    # histogram). Not shown in the flat summary tables, only used by the
    # HTML report's charts.
    return_distribution: list = field(default_factory=list)
    drawdown_distribution: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _max_losing_streak(pnls: np.ndarray) -> int:
    best = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)
    return best


def _resample_pnls(rng: np.random.Generator, pnls: np.ndarray, cfg: MonteCarloConfig) -> np.ndarray:
    n = len(pnls)
    if cfg.method == "shuffle":
        return rng.permutation(pnls)
    if cfg.method == "block_bootstrap":
        blocks = max(1, n // cfg.block_size)
        out = []
        while len(out) < n:
            start = rng.integers(0, n)
            block = [pnls[(start + j) % n] for j in range(cfg.block_size)]
            out.extend(block)
        return np.array(out[:n])
    # default: iid bootstrap with replacement
    idx = rng.integers(0, n, size=n)
    return pnls[idx]


def _apply_slippage_stress(pnls: np.ndarray, stress_pct: float) -> np.ndarray:
    if not stress_pct:
        return pnls
    factor = stress_pct / 100.0
    stressed = pnls.copy()
    stressed[stressed > 0] *= (1 - factor)
    stressed[stressed <= 0] *= (1 + factor)
    return stressed


def eval_pass_probability_for_trades(
    trades: list[Trade],
    rules: PropRules,
    mc_cfg: MonteCarloConfig | None = None,
) -> float:
    """
    Convenience wrapper around run_monte_carlo() that returns just the
    single number nearly every fold-level / candidate-level scoring path
    in the app actually wants: the probability of reaching the prop
    firm's profit target BEFORE hitting the daily-loss limit, the
    max-drawdown limit, or the consistency rule -- i.e.
    MonteCarloResult.evaluation_pass_probability.

    This is the shared primitive behind making "probability of passing"
    (rather than raw backtest profit, win rate, or R:R) the one thing
    every test in the app -- Iterative Refinement, Walk-Forward
    Optimization, the walk-forward-aware GA, CPCV, the Evolution Lab, and
    Quick Optimize/Full Pipeline -- actually optimizes and validates
    against, including at the per-fold / per-path level where those
    modules previously fell back to a plain backtest-stats metric like
    profit_factor because no Monte Carlo had been run yet for that slice
    of data.

    Uses a smaller default simulation count than a final-report Monte
    Carlo run (this is called once per fold/path/generation, often many
    times per search) -- callers that care about that tradeoff should
    pass their own mc_cfg. Returns 0.0 (not an exception) for an
    empty/too-small trade list, since "this slice produced nothing worth
    passing" is itself a valid, low, fold score rather than a hard
    failure.
    """
    if not trades:
        return 0.0
    cfg = mc_cfg or MonteCarloConfig(n_simulations=500)
    try:
        result = run_monte_carlo(trades, rules, cfg)
    except ValueError:
        return 0.0
    return result.evaluation_pass_probability


def run_monte_carlo(
    trades: list[Trade],
    rules: PropRules,
    cfg: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    cfg = cfg or MonteCarloConfig()
    if not trades:
        raise ValueError("Cannot run Monte Carlo simulation with zero trades.")

    rng = np.random.default_rng(cfg.random_seed)
    base_pnls = np.array([t.pnl for t in trades], dtype=float)
    base_dates = [pd.Timestamp(t.entry_time).normalize() for t in trades]
    # Every simulation below reassigns the SAME fixed calendar dates
    # (base_dates never changes) to a resampled sequence of P&L values --
    # only sim_pnls' order/values differ per simulation. That means the
    # date-to-trading-day bookkeeping simulate_account would otherwise
    # rebuild from scratch (via per-trade pandas Timestamp parsing and
    # dict lookups) on every single one of cfg.n_simulations calls is
    # actually identical every time, so it's computed once here instead.
    # See app.prop.simulator.DayStructure's docstring for the full
    # reasoning; this is a pure performance change with no effect on any
    # output value.
    day_structure = precompute_day_structure(base_dates)

    passed_flags, first_payout_flags, failed_before_payout_flags, multiple_payout_flags = [], [], [], []
    days_to_pass_list, days_to_first_payout_list = [], []
    return_pcts, payout_amounts, drawdown_pcts, losing_streaks = [], [], [], []
    total_withdrawals = 0.0

    for _ in range(cfg.n_simulations):
        sim_pnls = _resample_pnls(rng, base_pnls, cfg)
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)

        result = simulate_account(sim_pnls, base_dates, rules, _day_structure=day_structure)

        passed_flags.append(result.passed_evaluation)
        first_payout_flags.append(result.reached_first_payout)
        failed_before_payout_flags.append(result.failed and not result.reached_first_payout)
        multiple_payout_flags.append(len(result.payouts) > 1)

        if result.days_to_pass is not None:
            days_to_pass_list.append(result.days_to_pass)
        if result.first_payout_day_index is not None:
            days_to_first_payout_list.append(result.first_payout_day_index)

        return_pcts.append((result.final_balance - rules.account_size) / rules.account_size * 100.0)
        payout_amounts.append(result.total_payout_amount)
        total_withdrawals += result.total_payout_amount
        drawdown_pcts.append(result.max_drawdown_pct_reached)
        losing_streaks.append(_max_losing_streak(sim_pnls))

    passed_arr = np.array(passed_flags)
    first_payout_arr = np.array(first_payout_flags)
    failed_before_payout_arr = np.array(failed_before_payout_flags)
    multiple_payout_arr = np.array(multiple_payout_flags)
    return_arr = np.array(return_pcts)
    payout_arr = np.array(payout_amounts)
    dd_arr = np.array(drawdown_pcts)
    streak_arr = np.array(losing_streaks)

    ruin_arr = dd_arr >= rules.max_drawdown_pct  # account hit its max-drawdown floor at least once

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else 0.0

    result = MonteCarloResult(
        n_simulations=cfg.n_simulations,
        evaluation_pass_probability=float(passed_arr.mean() * 100),
        first_payout_probability=float(first_payout_arr.mean() * 100),
        failure_before_payout_probability=float(failed_before_payout_arr.mean() * 100),
        multiple_payout_probability=float(multiple_payout_arr.mean() * 100),
        median_days_to_pass=float(np.median(days_to_pass_list)) if days_to_pass_list else None,
        median_days_to_first_payout=float(np.median(days_to_first_payout_list)) if days_to_first_payout_list else None,
        average_days_to_first_payout=float(np.mean(days_to_first_payout_list)) if days_to_first_payout_list else None,
        median_return_pct=pct(return_arr, 50),
        mean_return_pct=float(return_arr.mean()) if len(return_arr) else 0.0,
        expected_payout=float(payout_arr.mean()) if len(payout_arr) else 0.0,
        median_payout=pct(payout_arr, 50),
        total_simulated_withdrawals=float(total_withdrawals),
        median_drawdown_pct=pct(dd_arr, 50),
        p95_drawdown_pct=pct(dd_arr, 95),
        worst_drawdown_pct=float(dd_arr.max()) if len(dd_arr) else 0.0,
        risk_of_ruin_pct=float(ruin_arr.mean() * 100),
        median_max_losing_streak=float(np.median(streak_arr)) if len(streak_arr) else 0.0,
        worst_max_losing_streak=int(streak_arr.max()) if len(streak_arr) else 0,
        return_percentiles={q: pct(return_arr, q) for q in (5, 25, 50, 75, 95)},
        drawdown_percentiles={q: pct(dd_arr, q) for q in (5, 25, 50, 75, 95)},
        days_to_payout_distribution=days_to_first_payout_list,
        return_distribution=return_arr.tolist(),
        drawdown_distribution=dd_arr.tolist(),
    )
    return result
