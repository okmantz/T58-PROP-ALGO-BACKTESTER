"""
Prop-Firm Survival Engine.

Every other number this app produces (net profit, profit factor, Sharpe,
even the plain Monte Carlo evaluation-pass/first-payout probabilities)
ultimately answers one question: "how profitable is this strategy?" That
is NOT the question a prop-firm trader actually needs answered. The real
question is:

    "Can this strategy survive a prop firm's rules long enough to get
     paid -- more than once -- after accounting for what it actually
     costs to keep trying?"

Those are different questions with different answers. A strategy with a
95% evaluation-pass probability that blows the funded account on its
first bad week is worse, in every way that matters to a trader's bank
account, than an 82%-pass strategy that reliably strings together three
or four payouts before it ever fails. Nothing else in this app currently
surfaces that difference. This module does.

It is deliberately built as a THIN layer on top of two modules that
already do the hard, validated work:

    app.prop.simulator     -- the single-account rules engine
                               (simulate_account / PropRules). Every
                               number in this module still comes from
                               that same function; this module never
                               reimplements prop-firm rule logic.
    app.monte_carlo.engine -- the resampling machinery (_resample_pnls /
                               _apply_slippage_stress / precompute_day_
                               structure / _max_losing_streak). Reused
                               here (not reimplemented) so a strategy's
                               survival numbers and its existing Monte
                               Carlo numbers are always computed the same
                               way, with the same statistical assumptions.

What's actually NEW here, that neither of those modules answers:

  Evaluation Survival   -- not just "did it pass" but WHY it typically
                           fails when it fails (daily-loss vs. max-
                           drawdown), how long a pass usually takes, how
                           bad the worst losing streak gets, and how long
                           the account takes to recover from its worst
                           drawdown episode.

  Funded Survival       -- probability of reaching not just the FIRST
                           payout but the 2nd and 3rd (most strategies'
                           odds fall off a cliff after the first -- this
                           is the number that reveals that), plus the
                           expected total payout actually collected
                           before the account eventually fails.

  Reset/Retry Economics -- the piece nothing in this app answers at all:
                           given what an evaluation attempt and a reset
                           actually cost, and given a trader's real cut
                           of each payout, what is the EXPECTED NET
                           PROFIT after however many attempts it
                           realistically takes -- including the
                           attempts that fail outright and cost money
                           with nothing to show for it? This is modeled
                           as a chain of independent attempts (see
                           `simulate_reset_chain`), each one an
                           independent resampled draw of the SAME
                           historical edge, continuing until the account
                           survives to the end of the available trade
                           history (treated as "still alive, stop
                           counting") or the trader runs out of resets
                           they're willing/able to pay for.

The `PropSurvivalScore` this module produces is the number this app
should actually be optimizing for when the goal is "give me a strategy I
can trust with my own money at a real prop firm" -- not profit factor,
not Sharpe, not even the plain evaluation-pass probability alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.engine import _apply_slippage_stress, _max_losing_streak, _resample_pnls
from app.prop.simulator import (
    DayStructure, PropRules, precompute_day_structure, simulate_account,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ResetEconomics:
    """
    What it actually costs a trader to keep trying, and what they actually
    keep when they get paid. None of this exists anywhere else in the
    app -- PropRules models the FIRM's rules (targets, drawdowns, payout
    thresholds); this models the TRADER's out-of-pocket economics around
    those rules.
    """
    evaluation_fee: float = 0.0        # up-front cost of the first evaluation attempt
    reset_fee: float | None = None     # cost of every attempt after the first; None = same as evaluation_fee
    profit_split_pct: float = 80.0     # trader's cut of every payout (the rest is the firm's)
    max_attempts: int = 3              # total lifetime attempts this analysis is willing to fund

    def __post_init__(self):
        self.evaluation_fee = max(float(self.evaluation_fee), 0.0)
        if self.reset_fee is not None:
            self.reset_fee = max(float(self.reset_fee), 0.0)
        self.profit_split_pct = min(max(float(self.profit_split_pct), 0.0), 100.0)
        self.max_attempts = max(int(self.max_attempts), 1)

    def resolved_reset_fee(self) -> float:
        return self.evaluation_fee if self.reset_fee is None else self.reset_fee


@dataclass
class PropSurvivalConfig:
    n_simulations: int = 5_000
    method: str = "bootstrap"          # "shuffle" | "bootstrap" | "block_bootstrap" -- see app.monte_carlo.engine
    block_size: int = 5
    slippage_stress_pct: float = 0.0
    random_seed: int | None = 42
    reset_economics: ResetEconomics = field(default_factory=ResetEconomics)
    # The reset-chain analysis (simulate_reset_chain) runs up to
    # reset_economics.max_attempts full simulate_account() calls per
    # "life" it draws, so it's capped independently of n_simulations
    # (which only ever runs ONE simulate_account() call per iteration) to
    # keep total wall-clock cost predictable. Raise it for a smoother
    # reset-economics distribution; lower it if this is the slow part of
    # a larger pipeline (e.g. Strategy Lab scoring dozens of finalists).
    life_simulations: int = 2_000


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSurvivalStats:
    probability_reach_profit_target: float   # balance crossed the target at some point, ignoring min-days/consistency
    probability_pass_evaluation: float       # FULL pass: target reached AND min trading days AND consistency rule
    probability_hit_daily_loss: float        # failed, specifically via the daily-loss-limit rule
    probability_hit_max_drawdown: float      # failed, specifically via the max-drawdown rule
    consistency_conditional_pass_rate: float | None  # of sims that reached the raw target, what fraction still passed once the consistency rule was applied (None if the target was never reached in any sim)
    median_days_required: float | None
    p90_days_required: float | None
    worst_losing_streak_median: float
    worst_losing_streak_p95: float
    median_recovery_days: float | None       # days from the sim's single worst drawdown trough back to its prior peak (None-excluded sims never recovered within the window -- see never_recovered_pct)
    never_recovered_pct: float                # % of sims whose worst drawdown was never reclaimed before the trade sequence ran out
    average_daily_pnl: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FundedSurvivalStats:
    probability_first_payout: float
    probability_second_payout: float
    probability_third_payout: float
    expected_payout_before_failure: float    # mean total $ withdrawn across the account's life (single attempt), whether it ultimately fails or the data simply runs out
    median_payout_amount: float              # median size of an individual payout event (not the total)
    median_days_to_first_payout: float | None
    median_single_attempt_lifetime_days: float  # median trading days survived in a single (no-reset) attempt

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ResetEconomicsResult:
    expected_net_profit_after_resets: float
    median_net_profit_after_resets: float
    probability_net_positive_after_resets: float
    probability_exhausts_resets_without_profit: float
    expected_attempts_used: float
    expected_fees_paid: float
    expected_gross_payouts: float
    expected_account_lifetime_days: float    # total trading days across every attempt in the chain, until it survives to the end of the data or runs out of resets
    max_attempts_configured: int
    profit_split_pct: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PropSurvivalResult:
    evaluation: EvaluationSurvivalStats
    funded: FundedSurvivalStats
    reset_economics: ResetEconomicsResult
    prop_survival_score: float               # 0-100 composite -- see _compute_survival_score
    score_breakdown: dict                    # {label: contribution} -- the "Explain" behind the score
    n_simulations: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evaluation": self.evaluation.to_dict(),
            "funded": self.funded.to_dict(),
            "reset_economics": self.reset_economics.to_dict(),
            "prop_survival_score": self.prop_survival_score,
            "score_breakdown": self.score_breakdown,
            "n_simulations": self.n_simulations,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _day_end_balances(pnls: np.ndarray, account_size: float, day_structure: DayStructure) -> np.ndarray:
    """End-of-day balance for every trading day in the sequence -- the one
    thing simulate_account() computes internally but doesn't return, and
    that both the raw-target-reached check and the recovery-time
    calculation below need. O(n_trades), same cost class as
    simulate_account() itself."""
    balance = account_size
    day_end = np.empty(day_structure.n_days, dtype=float)
    day_idx = day_structure.day_index_per_trade
    is_last = day_structure.is_last_of_day
    for i, pnl in enumerate(pnls):
        balance += pnl
        if is_last[i]:
            day_end[day_idx[i]] = balance
    return day_end


def _worst_drawdown_recovery_days(day_end_balances: np.ndarray) -> int | None:
    """Days from the SINGLE worst drawdown episode's trough back to
    reclaiming the peak that preceded it. Returns None if that episode is
    never reclaimed before the sequence ends (right-censored -- counted
    separately as `never_recovered_pct` rather than silently dropped)."""
    if len(day_end_balances) == 0:
        return None
    peak = day_end_balances[0]
    worst_dd = 0.0
    trough_idx = None
    trough_peak_level = peak
    for i, bal in enumerate(day_end_balances):
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > worst_dd:
            worst_dd = dd
            trough_idx = i
            trough_peak_level = peak
    if trough_idx is None or worst_dd <= 0:
        return 0
    for j in range(trough_idx + 1, len(day_end_balances)):
        if day_end_balances[j] >= trough_peak_level:
            return j - trough_idx
    return None


def _pct(flags: list[bool]) -> float:
    return float(np.mean(flags) * 100.0) if flags else 0.0


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


# ---------------------------------------------------------------------------
# Single-attempt survival analysis
# ---------------------------------------------------------------------------

def _run_single_attempt_survival(
    base_pnls: np.ndarray,
    base_dates: list,
    rules: PropRules,
    cfg: PropSurvivalConfig,
    day_structure: DayStructure,
    rng: np.random.Generator,
) -> tuple[EvaluationSurvivalStats, FundedSurvivalStats]:
    n = cfg.n_simulations
    target_balance = rules.account_size * (1 + rules.evaluation_profit_target_pct / 100.0)

    reached_target_flags: list[bool] = []
    passed_eval_flags: list[bool] = []
    hit_daily_loss_flags: list[bool] = []
    hit_max_dd_flags: list[bool] = []
    days_required: list[float] = []
    worst_streaks: list[float] = []
    recovery_days: list[float] = []
    never_recovered_flags: list[bool] = []
    daily_pnl_means: list[float] = []
    payout_counts: list[int] = []
    total_payout_amounts: list[float] = []
    individual_payout_amounts: list[float] = []
    lifetime_days: list[float] = []
    days_to_first_payout: list[float] = []

    for _ in range(n):
        sim_pnls = _resample_pnls(rng, base_pnls, _ResampleCfg(cfg.method, cfg.block_size))
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)

        result = simulate_account(sim_pnls, base_dates, rules, _day_structure=day_structure)
        day_end = _day_end_balances(sim_pnls, rules.account_size, day_structure)

        reached_target_flags.append(bool(np.any(day_end >= target_balance)))
        passed_eval_flags.append(result.passed_evaluation)
        hit_daily_loss_flags.append(bool(result.failed and result.failure_reason == "daily_loss_limit"))
        hit_max_dd_flags.append(bool(
            result.failed and result.failure_reason is not None
            and result.failure_reason.startswith("max_drawdown")
        ))
        if result.days_to_pass is not None:
            days_required.append(result.days_to_pass)
        worst_streaks.append(_max_losing_streak(sim_pnls))
        lifetime_days.append(result.trading_days_count)
        if day_structure.n_days > 0:
            daily_pnl_means.append(float(np.sum(sim_pnls)) / day_structure.n_days)

        recovery = _worst_drawdown_recovery_days(day_end)
        if recovery is None:
            never_recovered_flags.append(True)
        else:
            never_recovered_flags.append(False)
            recovery_days.append(recovery)

        payout_counts.append(len(result.payouts))
        total_payout_amounts.append(result.total_payout_amount)
        individual_payout_amounts.extend(p.amount for p in result.payouts)
        if result.first_payout_day_index is not None:
            days_to_first_payout.append(result.first_payout_day_index)

    payout_arr = np.array(payout_counts)
    reached_target_pct = _pct(reached_target_flags)
    passed_eval_pct = _pct(passed_eval_flags)
    consistency_conditional = (
        (passed_eval_pct / reached_target_pct * 100.0) if reached_target_pct > 0 else None
    )
    if consistency_conditional is not None:
        consistency_conditional = min(consistency_conditional, 100.0)

    evaluation = EvaluationSurvivalStats(
        probability_reach_profit_target=reached_target_pct,
        probability_pass_evaluation=passed_eval_pct,
        probability_hit_daily_loss=_pct(hit_daily_loss_flags),
        probability_hit_max_drawdown=_pct(hit_max_dd_flags),
        consistency_conditional_pass_rate=consistency_conditional,
        median_days_required=_median(days_required),
        p90_days_required=(float(np.percentile(days_required, 90)) if days_required else None),
        worst_losing_streak_median=float(np.median(worst_streaks)) if worst_streaks else 0.0,
        worst_losing_streak_p95=float(np.percentile(worst_streaks, 95)) if worst_streaks else 0.0,
        median_recovery_days=_median(recovery_days),
        never_recovered_pct=_pct(never_recovered_flags),
        average_daily_pnl=float(np.mean(daily_pnl_means)) if daily_pnl_means else 0.0,
    )

    funded = FundedSurvivalStats(
        probability_first_payout=_pct((payout_arr >= 1).tolist()),
        probability_second_payout=_pct((payout_arr >= 2).tolist()),
        probability_third_payout=_pct((payout_arr >= 3).tolist()),
        expected_payout_before_failure=float(np.mean(total_payout_amounts)) if total_payout_amounts else 0.0,
        median_payout_amount=_median(individual_payout_amounts) or 0.0,
        median_days_to_first_payout=_median(days_to_first_payout),
        median_single_attempt_lifetime_days=float(np.median(lifetime_days)) if lifetime_days else 0.0,
    )
    return evaluation, funded


# Small stand-in so this module never has to import MonteCarloConfig just
# to satisfy _resample_pnls' type hint -- it only ever reads .method and
# .block_size off whatever's passed in, so any object with those two
# attributes works identically.
@dataclass
class _ResampleCfg:
    method: str
    block_size: int


# ---------------------------------------------------------------------------
# Reset/retry economics -- the chain of independent attempts
# ---------------------------------------------------------------------------

def simulate_reset_chain(
    base_pnls: np.ndarray,
    base_dates: list,
    rules: PropRules,
    day_structure: DayStructure,
    reset: ResetEconomics,
    resample_cfg: "_ResampleCfg",
    slippage_stress_pct: float,
    rng: np.random.Generator,
) -> dict:
    """
    Draws ONE possible "life" of a trader repeatedly buying (or resetting)
    an evaluation with this exact strategy, up to `reset.max_attempts`
    times. Each attempt is an independent resampled draw of the SAME
    historical trade sequence -- consistent with every other resampling
    in this app (see app.monte_carlo.engine): it answers "if the future
    looks statistically like the past, what happens", not "what will
    literally happen next."

    An attempt ends the chain (no further resets needed) the moment it
    survives all the way to the end of the available historical trade
    count without failing -- treated as "the account is still alive and
    trading," not as a success/failure event requiring another purchase.
    An attempt that fails (busts the evaluation OR the funded account)
    consumes that attempt's fee and, if resets remain, starts a fresh
    attempt with a newly resampled sequence.

    Every payout collected along the way is credited at `reset.
    profit_split_pct` of its gross amount (the trader's actual take-home,
    not the firm's payout number) and every attempt's fee is a real cost
    whether or not that attempt ever got funded.
    """
    fees_paid = 0.0
    gross_payouts = 0.0
    net_payouts = 0.0
    total_days = 0
    payout_count = 0
    attempts_used = 0
    survived_to_end = False

    for attempt in range(1, reset.max_attempts + 1):
        attempts_used = attempt
        fee = reset.evaluation_fee if attempt == 1 else reset.resolved_reset_fee()
        fees_paid += fee

        sim_pnls = _resample_pnls(rng, base_pnls, resample_cfg)
        sim_pnls = _apply_slippage_stress(sim_pnls, slippage_stress_pct)
        result = simulate_account(sim_pnls, base_dates, rules, _day_structure=day_structure)

        total_days += result.trading_days_count
        payout_count += len(result.payouts)
        for p in result.payouts:
            gross_payouts += p.amount
            net_payouts += p.amount * (reset.profit_split_pct / 100.0)

        if not result.failed:
            # Reached the end of the resampled trade sequence without
            # busting -- there's no natural "end" event here (the strategy
            # simply keeps trading), so this chain stops WITHOUT needing
            # another reset. This is the "still alive" outcome, distinct
            # from running out of resets below.
            survived_to_end = True
            break
        # Otherwise this attempt busted the account (eval or funded stage)
        # -- loop continues into a fresh attempt if any remain.

    net_profit = net_payouts - fees_paid
    return {
        "attempts_used": attempts_used,
        "fees_paid": fees_paid,
        "gross_payouts": gross_payouts,
        "net_payouts": net_payouts,
        "net_profit": net_profit,
        "lifetime_days": total_days,
        "payout_count": payout_count,
        "exhausted_without_survival": (not survived_to_end),
    }


def _run_reset_economics(
    base_pnls: np.ndarray,
    base_dates: list,
    rules: PropRules,
    cfg: PropSurvivalConfig,
    day_structure: DayStructure,
    rng: np.random.Generator,
) -> ResetEconomicsResult:
    reset = cfg.reset_economics
    resample_cfg = _ResampleCfg(cfg.method, cfg.block_size)

    net_profits: list[float] = []
    attempts_list: list[int] = []
    fees_list: list[float] = []
    gross_list: list[float] = []
    lifetime_list: list[float] = []
    exhausted_without_profit: list[bool] = []

    for _ in range(cfg.life_simulations):
        life = simulate_reset_chain(
            base_pnls, base_dates, rules, day_structure, reset, resample_cfg,
            cfg.slippage_stress_pct, rng,
        )
        net_profits.append(life["net_profit"])
        attempts_list.append(life["attempts_used"])
        fees_list.append(life["fees_paid"])
        gross_list.append(life["gross_payouts"])
        lifetime_list.append(life["lifetime_days"])
        exhausted_without_profit.append(life["exhausted_without_survival"] and life["net_profit"] <= 0)

    return ResetEconomicsResult(
        expected_net_profit_after_resets=float(np.mean(net_profits)) if net_profits else 0.0,
        median_net_profit_after_resets=float(np.median(net_profits)) if net_profits else 0.0,
        probability_net_positive_after_resets=_pct([p > 0 for p in net_profits]),
        probability_exhausts_resets_without_profit=_pct(exhausted_without_profit),
        expected_attempts_used=float(np.mean(attempts_list)) if attempts_list else 0.0,
        expected_fees_paid=float(np.mean(fees_list)) if fees_list else 0.0,
        expected_gross_payouts=float(np.mean(gross_list)) if gross_list else 0.0,
        expected_account_lifetime_days=float(np.mean(lifetime_list)) if lifetime_list else 0.0,
        max_attempts_configured=reset.max_attempts,
        profit_split_pct=reset.profit_split_pct,
    )


# ---------------------------------------------------------------------------
# Prop Survival Score -- the composite this module exists to produce
# ---------------------------------------------------------------------------

def _compute_survival_score(
    evaluation: EvaluationSurvivalStats,
    funded: FundedSurvivalStats,
    reset_econ: ResetEconomicsResult,
) -> tuple[float, dict]:
    """
    A single 0-100 number that answers "can this strategy survive long
    enough to get paid, more than once, after what it actually costs to
    keep trying" -- deliberately NOT the same thing as any pure
    profitability metric. Weighted so that:

      - Passing the evaluation at all is necessary but not sufficient
        (0.20 weight) -- a strategy that only ever clears the eval and
        never gets paid still scores low.
      - Getting repeated payouts (1st/2nd/3rd, tapering weights) matters
        more than getting just one, since a strategy's odds typically
        fall off sharply after the first payout and that fall-off is
        exactly what a pure eval-pass or first-payout number hides.
      - Net-positive economics AFTER resets (0.25 weight) is the single
        largest component, since this is the number a trader's actual
        bank account experiences -- a strategy that passes evaluations
        constantly but never nets positive after paying for the
        privilege of resetting is not, in any real sense, a survivable
        strategy.
      - The two funded-account failure modes (daily-loss / max-drawdown)
        are subtracted, since a strategy that fails those rules often is
        actively dangerous to trade even when its average-case numbers
        look fine.

    This is one reasonable weighting, not a law of nature -- treat the
    breakdown dict as the actual audit trail, and the single score as a
    convenient sort key on top of it.
    """
    contributions = {
        "evaluation_pass": evaluation.probability_pass_evaluation * 0.20,
        "first_payout": funded.probability_first_payout * 0.15,
        "second_payout": funded.probability_second_payout * 0.10,
        "third_payout": funded.probability_third_payout * 0.05,
        "net_positive_after_resets": reset_econ.probability_net_positive_after_resets * 0.25,
        "daily_loss_penalty": -evaluation.probability_hit_daily_loss * 0.125,
        "max_drawdown_penalty": -evaluation.probability_hit_max_drawdown * 0.125,
    }
    raw_score = sum(contributions.values())
    score = min(max(raw_score, 0.0), 100.0)
    return score, contributions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_prop_survival_analysis(
    trades: list[Trade],
    rules: PropRules,
    cfg: PropSurvivalConfig | None = None,
) -> PropSurvivalResult:
    """
    The one function everything else in this module exists to support.
    Reuses the exact same historical trade P&L sequence + resampling
    machinery as app.monte_carlo.engine.run_monte_carlo -- this is meant
    to be run ALONGSIDE that function (or instead of it, for a report
    that wants the fuller survival picture), never as a replacement for
    app.prop.simulator's single deterministic historical run.
    """
    cfg = cfg or PropSurvivalConfig()
    if not trades:
        raise ValueError("Cannot run a prop survival analysis with zero trades.")

    rng = np.random.default_rng(cfg.random_seed)
    base_pnls = np.array([t.pnl for t in trades], dtype=float)
    base_dates = [pd.Timestamp(t.entry_time).normalize() for t in trades]
    day_structure = precompute_day_structure(base_dates)

    evaluation, funded = _run_single_attempt_survival(base_pnls, base_dates, rules, cfg, day_structure, rng)
    reset_econ = _run_reset_economics(base_pnls, base_dates, rules, cfg, day_structure, rng)
    score, breakdown = _compute_survival_score(evaluation, funded, reset_econ)

    notes: list[str] = []
    if evaluation.probability_pass_evaluation < 40:
        notes.append(
            f"Evaluation-pass probability is only {evaluation.probability_pass_evaluation:.1f}% -- "
            "every downstream funded-account and reset-economics number below is conditioned on an "
            "event that itself rarely happens; treat them as illustrative, not a promise."
        )
    if reset_econ.probability_net_positive_after_resets < 50:
        notes.append(
            f"Net profit after resets is positive in only "
            f"{reset_econ.probability_net_positive_after_resets:.1f}% of simulated lifetimes at the "
            f"configured {reset_econ.profit_split_pct:.0f}% profit split and "
            f"{reset_econ.max_attempts_configured} max attempt(s) -- this strategy is, on the "
            "numbers actually costed here, not clearly worth paying to trade at a prop firm."
        )
    if evaluation.never_recovered_pct > 25:
        notes.append(
            f"In {evaluation.never_recovered_pct:.1f}% of simulations, the account's worst drawdown "
            "was never reclaimed before the available trade history ran out -- the recovery-time "
            "numbers above are understated versus a strategy that always recovers."
        )

    return PropSurvivalResult(
        evaluation=evaluation,
        funded=funded,
        reset_economics=reset_econ,
        prop_survival_score=score,
        score_breakdown=breakdown,
        n_simulations=cfg.n_simulations,
        notes=notes,
    )
