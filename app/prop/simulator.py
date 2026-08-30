"""
Prop-firm evaluation + funded-account simulator.

Given a chronological sequence of trade P&Ls (plus their dates), walks the
account forward under a configurable set of prop-firm rules and determines:
  - whether/when the evaluation is passed
  - whether/when the account is failed (and which rule caused it)
  - whether/when the funded account reaches its first payout, and all
    subsequent payouts

This module is independent of *how* the trade sequence was produced, so the
exact same simulate_account() function is used both for the single
historical trade sequence (section 4 of the spec) and for every resampled
sequence inside the Monte Carlo engine (section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type

import pandas as pd


@dataclass
class PropRules:
    account_size: float = 100_000.0
    evaluation_profit_target_pct: float = 8.0     # % gain required to pass evaluation
    daily_loss_limit_pct: float = 5.0              # max loss in a single day (% of account_size)
    max_drawdown_pct: float = 10.0                 # overall max drawdown allowed
    drawdown_type: str = "trailing"                 # "trailing" | "static"
    drawdown_check_mode: str = "intrabar"           # "intrabar" | "eod" -- see note below
    consistency_rule_pct: float | None = 30.0       # best single day's profit <= this % of total profit
    min_trading_days: int = 5
    payout_threshold_pct: float = 0.0               # extra profit % (above account size) required before 1st payout eligibility, funded stage
    payout_cap_pct: float | None = None             # max % of available profit withdrawable per payout (None = 100%)
    payout_frequency_days: int = 14                 # min days between payouts
    required_buffer_pct: float = 0.0                # profit buffer that must be maintained above account_size before payout
    max_position_size: float | None = None          # informational cap on units (enforced in RiskConfig)

    # drawdown_check_mode controls WHEN the daily-loss-limit and max-drawdown
    # failure checks are evaluated:
    #   "intrabar" (default, conservative): checked after every single trade
    #   as it closes, matching a firm that monitors floating equity in real
    #   time and can auto-liquidate mid-day the instant a floor is crossed.
    #   "eod": checked only once per calendar day, using that day's final
    #   cumulative balance after all of that day's trades -- matching firms
    #   (many real futures prop firms, including both evaluated in this
    #   codebase's research notes) that explicitly state an "EOD" drawdown
    #   type: you can be deep underwater intraday and still be fine as long
    #   as you close the day above the floor. Using "intrabar" against a
    #   firm that is actually "eod" understates your true pass probability;
    #   using "eod" against a firm that is actually real-time overstates it.
    #   Match this to what the specific firm's rules document actually say.


@dataclass
class PayoutEvent:
    day_index: int
    date: str
    amount: float
    balance_after: float


@dataclass
class AccountSimResult:
    passed_evaluation: bool
    failed: bool
    failure_reason: str | None
    failure_day_index: int | None
    days_to_pass: int | None
    first_payout_day_index: int | None
    first_payout_amount: float | None
    payouts: list[PayoutEvent] = field(default_factory=list)
    final_balance: float = 0.0
    max_drawdown_pct_reached: float = 0.0
    trading_days_count: int = 0

    @property
    def reached_first_payout(self) -> bool:
        return self.first_payout_day_index is not None

    @property
    def total_payout_amount(self) -> float:
        return sum(p.amount for p in self.payouts)


@dataclass
class DayStructure:
    """The part of simulate_account's bookkeeping that depends ONLY on
    `trade_dates`, never on the P&L values themselves: which calendar day
    each trade in the sequence falls on, and which trades are the last of
    their day. The Monte Carlo engine resamples/shuffles P&L VALUES across
    thousands of simulations while reusing the exact same (fixed,
    historical) trade dates every time -- see app.monte_carlo.engine's
    run_monte_carlo -- so this structure is identical across every one of
    those simulations and only needs to be computed once, not
    re-derived (via per-trade pandas Timestamp parsing and dict lookups)
    on every single call. Precomputing it once and passing it in cut a
    meaningful amount of wall-clock time out of every Monte Carlo run,
    the walk-forward-aware GA, and Search Lab -- all of which call
    simulate_account many thousands of times per run -- without changing
    a single output value.
    """
    day_index_per_trade: list       # length == n_trades; which day (0-based, chronological) each trade belongs to
    is_last_of_day: list            # length == n_trades; True where a trade is the last one of its calendar day
    day_dates: list                 # length == n_days; the normalized date for each day index
    n_days: int


def precompute_day_structure(trade_dates: list) -> DayStructure:
    """Builds the (dates-only) bookkeeping simulate_account needs, once,
    so it can be reused across many calls that all share the same
    `trade_dates` but different `trade_pnls` (exactly the Monte Carlo
    engine's resampling pattern). See DayStructure's docstring."""
    dates_norm = [pd.Timestamp(d).normalize() for d in trade_dates]
    day_index_map: dict = {}
    day_order: list = []
    day_index_per_trade: list = []
    for d in dates_norm:
        idx = day_index_map.get(d)
        if idx is None:
            idx = len(day_order)
            day_index_map[d] = idx
            day_order.append(d)
        day_index_per_trade.append(idx)
    n = len(dates_norm)
    is_last_of_day = [
        (i == n - 1) or (dates_norm[i] != dates_norm[i + 1])
        for i in range(n)
    ]
    return DayStructure(
        day_index_per_trade=day_index_per_trade, is_last_of_day=is_last_of_day,
        day_dates=day_order, n_days=len(day_order),
    )


def simulate_account(
    trade_pnls: list[float],
    trade_dates: list,
    rules: PropRules,
    _day_structure: "DayStructure | None" = None,
) -> AccountSimResult:
    """
    trade_pnls: P&L of each trade (account-currency $), in chronological order
    trade_dates: date (or datetime) of each trade, same order/length as trade_pnls

    _day_structure: internal fast-path for callers (the Monte Carlo engine)
    that invoke this function many times with the SAME trade_dates and only
    trade_pnls changing -- pass a DayStructure from precompute_day_structure()
    once, computed from trade_dates, to skip re-deriving it on every call.
    Every other caller can ignore this parameter entirely; it's derived
    from trade_dates automatically when omitted, with identical results.
    """
    if len(trade_pnls) == 0:
        return AccountSimResult(
            passed_evaluation=False, failed=False, failure_reason="No trades generated.",
            failure_day_index=None, days_to_pass=None, first_payout_day_index=None,
            first_payout_amount=None, final_balance=rules.account_size,
        )

    balance = rules.account_size
    trailing_peak = rules.account_size
    static_floor = rules.account_size * (1 - rules.max_drawdown_pct / 100.0)

    stage = "evaluation"
    passed_evaluation = False
    days_to_pass = None
    failed = False
    failure_reason = None
    failure_day_index = None

    day_structure = _day_structure if _day_structure is not None else precompute_day_structure(trade_dates)
    day_index_per_trade = day_structure.day_index_per_trade
    is_last_of_day = day_structure.is_last_of_day
    day_dates = day_structure.day_dates
    daily_pnl = [0.0] * day_structure.n_days
    day_profit_history = [0.0] * day_structure.n_days  # day index -> cumulative pnl that day

    payouts: list[PayoutEvent] = []
    first_payout_day_index = None
    first_payout_amount = None

    last_payout_day_index = -10 ** 9
    payout_baseline_balance = rules.account_size  # balance at last payout (or eval pass, for funded profit tracking)
    max_dd_pct_reached = 0.0

    total_profit_since_start = 0.0
    best_day_profit = 0.0

    last_day_idx_reached = -1

    for i, pnl in enumerate(trade_pnls):
        cur_day_idx = day_index_per_trade[i]
        last_day_idx_reached = cur_day_idx

        balance += pnl
        daily_pnl[cur_day_idx] += pnl
        day_profit_history[cur_day_idx] += pnl
        total_profit_since_start += pnl
        best_day_profit = max(best_day_profit, day_profit_history[cur_day_idx])

        check_now = (rules.drawdown_check_mode == "intrabar") or is_last_of_day[i]

        if not check_now:
            # "eod" mode: this trade isn't the day's last -- defer the
            # peak/drawdown/failure evaluation until the day is complete.
            continue

        trailing_peak = max(trailing_peak, balance)
        if rules.drawdown_type == "trailing":
            dd_floor = trailing_peak * (1 - rules.max_drawdown_pct / 100.0)
        else:
            dd_floor = static_floor
        current_dd_pct = max(0.0, (trailing_peak - balance) / trailing_peak * 100.0) if trailing_peak else 0.0
        max_dd_pct_reached = max(max_dd_pct_reached, current_dd_pct)

        # --- Failure checks (apply in both evaluation and funded stages) ---
        if daily_pnl[cur_day_idx] <= -rules.account_size * (rules.daily_loss_limit_pct / 100.0):
            failed = True
            failure_reason = "daily_loss_limit"
            failure_day_index = cur_day_idx
            break

        if balance <= dd_floor:
            failed = True
            failure_reason = f"max_drawdown ({rules.drawdown_type})"
            failure_day_index = cur_day_idx
            break

        # --- Evaluation pass check ---
        if stage == "evaluation":
            target_balance = rules.account_size * (1 + rules.evaluation_profit_target_pct / 100.0)
            trading_days_so_far = cur_day_idx + 1
            if balance >= target_balance and trading_days_so_far >= rules.min_trading_days:
                consistency_ok = True
                if rules.consistency_rule_pct is not None and total_profit_since_start > 0:
                    consistency_ok = (best_day_profit / total_profit_since_start * 100.0) <= rules.consistency_rule_pct
                if consistency_ok:
                    stage = "funded"
                    passed_evaluation = True
                    days_to_pass = trading_days_so_far
                    payout_baseline_balance = balance
                    last_payout_day_index = cur_day_idx  # start payout clock from pass date

        # --- Funded stage payout check ---
        elif stage == "funded":
            profit_since_baseline = balance - payout_baseline_balance
            required_profit = rules.account_size * (rules.payout_threshold_pct / 100.0) \
                + rules.account_size * (rules.required_buffer_pct / 100.0)
            days_since_last_payout = cur_day_idx - last_payout_day_index
            if profit_since_baseline > required_profit and days_since_last_payout >= rules.payout_frequency_days:
                withdrawable = profit_since_baseline - rules.account_size * (rules.required_buffer_pct / 100.0)
                if rules.payout_cap_pct is not None:
                    withdrawable = min(withdrawable, profit_since_baseline * (rules.payout_cap_pct / 100.0))
                withdrawable = max(withdrawable, 0.0)
                if withdrawable > 0:
                    balance -= withdrawable
                    payout = PayoutEvent(
                        day_index=cur_day_idx,
                        date=str(day_dates[cur_day_idx].date()),
                        amount=withdrawable,
                        balance_after=balance,
                    )
                    payouts.append(payout)
                    if first_payout_day_index is None:
                        first_payout_day_index = cur_day_idx
                        first_payout_amount = withdrawable
                    payout_baseline_balance = balance
                    last_payout_day_index = cur_day_idx
                    trailing_peak = max(trailing_peak, balance)

    return AccountSimResult(
        passed_evaluation=passed_evaluation,
        failed=failed,
        failure_reason=failure_reason,
        failure_day_index=failure_day_index,
        days_to_pass=days_to_pass,
        first_payout_day_index=first_payout_day_index,
        first_payout_amount=first_payout_amount,
        payouts=payouts,
        final_balance=balance,
        max_drawdown_pct_reached=max_dd_pct_reached,
        trading_days_count=last_day_idx_reached + 1,
    )


def summarize_single_run(result: AccountSimResult) -> dict:
    """
    Section-4-style summary for the single deterministic historical trade
    sequence. Pass/fail/payout rates here are necessarily 0% or 100% since
    there is only one run -- statistical distributions over many possible
    trade sequences are the job of the Monte Carlo engine (section 5/6).
    """
    return {
        "evaluation_pass_pct": 100.0 if result.passed_evaluation else 0.0,
        "evaluation_failure_pct": 100.0 if result.failed and not result.passed_evaluation else 0.0,
        "first_payout_pct": 100.0 if result.reached_first_payout else 0.0,
        "days_to_pass": result.days_to_pass,
        "days_to_first_payout": result.first_payout_day_index,
        "first_payout_amount": result.first_payout_amount,
        "total_payouts": len(result.payouts),
        "total_payout_amount": result.total_payout_amount,
        "final_balance": result.final_balance,
        "max_drawdown_pct": result.max_drawdown_pct_reached,
        "failure_reason": result.failure_reason,
        "trading_days_count": result.trading_days_count,
    }
