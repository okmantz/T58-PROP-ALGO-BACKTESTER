"""
Bar-by-bar trade execution simulator.

Consumes OHLCV data + a standardized signal series (-1/0/1) + risk config
and produces a discrete trade list. Entries occur on the bar the signal
changes (filled at that bar's close, adjusted for spread/slippage); each
open trade is then walked forward bar-by-bar checking for stop-loss /
take-profit intrabar hits (using high/low) or a signal-driven exit.

This is intentionally a straightforward, transparent simulation appropriate
for an MVP -- no partial fills, no multi-leg positions, one open trade at a
time (consistent with the standardized long/flat/short signal model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskState
from app.backtest.risk import RiskConfig


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str        # "stop_loss" | "take_profit" | "signal" | "end_of_data"
    commission: float
    equity_after: float
    initial_risk: float | None = None  # |entry - stop| in raw price units, at entry time
    adaptive_risk_multiplier: float = 1.0     # position-size multiplier in effect when this trade was OPENED
    adaptive_risk_rules_active: tuple = ()    # human-readable labels of whichever adaptive-risk rule(s) fired

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = str(self.entry_time)
        d["exit_time"] = str(self.exit_time)
        return d


DEFAULT_STOP_PCT_OF_PRICE = 0.01  # 1% of entry price, used only when a strategy defines no stop at all


def run_execution(
    df: pd.DataFrame,
    signals: pd.Series,
    risk: RiskConfig,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
    stop_loss_distance: pd.Series | None = None,
    take_profit_distance: pd.Series | None = None,
    trailing_stop_distance: pd.Series | None = None,
    breakeven_trigger_r: float | None = None,
    adaptive_risk: AdaptiveRiskConfig | None = None,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    Returns (trades, equity_curve_df) where equity_curve_df has columns
    [timestamp, equity] for every bar in df.

    stop_loss_distance / take_profit_distance: optional per-bar distances
    in raw price units (e.g. an ATR-multiple stop). When provided, these
    take precedence over the fixed stop_loss_pips/take_profit_pips for
    that entry bar.

    trailing_stop_distance: optional per-bar distance in raw price units.
    The distance is captured once at trade entry and then used to ratchet
    the stop toward price as the trade moves favorably; it never widens.

    breakeven_trigger_r: once open profit reaches this multiple of the
    trade's initial risk (entry-to-stop distance), the stop is moved to
    the entry price (only ever tightened, never loosened).

    adaptive_risk: optional declarative money-management overlay (see
    app.backtest.adaptive_risk) -- scales the SIZE of each new entry by
    whatever multiplier its rules currently imply (consecutive losses,
    today's realized P&L, progress toward a profit target). Evaluated
    fresh at every entry decision using only trade outcomes ALREADY
    realized as of that bar, so it introduces no lookahead. None/disabled
    means every entry uses its full nominal size, unchanged from before
    this parameter existed.
    """
    n = len(df)
    equity = risk.initial_balance
    equity_curve = []
    trades: list[Trade] = []
    adaptive_state = AdaptiveRiskState(initial_balance=risk.initial_balance)

    # Rolling ATR (14-bar, simple True Range mean) purely as a volatility
    # yardstick for the fixed-pips-vs-instrument sanity check below -- this
    # is intentionally independent of anything a strategy itself computes.
    # See the pip_scale_mismatch check for why a price-ratio band alone
    # isn't enough: a fixed-pips stop can land just inside that ratio band
    # (e.g. ~0.03% of price) yet still be a small fraction of a genuinely
    # volatile instrument's typical bar-to-bar range -- an index at
    # several thousand points per share with a triple-digit-point ATR is
    # the textbook case. Comparing the stop distance to the instrument's
    # own recent ATR catches that; comparing it only to price does not.
    _tr_high = df["high"].astype(float)
    _tr_low = df["low"].astype(float)
    _tr_prev_close = df["close"].astype(float).shift(1)
    _true_range = pd.concat(
        [
            _tr_high - _tr_low,
            (_tr_high - _tr_prev_close).abs(),
            (_tr_low - _tr_prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    _atr_for_mismatch_check = _true_range.rolling(14, min_periods=1).mean().to_numpy()

    open_trade: dict | None = None
    trades_today: dict[pd.Timestamp, int] = {}
    pnl_today: dict[pd.Timestamp, float] = {}
    fallback_stop_count = 0
    pip_scale_mismatch_count = 0
    pip_scale_mismatch_worst_ratio = None  # smallest (stop_distance / entry_price) seen, for the warning
    atr_scale_mismatch_count = 0
    atr_scale_mismatch_worst_ratio = None  # smallest (stop_distance / ATR) seen, for the warning
    gap_loss_count = 0
    clamped_loss_count = 0
    account_blown = False
    account_blown_at = None
    daily_limit_amount = (
        risk.initial_balance * (risk.daily_loss_limit_pct / 100.0)
        if risk.daily_loss_limit_pct is not None
        else None
    )
    blown_floor = risk.account_blown_floor()  # None if no floor configured

    def _clamp_loss(pnl_value: float, equity_at_entry: float) -> float:
        """A single trade's loss can never realistically exceed what the
        account actually has to lose (negative-balance protection) or a
        configured hard per-trade ceiling -- see RiskConfig.max_trade_loss.
        Wins are never touched."""
        nonlocal clamped_loss_count
        if pnl_value >= 0 or not math.isfinite(pnl_value):
            return pnl_value
        cap = risk.max_trade_loss(equity_at_entry)
        if cap and -pnl_value > cap:
            clamped_loss_count += 1
            return -cap
        return pnl_value

    sig = signals.values
    ts = df["timestamp"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    sl_dist_vals = stop_loss_distance.values if stop_loss_distance is not None else None
    tp_dist_vals = take_profit_distance.values if take_profit_distance is not None else None
    trail_dist_vals = trailing_stop_distance.values if trailing_stop_distance is not None else None

    spread_price = risk.spread_pips * risk.pip_size
    slip_price = risk.slippage_pips * risk.pip_size

    force_closed_count = 0

    for i in range(n):
        bar_date = pd.Timestamp(ts[i]).normalize()

        # --- manage open trade: trailing stop / break-even, then stop/take intrabar ---
        if open_trade is not None:
            direction = open_trade["direction"]

            favorable_extreme = highs[i] if direction == 1 else lows[i]
            if direction == 1:
                open_trade["best_price"] = max(open_trade["best_price"], favorable_extreme)
            else:
                open_trade["best_price"] = min(open_trade["best_price"], favorable_extreme)

            # Mark-to-market daily-loss check, using the ADVERSE intrabar
            # extreme (low for a long, high for a short) rather than the
            # close. A prop firm's daily-loss floor is monitored on
            # floating equity in real time, not just on realized P&L at
            # the moment a trade happens to close — a trade that dips
            # deep underwater and recovers by the close of the bar can
            # still have breached (and been auto-liquidated at) the daily
            # floor intrabar. Checking only realized same-day P&L (the
            # old behavior) silently let strategies "survive" daily-loss
            # breaches that a real funded account would have been
            # stopped out of.
            adverse_extreme = lows[i] if direction == 1 else highs[i]
            floating_adverse_pnl = (adverse_extreme - open_trade["entry_price"]) * open_trade["size"] * direction
            day_realized_so_far = pnl_today.get(bar_date, 0.0)
            if (
                daily_limit_amount is not None
                and (day_realized_so_far + floating_adverse_pnl) <= -daily_limit_amount
            ):
                # Force-close at the adverse extreme (the point the real
                # account would have been liquidated at), paying the same
                # round-turn cost as any other exit.
                filled_exit_price = adverse_extreme - (spread_price + slip_price) * direction
                pnl = (filled_exit_price - open_trade["entry_price"]) * open_trade["size"] * direction
                pnl -= risk.commission_per_trade
                if not math.isfinite(pnl):
                    pnl = 0.0
                pnl = _clamp_loss(pnl, open_trade["equity_at_entry"])
                equity += pnl
                trades.append(Trade(
                    entry_time=open_trade["entry_time"],
                    exit_time=pd.Timestamp(ts[i]),
                    direction=direction,
                    entry_price=open_trade["entry_price"],
                    exit_price=filled_exit_price,
                    size=open_trade["size"],
                    pnl=pnl,
                    pnl_pct=(pnl / open_trade["equity_at_entry"]) * 100 if open_trade["equity_at_entry"] else 0.0,
                    exit_reason="daily_loss_limit_forced_close",
                    commission=risk.commission_per_trade,
                    equity_after=equity,
                    initial_risk=open_trade["initial_risk"],
                    adaptive_risk_multiplier=open_trade["adaptive_multiplier"],
                    adaptive_risk_rules_active=tuple(open_trade["adaptive_rules_active"]),
                ))
                adaptive_state.record_trade_close(pnl, is_new_day=bar_date not in pnl_today)
                pnl_today[bar_date] = pnl_today.get(bar_date, 0.0) + pnl
                open_trade = None
                force_closed_count += 1
                equity_curve.append((ts[i], equity))
                continue

            stop = open_trade["stop_price"]

            # Break-even: once profit reaches the configured R multiple,
            # move the stop to entry (only ever tightens the stop).
            if (
                breakeven_trigger_r is not None
                and open_trade["initial_risk"]
                and not open_trade["breakeven_done"]
            ):
                profit_dist = (open_trade["best_price"] - open_trade["entry_price"]) * direction
                if profit_dist >= breakeven_trigger_r * open_trade["initial_risk"]:
                    candidate = open_trade["entry_price"]
                    if stop is None or (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                        stop = candidate
                    open_trade["breakeven_done"] = True

            # Trailing stop: ratchet toward price, never away from it.
            if open_trade.get("trailing_distance"):
                candidate = open_trade["best_price"] - direction * open_trade["trailing_distance"]
                if stop is None or (direction == 1 and candidate > stop) or (direction == -1 and candidate < stop):
                    stop = candidate

            open_trade["stop_price"] = stop
            take = open_trade["take_price"]
            exit_price = None
            reason = None

            if direction == 1:
                if stop is not None and lows[i] <= stop:
                    # Honest fill: a resting stop that the bar gapped
                    # straight through does NOT fill at the stop price —
                    # it fills at the open, which is worse. Filling every
                    # stop at its exact level is one of the most common
                    # sources of a fake backtest edge.
                    exit_price, reason = min(stop, opens[i]), "stop_loss"
                elif take is not None and highs[i] >= take:
                    exit_price, reason = take, "take_profit"
            else:
                if stop is not None and highs[i] >= stop:
                    exit_price, reason = max(stop, opens[i]), "stop_loss"
                elif take is not None and lows[i] <= take:
                    exit_price, reason = take, "take_profit"

            # signal-driven exit (flat or reversal) takes effect at close if no SL/TP hit
            if exit_price is None and sig[i] != direction:
                exit_price, reason = closes[i], "signal"

            if exit_price is not None:
                # Every exit is a real transaction and pays the same
                # round-turn cost the entry did — crediting a stop/take/
                # signal exit at its exact quoted level (with no spread or
                # slippage) flatters every single trade by that amount.
                filled_exit_price = exit_price - (spread_price + slip_price) * direction
                pnl = (filled_exit_price - open_trade["entry_price"]) * open_trade["size"] * direction
                pnl -= risk.commission_per_trade
                if not math.isfinite(pnl):
                    # Guard against a runaway/degenerate trade (e.g. an
                    # entry sized off a near-zero ATR-based stop distance)
                    # ever corrupting the equity curve with NaN/inf. This
                    # should be rare; if you see it often, your stop
                    # distance or pip size for this instrument is almost
                    # certainly misconfigured.
                    pnl = 0.0
                    reason = f"{reason}_invalid_pnl_skipped"
                if (
                    reason == "stop_loss"
                    and open_trade["initial_risk"]
                    and abs(pnl) > 3 * open_trade["initial_risk"] * open_trade["size"]
                ):
                    # The stop fired, but the fill was still several times
                    # worse than the risk this trade was sized for -- almost
                    # always a genuine price gap jumping straight past the
                    # resting stop in one bar, not a bug. Surfaced as a
                    # warning below so a handful of outsized losses in an
                    # otherwise-sane backtest don't get mistaken for a
                    # broken engine.
                    gap_loss_count += 1
                pnl = _clamp_loss(pnl, open_trade["equity_at_entry"])
                equity += pnl
                trades.append(Trade(
                    entry_time=open_trade["entry_time"],
                    exit_time=pd.Timestamp(ts[i]),
                    direction=direction,
                    entry_price=open_trade["entry_price"],
                    exit_price=filled_exit_price,
                    size=open_trade["size"],
                    pnl=pnl,
                    pnl_pct=(pnl / open_trade["equity_at_entry"]) * 100 if open_trade["equity_at_entry"] else 0.0,
                    exit_reason=reason,
                    commission=risk.commission_per_trade,
                    equity_after=equity,
                    initial_risk=open_trade["initial_risk"],
                    adaptive_risk_multiplier=open_trade["adaptive_multiplier"],
                    adaptive_risk_rules_active=tuple(open_trade["adaptive_rules_active"]),
                ))
                adaptive_state.record_trade_close(pnl, is_new_day=bar_date not in pnl_today)
                pnl_today[bar_date] = pnl_today.get(bar_date, 0.0) + pnl
                open_trade = None

        # --- mark-to-market equity curve point for this bar ---
        # Realized equity plus the floating P&L of any still-open
        # position, valued at this bar's close. This is what feeds the
        # drawdown statistics (app.backtest.statistics) -- using realized
        # equity alone understated true intrabar/multi-day drawdown
        # whenever a trade was open across the peak.
        if open_trade is not None:
            floating_close_pnl = (
                (closes[i] - open_trade["entry_price"]) * open_trade["size"] * open_trade["direction"]
            )
            mtm_equity = equity + floating_close_pnl
        else:
            mtm_equity = equity
        equity_curve.append((ts[i], mtm_equity))

        # --- account-blown circuit breaker ---
        # A real prop/broker account is terminated the instant it breaches
        # the firm's max-drawdown floor -- it does not keep "trading"
        # itself into ever-deeper negative equity. Once realized equity
        # crosses that floor (or the account has literally run out of
        # money), stop opening any new trades for the remainder of the
        # run; this is what makes a single misconfigured/gapped trade
        # (see max_trade_loss above) unable to cascade into the kind of
        # -$50,000-on-a-$50,000-account or -$2.5M results this was built
        # to prevent. Any already-open trade still manages normally
        # (stop/target/signal exits) -- only NEW entries are blocked.
        if not account_blown and (equity <= 0 or (blown_floor is not None and equity <= blown_floor)):
            account_blown = True
            account_blown_at = pd.Timestamp(ts[i])

        # --- consider new entry ---
        day_realized_pnl = pnl_today.get(bar_date, 0.0)
        daily_limit_breached = (
            daily_limit_amount is not None and day_realized_pnl <= -daily_limit_amount
        )
        if open_trade is None and sig[i] != 0 and not daily_limit_breached and not account_blown:
            n_today = trades_today.get(bar_date, 0)
            if n_today < risk.max_trades_per_day:
                direction = int(sig[i])
                raw_price = closes[i]
                entry_price = raw_price + (spread_price + slip_price) * direction

                bar_sl_distance = float(sl_dist_vals[i]) if sl_dist_vals is not None and not pd.isna(sl_dist_vals[i]) else None
                bar_tp_distance = float(tp_dist_vals[i]) if tp_dist_vals is not None and not pd.isna(tp_dist_vals[i]) else None
                bar_trail_distance = float(trail_dist_vals[i]) if trail_dist_vals is not None and not pd.isna(trail_dist_vals[i]) else None

                used_fallback_stop = False
                if not bar_sl_distance and not stop_loss_pips:
                    # The strategy defined no stop loss at all (no ATR-based
                    # distance and no fixed pips). Sizing off a hardcoded
                    # small pip count here would be wrong for any
                    # non-FX-scaled instrument (e.g. gold, indices, crypto)
                    # and would also leave the position completely
                    # unprotected. Fall back to a stop sized as a percentage
                    # of the entry price instead — this scales correctly
                    # regardless of instrument or pip size.
                    bar_sl_distance = abs(raw_price) * DEFAULT_STOP_PCT_OF_PRICE
                    used_fallback_stop = True

                if bar_sl_distance:
                    sizing_pips = bar_sl_distance / risk.pip_size if risk.pip_size else 0
                else:
                    sizing_pips = stop_loss_pips or 0
                size = risk.position_size(equity, sizing_pips)

                adaptive_multiplier = 1.0
                adaptive_rules_active: list[str] = []
                if adaptive_risk is not None and adaptive_risk.enabled:
                    adaptive_multiplier = adaptive_state.active_multiplier(adaptive_risk)
                    adaptive_rules_active = adaptive_state.active_rule_labels(adaptive_risk)
                    size *= adaptive_multiplier

                if not math.isfinite(size) or size <= 0:
                    # Degenerate sizing (e.g. an ATR-based stop distance
                    # that rounds to ~0 for this bar) — skip this entry
                    # rather than opening a trade with an invalid size.
                    trades_today[bar_date] = n_today  # no-op, keeps loop simple
                else:
                    if used_fallback_stop:
                        fallback_stop_count += 1
                    stop_price = None
                    take_price = None
                    if bar_sl_distance:
                        stop_price = entry_price - direction * bar_sl_distance
                    elif stop_loss_pips:
                        fixed_stop_distance = stop_loss_pips * risk.pip_size
                        stop_price = entry_price - direction * fixed_stop_distance
                        # Sanity-check the fixed pip stop against this bar's
                        # actual price level. A strategy's STOP_LOSS_PIPS is
                        # only meaningful if risk.pip_size matches the
                        # instrument it's being tested against (e.g. 0.0001
                        # for 4-decimal FX pairs) -- testing an FX-calibrated
                        # strategy against gold, an index, crypto, or a
                        # differently-quoted instrument while pip_size stays
                        # at its FX default silently turns a "25 pip" stop
                        # into a wildly wrong fraction of price, producing
                        # position sizes (and eventual losses) that bear no
                        # relation to the intended risk. This never changes
                        # behavior -- it only counts how often it happens so
                        # a clear warning can be raised below.
                        if raw_price:
                            distance_ratio = abs(fixed_stop_distance) / abs(raw_price)
                            if distance_ratio < 0.0002 or distance_ratio > 0.25:
                                pip_scale_mismatch_count += 1
                                if pip_scale_mismatch_worst_ratio is None or distance_ratio < pip_scale_mismatch_worst_ratio:
                                    pip_scale_mismatch_worst_ratio = distance_ratio
                        # Second, independent check: the same fixed-pips
                        # stop compared to this instrument's own recent ATR
                        # rather than to its raw price level. A stop can be
                        # a perfectly ordinary-looking fraction of price
                        # (comfortably inside the 0.02%-25% band above) and
                        # still be tiny next to how much this instrument
                        # actually moves bar to bar -- that's still a real
                        # pip_size/instrument mismatch, just one the
                        # price-ratio check alone can miss.
                        bar_atr = _atr_for_mismatch_check[i]
                        if bar_atr and bar_atr > 0:
                            atr_ratio = abs(fixed_stop_distance) / bar_atr
                            if atr_ratio < 0.15:
                                atr_scale_mismatch_count += 1
                                if atr_scale_mismatch_worst_ratio is None or atr_ratio < atr_scale_mismatch_worst_ratio:
                                    atr_scale_mismatch_worst_ratio = atr_ratio
                    if bar_tp_distance:
                        take_price = entry_price + direction * bar_tp_distance
                    elif take_profit_pips:
                        take_price = entry_price + direction * take_profit_pips * risk.pip_size

                    initial_risk = abs(entry_price - stop_price) if stop_price is not None else None

                    open_trade = {
                        "entry_time": pd.Timestamp(ts[i]),
                        "direction": direction,
                        "entry_price": entry_price,
                        "size": size,
                        "stop_price": stop_price,
                        "take_price": take_price,
                        "equity_at_entry": equity,
                        "best_price": entry_price,
                        "initial_risk": initial_risk,
                        "breakeven_done": False,
                        "trailing_distance": bar_trail_distance,
                        "adaptive_multiplier": adaptive_multiplier,
                        "adaptive_rules_active": adaptive_rules_active,
                    }
                    trades_today[bar_date] = n_today + 1

    # close any still-open trade at final bar close
    if open_trade is not None:
        i = n - 1
        direction = open_trade["direction"]
        exit_price = closes[i] - (spread_price + slip_price) * direction
        pnl = (exit_price - open_trade["entry_price"]) * open_trade["size"] * direction
        pnl -= risk.commission_per_trade
        if not math.isfinite(pnl):
            pnl = 0.0
        pnl = _clamp_loss(pnl, open_trade["equity_at_entry"])
        equity += pnl
        trades.append(Trade(
            entry_time=open_trade["entry_time"],
            exit_time=pd.Timestamp(ts[i]),
            direction=direction,
            entry_price=open_trade["entry_price"],
            exit_price=exit_price,
            size=open_trade["size"],
            pnl=pnl,
            pnl_pct=(pnl / open_trade["equity_at_entry"]) * 100 if open_trade["equity_at_entry"] else 0.0,
            exit_reason="end_of_data",
            commission=risk.commission_per_trade,
            equity_after=equity,
            initial_risk=open_trade["initial_risk"],
            adaptive_risk_multiplier=open_trade["adaptive_multiplier"],
            adaptive_risk_rules_active=tuple(open_trade["adaptive_rules_active"]),
        ))
        adaptive_state.record_trade_close(pnl, is_new_day=pd.Timestamp(ts[i]).normalize() not in pnl_today)

    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])

    if account_blown:
        import warnings
        warnings.warn(
            f"Account BLOWN at {account_blown_at}: equity crossed the configured "
            f"account-survivability floor (max_account_drawdown_pct) or hit zero. "
            "No new trades were opened for the remainder of this run -- exactly "
            "like a real prop/broker account being terminated rather than being "
            "allowed to keep 'trading' itself into deeper negative equity. If you "
            "want to know what happens after a re-funded account, run a fresh "
            "backtest starting from that point rather than reading further trades "
            "on this run as real.",
            RuntimeWarning,
        )

    if clamped_loss_count:
        import warnings
        warnings.warn(
            f"{clamped_loss_count} trade(s) had their simulated loss capped at "
            "RiskConfig.max_trade_loss (a hard per-trade ceiling, default 3x the "
            "trade's own intended risk) instead of being allowed to realize the "
            "full computed loss. This models real negative-balance protection / "
            "firm loss floors -- without it, a pip_size mismatch or an extreme "
            "gap-through fill could report a single trade losing more than the "
            "entire account, which cannot happen live. If you see this often, "
            "the underlying cause (usually pip_scale_mismatch, see any warning "
            "above) is still worth fixing -- capping the number doesn't fix the "
            "sizing bug that produced it.",
            RuntimeWarning,
        )

    if force_closed_count:
        import warnings
        warnings.warn(
            f"{force_closed_count} trade(s) were force-closed intrabar because "
            "floating losses breached the configured daily_loss_limit_pct before "
            "the trade's own stop/target/signal exit would have fired. This "
            "mirrors a real prop firm auto-liquidating a funded account the "
            "moment equity crosses the daily floor, which realized-P&L-only "
            "accounting previously missed.",
            RuntimeWarning,
        )

    if fallback_stop_count:
        import warnings
        warnings.warn(
            f"{fallback_stop_count} trade(s) had no stop loss defined by the "
            f"strategy at all (no fixed pips, no ATR-based distance) — a "
            f"{DEFAULT_STOP_PCT_OF_PRICE * 100:.0f}%-of-price protective stop "
            "was used instead purely for sane position sizing and account "
            "protection. Add a real stop loss / STOP_LOSS_PIPS to the "
            "strategy for accurate results.",
            RuntimeWarning,
        )

    if pip_scale_mismatch_count:
        import warnings
        warnings.warn(
            f"{pip_scale_mismatch_count} trade(s) had a fixed-pips stop "
            f"whose price distance was an implausible fraction of this "
            f"instrument's price (as small as {pip_scale_mismatch_worst_ratio:.5%} "
            f"of price on the worst one). This almost always means the "
            f"configured pip_size ({risk.pip_size}) doesn't match the "
            "instrument actually being tested — e.g. an FX-calibrated "
            "strategy (pip_size 0.0001) run against gold, an index, crypto, "
            "or a JPY pair. Position sizing and every stop distance for "
            "these trades is unreliable. Set pip_size to match the real "
            "instrument (e.g. 0.01 for gold/JPY pairs) in the Risk "
            "configuration before trusting this result.",
            RuntimeWarning,
        )

    if atr_scale_mismatch_count:
        import warnings
        warnings.warn(
            f"{atr_scale_mismatch_count} trade(s) had a fixed-pips stop "
            f"under 15% of this instrument's own recent ATR (as little as "
            f"{atr_scale_mismatch_worst_ratio:.1%} of one bar's typical "
            f"range on the worst one) — a stop that tight gets taken out "
            f"by ordinary noise almost every time, regardless of whether "
            f"the strategy's entry logic is any good. This can happen even "
            f"when the stop looks like a normal fraction of price (see "
            f"pip_scale_mismatch above, if also shown): a high-priced but "
            f"volatile instrument, such as an equity index priced in the "
            f"thousands, can still move far more per bar than a fixed pip "
            f"count sized for a lower-volatility instrument like gold or "
            f"FX ever assumed. Set pip_size to match the real instrument "
            "(or click \"detect pip size from data\" on the Data tab) "
            "before trusting this result, or use an ATR-based dynamic stop "
            "instead of a fixed pip count.",
            RuntimeWarning,
        )

    if gap_loss_count:
        import warnings
        warnings.warn(
            f"{gap_loss_count} trade(s) hit their stop loss but filled at a "
            "price more than 3x further away than the intended risk -- the "
            "market gapped straight past the resting stop within a single "
            "bar. This is a real, honestly-simulated gap-through fill (see "
            "the 'honest fill' comment above), not an engine bug, but it "
            "means a handful of trades lost far more than the strategy's "
            "nominal per-trade risk. Worth checking whether this instrument/"
            "timeframe is prone to large single-bar gaps (news events, "
            "weekend opens, thin low-timeframe data) before trusting the "
            "risk-of-ruin numbers.",
            RuntimeWarning,
        )

    return trades, equity_df
