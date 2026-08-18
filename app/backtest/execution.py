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

from dataclasses import dataclass, asdict

import pandas as pd

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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = str(self.entry_time)
        d["exit_time"] = str(self.exit_time)
        return d


def run_execution(
    df: pd.DataFrame,
    signals: pd.Series,
    risk: RiskConfig,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    Returns (trades, equity_curve_df) where equity_curve_df has columns
    [timestamp, equity] for every bar in df.
    """
    n = len(df)
    equity = risk.initial_balance
    equity_curve = []
    trades: list[Trade] = []

    open_trade: dict | None = None
    trades_today: dict[pd.Timestamp, int] = {}

    sig = signals.values
    ts = df["timestamp"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    spread_price = risk.spread_pips * risk.pip_size
    slip_price = risk.slippage_pips * risk.pip_size

    for i in range(n):
        bar_date = pd.Timestamp(ts[i]).normalize()
        equity_curve.append((ts[i], equity))

        # --- manage open trade: check stop/take intrabar ---
        if open_trade is not None:
            direction = open_trade["direction"]
            stop = open_trade["stop_price"]
            take = open_trade["take_price"]
            exit_price = None
            reason = None

            if direction == 1:
                if stop is not None and lows[i] <= stop:
                    exit_price, reason = stop, "stop_loss"
                elif take is not None and highs[i] >= take:
                    exit_price, reason = take, "take_profit"
            else:
                if stop is not None and highs[i] >= stop:
                    exit_price, reason = stop, "stop_loss"
                elif take is not None and lows[i] <= take:
                    exit_price, reason = take, "take_profit"

            # signal-driven exit (flat or reversal) takes effect at close if no SL/TP hit
            if exit_price is None and sig[i] != direction:
                exit_price, reason = closes[i], "signal"

            if exit_price is not None:
                pnl = (exit_price - open_trade["entry_price"]) * open_trade["size"] * direction
                pnl -= risk.commission_per_trade
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
                    exit_reason=reason,
                    commission=risk.commission_per_trade,
                    equity_after=equity,
                ))
                open_trade = None

        # --- consider new entry ---
        if open_trade is None and sig[i] != 0:
            n_today = trades_today.get(bar_date, 0)
            if n_today < risk.max_trades_per_day:
                direction = int(sig[i])
                raw_price = closes[i]
                entry_price = raw_price + (spread_price + slip_price) * direction
                size = risk.position_size(equity, stop_loss_pips or 0)
                stop_price = None
                take_price = None
                if stop_loss_pips:
                    stop_price = entry_price - direction * stop_loss_pips * risk.pip_size
                if take_profit_pips:
                    take_price = entry_price + direction * take_profit_pips * risk.pip_size

                open_trade = {
                    "entry_time": pd.Timestamp(ts[i]),
                    "direction": direction,
                    "entry_price": entry_price,
                    "size": size,
                    "stop_price": stop_price,
                    "take_price": take_price,
                    "equity_at_entry": equity,
                }
                trades_today[bar_date] = n_today + 1

    # close any still-open trade at final bar close
    if open_trade is not None:
        i = n - 1
        direction = open_trade["direction"]
        exit_price = closes[i]
        pnl = (exit_price - open_trade["entry_price"]) * open_trade["size"] * direction
        pnl -= risk.commission_per_trade
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
        ))

    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    return trades, equity_df
