"""
Ported from: dakar-sniper-v2.mq5 ("Dakar Sniper V2") and prop-sniper-v1.mq5
("Prop Sniper V1") -- both by the same author, both implementing the
identical "Judas Sweep" London Killzone strategy with only cosmetic
differences (variable names, exact session-trigger hour). Only ONE port
is provided since running both would just be running the same idea twice.

STRATEGY LOGIC (unchanged from the original):
  Range      : the Asian session's high/low, measured from 00:00 up to
               ASIAN_SESSION_END_HOUR (server-hour proxy, see note below)
  Sweep      : during the London Killzone window, price pokes ABOVE the
               Asian high but then CLOSES back below it (a false breakout
               / liquidity grab) with a strong bearish rejection candle
               (body > 1.5x ATR)                              -> SHORT
               ...or symmetrically pokes BELOW the Asian low and closes
               back above it with a bullish rejection candle  -> LONG
  Stop loss  : just beyond the sweep wick (wick distance + a small buffer)
  Take profit: stop distance * 3.0 (Risk:Reward 1:3, matches RiskRewardRatio)
  One trade at a time, one attempt per Killzone window per day (matches
  the original's "one trade per M5 candle, only during the Killzone hour"
  gating)

WHAT DIDN'T CARRY OVER, AND WHY:
  - Phase 1 / Phase 2 risk auto-switching and the associated daily/max
    drawdown circuit breakers: these depend on the ACCOUNT'S running
    profit and equity, which generate_signals(df) cannot see (it runs
    once, statelessly, before any P&L exists -- see the python.py
    adapter's docstring). Use a single, fixed risk-per-trade via T58's
    RiskConfig instead (the original's Phase 1 risk was 2%; its Phase 2
    "survival mode" risk was 0.75% -- pick whichever matches how you'd
    actually run the account, or run this twice at each setting).
  - The 90%-of-limit daily/max-drawdown safety margins: same reasoning --
    set RiskConfig.daily_loss_limit_pct and the Prop Rules tab's
    drawdown fields to match your actual firm's numbers; T58 enforces
    these at the engine level, which is the correct place for them.
  - The "Africa latency" broker-offset/spread-filter cosmetics: these are
    execution-environment concerns (slippage tolerance, spread filtering),
    already covered by T58's own RiskConfig.slippage_pips/spread_pips.

WHAT NEEDS YOUR ATTENTION: the Asian-range/Killzone hours below are UTC-
ish placeholders (Asian range = 00:00-07:00, Killzone = 08:00-10:00) --
the original computed these relative to the BROKER'S server clock via a
configurable offset. Check your CSV's timestamp timezone and adjust
ASIAN_SESSION_END_HOUR / LONDON_KZ_START / LONDON_KZ_END below to match,
or the "Asian range" and "Killzone" won't line up with the sessions you
actually intend.

NO LOOKAHEAD: the Asian high/low for "today" is computed only from bars
whose timestamp falls within today's 00:00-ASIAN_SESSION_END_HOUR window,
using groupby-per-day + expanding max/min -- so at any bar inside that
window, only that day's bars UP TO AND INCLUDING the current bar
contribute (the original updates its Asian range as the session runs,
not with perfect hindsight over the full session); the sweep/rejection
check uses only the current bar's own OHLC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "Judas Sweep London Killzone (ported from MQL5, dakar-sniper-v2/prop-sniper-v1)"

ASIAN_SESSION_END_HOUR = 7    # Asian range = today's bars with hour < this
LONDON_KZ_START = 8           # Killzone window (inclusive)
LONDON_KZ_END = 10            # Killzone window (exclusive)

ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.5
WICK_BUFFER_PIPS_EQUIV = 0.0002   # ~20 points on a 5-digit FX quote; adjust for your instrument
RR_RATIO = 3.0


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def generate_signals(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"])
    date = ts.dt.date
    hour = ts.dt.hour

    atr = _atr(df, ATR_PERIOD)

    # Asian range: running (expanding) high/low of TODAY's bars with
    # hour < ASIAN_SESSION_END_HOUR, forward-filled across the rest of the
    # day once the Asian session ends (mirrors the original reading a
    # fixed iHighest/iLowest over that morning's completed bars).
    is_asian = hour < ASIAN_SESSION_END_HOUR
    asian_high_running = df["high"].where(is_asian).groupby(date).cummax()
    asian_low_running = df["low"].where(is_asian).groupby(date).cummin()
    asian_high = asian_high_running.groupby(date).ffill()
    asian_low = asian_low_running.groupby(date).ffill()

    in_killzone = (hour >= LONDON_KZ_START) & (hour < LONDON_KZ_END)

    body = df["close"] - df["open"]
    displacement_ok = body.abs() > (atr * DISPLACEMENT_ATR_MULT)

    # Bearish Judas Swing: sweep the Asian high, close back below it.
    sell_sweep = (
        in_killzone
        & (df["high"] > asian_high)
        & (df["close"] < asian_high)
        & (df["close"] < df["open"])
        & displacement_ok
    )
    # Bullish Judas Swing: sweep the Asian low, close back above it.
    buy_sweep = (
        in_killzone
        & (df["low"] < asian_low)
        & (df["close"] > asian_low)
        & (df["close"] > df["open"])
        & displacement_ok
    )

    sl_dist_sell = (df["high"] - df["close"]) + WICK_BUFFER_PIPS_EQUIV
    sl_dist_buy = (df["close"] - df["low"]) + WICK_BUFFER_PIPS_EQUIV

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[buy_sweep] = 1.0
    raw[sell_sweep] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    # The sweep/rejection wick is a ONE-BAR EVENT -- unlike a smooth
    # trailing indicator (ATR, EMA), it isn't meaningful to recompute from
    # whatever bar happens to be current many bars into a held position.
    # So the captured distance at the actual trigger bar is forward-filled
    # alongside the held signal, rather than recomputed from each new
    # bar's own (unrelated) OHLC -- a same-direction re-entry after a
    # stop-out reuses the original sweep's risk framing instead of
    # inventing a new number from irrelevant later price action.
    raw_sl = pd.Series(np.nan, index=df.index)
    raw_sl[buy_sweep] = sl_dist_buy[buy_sweep]
    raw_sl[sell_sweep] = sl_dist_sell[sell_sweep]
    stop_loss_distance = raw_sl.ffill()
    # Fallback for the (rare) case the signal is held before any trigger's
    # distance has ever been captured -- keeps every bar's stop defined.
    stop_loss_distance = stop_loss_distance.fillna(atr * DISPLACEMENT_ATR_MULT)

    take_profit_distance = stop_loss_distance * RR_RATIO

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance

    return signal
