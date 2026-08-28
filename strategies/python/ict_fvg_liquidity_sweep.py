"""
Ported from: ict-sniper.pine ("Prop Firm ICT Sniper | Survival Mode",
Alexandre Albert Ndour). Original combined a confirmed swing-pivot
liquidity sweep with a Fair Value Gap displacement candle, sizing each
trade dynamically off the FVG's own boundary as the stop.

STRATEGY LOGIC (unchanged from the original):
  Pivots     : a swing high/low confirmed after SENSITIVITY bars on each
               side (`ta.pivothigh`/`ta.pivotlow` -- ported below as a
               proper CONFIRMED pivot: a pivot at bar i-SENSITIVITY only
               becomes known/usable starting at bar i, exactly like Pine's
               own behavior, not before)
  Sweep      : price pokes beyond the last confirmed swing high but
               closes back below it (Buy-Side Liquidity swept) -- or the
               mirror image on the low side (Sell-Side Liquidity swept)
  FVG        : a 3-bar Fair Value Gap in the direction of the sweep,
               requiring body > 1.5x ATR(14) and gap size > MIN_FVG_PIPS
  Entry      : sweep + same-direction FVG both present, AND price is
               currently trading back inside that FVG's zone
  Stop       : the far edge of the FVG zone (+ a small buffer) -- this is
               the strategy's own dynamic, per-trade stop, exactly as the
               original computed it
  Take profit: stop distance * 3.0 (Risk:Reward, matches rr_ratio)

WHAT DIDN'T CARRY OVER, AND WHY:
  - Phase 1 / Phase 2 risk-per-trade auto-switching and the daily-loss
    circuit breaker: account-state concerns generate_signals(df) cannot
    see statelessly (see the python.py adapter's docstring). Use a single
    fixed risk % via T58's RiskConfig, and set RiskConfig.daily_loss_
    limit_pct to match the original's daily-loss floor instead.

NO LOOKAHEAD:
  - Pivots are shifted forward by SENSITIVITY bars before use, so a pivot
    is only ever referenced starting on the bar it's actually confirmed
    on -- never on the bar it occurred (which Pine itself also can't do,
    since a pivot needs bars on both sides to exist before it's a pivot).
  - FVG and sweep checks use only the current and PAST bars' OHLC.
  - The stop/target distance for a held position is captured at the exact
    trigger bar and forward-filled (see judas_sweep_killzone.py's note on
    why a one-bar EVENT's risk framing shouldn't be recomputed from
    unrelated later price action during a same-direction re-entry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "ICT Sniper (ported from PineScript)"

SENSITIVITY = 5          # swing pivot confirmation window (each side)
MIN_FVG_PIPS_EQUIV = 0.00008   # ~0.8 pip on a 5-digit FX quote -- CALIBRATE this per instrument;
                                # the original compared against syminfo.mintick-normalized units,
                                # which this port can't know in advance. Too high a value here
                                # means "no FVG ever qualifies" on quieter instruments/timeframes.
ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.5
STOP_BUFFER_EQUIV = 0.00005
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


def _confirmed_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    cond = pd.Series(True, index=high.index)
    for k in range(1, left + 1):
        cond &= high > high.shift(k)
    for k in range(1, right + 1):
        cond &= high > high.shift(-k)
    pivot_at_occurrence = high.where(cond)
    # Only known once `right` further bars have closed -- shift forward so
    # it becomes available on the bar it's actually confirmed on.
    return pivot_at_occurrence.shift(right)


def _confirmed_pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    cond = pd.Series(True, index=low.index)
    for k in range(1, left + 1):
        cond &= low < low.shift(k)
    for k in range(1, right + 1):
        cond &= low < low.shift(-k)
    pivot_at_occurrence = low.where(cond)
    return pivot_at_occurrence.shift(right)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    atr = _atr(df, ATR_PERIOD)

    pivot_high = _confirmed_pivot_high(high, SENSITIVITY, SENSITIVITY).ffill()
    pivot_low = _confirmed_pivot_low(low, SENSITIVITY, SENSITIVITY).ffill()

    swept_bsl = (high > pivot_high) & (close < pivot_high)   # buy-side liquidity swept
    swept_ssl = (low < pivot_low) & (close > pivot_low)      # sell-side liquidity swept

    # The original tracks `last_sweep` as a PERSISTENT state variable (Pine
    # `var string last_sweep`) that holds "BSL"/"SSL" across many
    # subsequent bars until overwritten by a new sweep -- it is NOT a
    # same-bar-only flag. An FVG confirming several bars after the sweep
    # still counts. Forward-filling reproduces that persistence correctly.
    last_sweep = pd.Series(np.nan, index=df.index, dtype=object)
    last_sweep[swept_bsl] = "BSL"
    last_sweep[swept_ssl] = "SSL"
    last_sweep = last_sweep.ffill()

    # NOTE ON FIDELITY: the original's "FVG" check is `low < high[2]` /
    # `high > low[2]` -- this tests whether the current bar's range
    # OVERLAPS the bar-from-2-ago's range, not a true no-overlap gap (a
    # textbook Fair Value Gap would require `low > high[2]` with NO
    # overlap). That's a looseness in the SOURCE script, not a porting
    # error -- it's kept as-is rather than "fixed" into a stricter
    # definition the original never actually implemented.
    body = (close - open_).abs()
    fvg_top = high.shift(2).where(low < high.shift(2))
    fvg_bot = low.shift(2).where(high > low.shift(2))
    fvg_size = fvg_top - fvg_bot   # only defined where BOTH conditions above hold

    is_fvg_bull = (close > open_) & (body > atr * DISPLACEMENT_ATR_MULT) & (fvg_size > MIN_FVG_PIPS_EQUIV)
    is_fvg_bear = (close < open_) & (body > atr * DISPLACEMENT_ATR_MULT) & (fvg_size > MIN_FVG_PIPS_EQUIV)

    # Carry the most recent FVG zone forward, same as the original's
    # `var float bull_fvg_top/bot` persistence (both sides share the same
    # fvg_top/fvg_bot pair in the source, captured at whichever bar
    # confirmed that side's FVG).
    bull_fvg_top = fvg_top.where(is_fvg_bull).ffill()
    bull_fvg_bot = fvg_bot.where(is_fvg_bull).ffill()
    bear_fvg_top = fvg_top.where(is_fvg_bear).ffill()
    bear_fvg_bot = fvg_bot.where(is_fvg_bear).ffill()

    cond_buy = (last_sweep == "SSL") & is_fvg_bull & (low >= bull_fvg_bot) & (low <= bull_fvg_top)
    cond_sell = (last_sweep == "BSL") & is_fvg_bear & (high <= bear_fvg_top) & (high >= bear_fvg_bot)

    sl_dist_buy = (close - bull_fvg_bot) + STOP_BUFFER_EQUIV
    sl_dist_sell = (bear_fvg_top - close) + STOP_BUFFER_EQUIV

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[cond_buy] = 1.0
    raw[cond_sell] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    raw_sl = pd.Series(np.nan, index=df.index)
    raw_sl[cond_buy] = sl_dist_buy[cond_buy]
    raw_sl[cond_sell] = sl_dist_sell[cond_sell]
    stop_loss_distance = raw_sl.ffill().fillna(atr * DISPLACEMENT_ATR_MULT)
    take_profit_distance = stop_loss_distance * RR_RATIO

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance

    return signal
