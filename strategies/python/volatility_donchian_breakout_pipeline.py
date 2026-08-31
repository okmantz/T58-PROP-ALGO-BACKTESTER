"""
NEW STRATEGY -- Volatility-Filtered Donchian Breakout (built from scratch,
2026-08-30, in response to the "all 9 disproven" pipeline review).

This is a genuinely different mechanism from every strategy in the prior
batch (which were all pullback/sweep/scoring variants of trend-following
or ICT/SMC logic). This one trades breakouts of recent price extremes,
filtered by an expanding-volatility regime, using only price/ATR-derived
measures that scale automatically with whatever instrument is loaded --
there is no fixed pip/price constant anywhere in this file (the exact bug
class that broke MIN_FVG_PIPS_EQUIV in ict_fvg_liquidity_sweep.py).

STRATEGY LOGIC:
  Channel     : rolling N-bar Donchian channel (highest high / lowest low
                of the PRIOR N bars -- the current, still-forming bar is
                excluded from its own breakout threshold)
  Entry       : close breaks above the upper channel (long) or below the
                lower channel (short) -- AND ATR(14) is currently above
                its own 50-bar rolling average, i.e. volatility is
                expanding, not contracting. This skips breakouts fired
                during quiet/choppy regimes, which is where most Donchian
                breakout systems bleed out on false starts.
  Stop        : ATR(14) x STOP_ATR_MULT, captured dynamically per trade
  Target      : stop distance x RR_RATIO
  Trail       : once in profit, trails at ATR x TRAIL_ATR_MULT
  Breakeven   : moves to breakeven at +1R (BREAKEVEN_TRIGGER_R)
  Exit        : an opposite-direction breakout reverses the position
                (T58's standard long/flat/short signal model), or the
                engine's own stop/target/trail closes it first

WHY THIS IS INSTRUMENT-AGNOSTIC:
  Every threshold here is either a lookback window (bars, not price) or an
  ATR multiple. There is no equivalent of the old pip_size/absolute-price
  problem -- this file needs zero recalibration to run sanely on AAPL,
  XAUUSD, EURUSD, or anything else. Position sizing and risk-per-trade are
  still handled by T58's own RiskConfig, exactly as with every other
  strategy in this library.

NO LOOKAHEAD:
  - Both channel boundaries are shifted forward by 1 bar before comparison
    against the current close, so a breakout is only ever measured against
    bars that had FULLY CLOSED before the current bar.
  - ATR and its rolling average use only current/past bars (standard
    trailing rolling windows -- no centering).
  - Stop/target/trail distances are captured at the exact entry bar and
    forward-filled for the life of the trade, same convention used
    elsewhere in this library (see ict_fvg_liquidity_sweep.py's note on
    why a one-bar event's risk framing shouldn't be recomputed later).

WHAT THIS DOESN'T CLAIM:
  This has NOT been run through T58's actual Full Pipeline -- no GA
  search, no walk-forward folds, no Monte Carlo, no significance gate.
  Those need the real engine and your real data, which weren't available
  in the session that wrote this file. Treat every constant below as a
  reasonable starting point for the GA to refine, not a validated setting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "Volatility-Filtered Donchian Breakout"

DONCHIAN_PERIOD = 17        # bars in the breakout channel
ATR_PERIOD = 42
VOL_FILTER_LOOKBACK = 132    # bars in the ATR-regime rolling average
STOP_ATR_MULT = 1.289338
RR_RATIO = 3.273576
TRAIL_ATR_MULT = 1.21484
BREAKEVEN_TRIGGER_R = 1.101175


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
    high, low, close = df["high"], df["low"], df["close"]

    atr = _atr(df, ATR_PERIOD)
    atr_avg = atr.rolling(VOL_FILTER_LOOKBACK, min_periods=VOL_FILTER_LOOKBACK).mean()
    expanding_vol = atr > atr_avg

    # Prior N bars only -- shift(1) excludes the current, still-forming bar
    # from its own breakout threshold (the #1 lookahead trap for channel
    # systems: comparing today's close against a channel that includes
    # today's own high/low).
    upper_channel = high.rolling(DONCHIAN_PERIOD).max().shift(1)
    lower_channel = low.rolling(DONCHIAN_PERIOD).min().shift(1)

    breakout_long = (close > upper_channel) & expanding_vol
    breakout_short = (close < lower_channel) & expanding_vol

    raw = pd.Series(0.0, index=df.index)
    raw[breakout_long] = 1.0
    raw[breakout_short] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    stop_dist = atr * STOP_ATR_MULT
    raw_sl = pd.Series(np.nan, index=df.index)
    raw_sl[breakout_long] = stop_dist[breakout_long]
    raw_sl[breakout_short] = stop_dist[breakout_short]
    stop_loss_distance = raw_sl.ffill().fillna(stop_dist)
    take_profit_distance = stop_loss_distance * RR_RATIO
    trailing_stop_distance = atr * TRAIL_ATR_MULT

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance
    signal.attrs["trailing_stop_distance"] = trailing_stop_distance
    signal.attrs["breakeven_trigger_r"] = BREAKEVEN_TRIGGER_R

    return signal
