"""
Ported from: ultimate-propfirm-strategy.pine ("Ultimate PropFirm Strategy",
QuantAlgo / Alexandre Albert Ndour).

A NOTE ON WHY THIS ONE FILE STANDS IN FOR THREE PINE SCRIPTS: propfirm-
elite-strategy.pine, propfirm-ultra-strategy.pine, and this one are all
the SAME underlying "EMA stack + BOS + FVG + OB + volume score" engine
by the same author, at different iteration stages -- elite/ultra add an
adaptive score threshold, regime detection (compression/news-spike/high-
vol classification), Break-Block/CHOCH/Inducement variants, and a
consecutive-loss-triggered risk-reduction mode on top of this same core.
Every one of those additions is ACCOUNT-STATE dependent (today's trade
outcomes, a running win/loss streak) and literally cannot be seen from
inside a stateless generate_signals(df) call (see the python.py adapter's
own docstring on this exact limitation). Porting elite/ultra "faithfully"
would mean porting the entry logic below AND silently dropping their
adaptive/recovery layer anyway -- so rather than ship three near-
duplicate files that quietly can't do what their names promise, this is
the one, clean, fully-expressible version of the shared engine. If you
want the more elaborate scoring (adaptive threshold, CHOCH, breaker
blocks, inducement) added on top of this base, say so and it can be
built out explicitly -- just know going in that the recovery-mode risk
scaling those versions advertise isn't portable to this architecture,
same as the daily-loss/max-trades gates below.

STRATEGY LOGIC (unchanged from the original):
  Trend      : EMA(9) > EMA(21) > EMA(50) > EMA(200), fully stacked
  BOS        : close crosses above/below the highest-high/lowest-low of
               the prior BOS_LEN bars
  FVG        : 3-bar gap (low[1] > high[3] / high[1] < low[3])
  OB         : a 2-candle reversal pattern (down candle then up candle
               that closes above the down candle's high, or the mirror)
  Volume     : current volume > 1.2x its 5-bar average
  Score      : BOS (3) + EMA stack (2) + FVG (2) + OB (2) + volume (1)
               = up to 10; enter at score >= 5
  Stop loss  : ATR(14) * 1.5
  Take profit: stop distance * 2.0 (Risk:Reward, matches the source)
  Trailing   : ATR(14) * 0.7 (the original's trail_offset)
  Session    : configurable trading-hours window (default 07:00-17:00,
               timestamp-hour proxy -- see note below)

WHAT DIDN'T CARRY OVER, AND WHY:
  - Daily loss limit / max trades per day: account-state, can't be seen
    from inside generate_signals(df). Use RiskConfig.daily_loss_limit_pct
    and RiskConfig.max_trades_per_day to match InpDailyLossMax/
    maxTradesDay instead -- T58 enforces these at the engine level,
    which is the correct place for them.
  - The session filter IS just a time-of-day mask (not account state), so
    it WAS kept -- but the original's `input.session("0700-1700")` is
    relative to the chart's own timezone, which this port can't know in
    advance. Adjust SESSION_START_HOUR/SESSION_END_HOUR below to match
    your data.

NO LOOKAHEAD: BOS uses `high[1]`/`low[1]` (prior, already-closed bars)
inside the rolling window before comparing to the current close; the FVG
and OB checks reference only bars 1-3 back; EMAs/ATR are standard
trailing indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "PropFirm Score Engine (ported from PineScript, ultimate/elite/ultra family)"

EMA_1, EMA_2, EMA_3, EMA_4 = 9, 21, 50, 200
BOS_LEN = 10
ATR_PERIOD = 14
VOL_LOOKBACK = 5
VOL_MULT = 1.2

SL_ATR_MULT = 1.5
RR_RATIO = 2.0
TRAIL_ATR_MULT = 0.7
MIN_SCORE = 5

SESSION_START_HOUR = 7   # inclusive
SESSION_END_HOUR = 17    # exclusive


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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
    close, open_, high, low, volume = df["close"], df["open"], df["high"], df["low"], df["volume"]

    ema1, ema2, ema3, ema4 = (_ema(close, p) for p in (EMA_1, EMA_2, EMA_3, EMA_4))
    atr = _atr(df, ATR_PERIOD)

    high_bos = high.shift(1).rolling(BOS_LEN).max()
    low_bos = low.shift(1).rolling(BOS_LEN).min()
    bull_bos = (close > high_bos) & (close.shift(1) <= high_bos.shift(1))
    bear_bos = (close < low_bos) & (close.shift(1) >= low_bos.shift(1))

    bull_fvg = low.shift(1) > high.shift(3)
    bear_fvg = high.shift(1) < low.shift(3)

    bull_ob = (close.shift(2) < open_.shift(2)) & (close.shift(1) > open_.shift(1)) & (close.shift(1) > high.shift(2))
    bear_ob = (close.shift(2) > open_.shift(2)) & (close.shift(1) < open_.shift(1)) & (close.shift(1) < low.shift(2))

    vol_avg = volume.rolling(VOL_LOOKBACK).mean()
    vol_spike = volume > (vol_avg * VOL_MULT)

    bull_stack = (ema1 > ema2) & (ema2 > ema3) & (ema3 > ema4)
    bear_stack = (ema1 < ema2) & (ema2 < ema3) & (ema3 < ema4)

    bull_score = (
        bull_bos.astype(int) * 3
        + bull_stack.astype(int) * 2
        + bull_fvg.astype(int) * 2
        + bull_ob.astype(int) * 2
        + vol_spike.astype(int)
    )
    bear_score = (
        bear_bos.astype(int) * 3
        + bear_stack.astype(int) * 2
        + bear_fvg.astype(int) * 2
        + bear_ob.astype(int) * 2
        + vol_spike.astype(int)
    )

    hour = pd.to_datetime(df["timestamp"]).dt.hour
    in_session = (hour >= SESSION_START_HOUR) & (hour < SESSION_END_HOUR)

    long_trigger = (bull_score >= MIN_SCORE) & in_session
    short_trigger = (bear_score >= MIN_SCORE) & in_session

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[long_trigger] = 1.0
    raw[short_trigger] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    stop_loss_distance = atr * SL_ATR_MULT
    take_profit_distance = stop_loss_distance * RR_RATIO
    trailing_stop_distance = atr * TRAIL_ATR_MULT

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance
    signal.attrs["trailing_stop_distance"] = trailing_stop_distance

    return signal
