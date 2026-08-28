"""
Ported from: smc-quant-engine.pine ("SMC Quant Engine", QuantAlgo /
Alexandre Albert Ndour). Original combined a 4H EMA bias filter with
confirmed-pivot BOS/liquidity-sweep detection, a loose FVG/order-block
check, an ADX trend filter, a session window, and a volume spike, scored
out of 8 with a minimum-score entry threshold.

STRATEGY LOGIC (unchanged from the original):
  HTF bias   : 4H EMA(50) vs EMA(200), using the LAST FULLY CLOSED 4H bar
               (see the lookahead note below -- ported carefully, since
               this exact kind of HTF filter is the single most common
               source of a real lookahead bug in this app's own history)
  BOS/Sweep  : confirmed 5-bar swing pivots; BOS = close breaks the
               pivot and the prior bar hadn't yet; Sweep = a wick beyond
               the pivot that closes back on the other side
  FVG        : 3-bar gap (low > high[2] / high < low[2] -- a proper
               no-overlap gap, this script's version is NOT the loose
               one smc-quant-engine's sibling scripts use)
  Volume     : current volume > 1.2x its 20-bar average
  ADX filter : ADX(14) > 20 required for ANY entry (trend must be real)
  Session    : 07:00-21:00 UTC (timestamp-hour proxy, see note)
  Score      : HTF align (1) + sweep (2) + BOS (2) + FVG (1) + OB (1) +
               volume (1) = up to 8; enter at score >= 5
  Stop loss  : ATR(14) * 1.5
  Take profit: TP1 at stop distance * 1.5 (this port targets TP1 only --
               the original's TP2/partial-scale-out at 50% qty on TP1
               can't be expressed by T58's single-target signal model)

WHAT DIDN'T CARRY OVER, AND WHY:
  - The daily-drawdown-based `dd_ok` gate: account-state, can't be seen
    from inside generate_signals(df). Use RiskConfig.daily_loss_limit_pct
    to match the original's max_dd instead.
  - Partial take-profit scaling (50% out at TP1, rest to TP2): T58's
    signal model places one stop and one target per trade. This port
    uses TP1 (the original's own first, most conservative target) as THE
    target, rather than approximating a partial-exit scheme it can't
    faithfully express.

NO LOOKAHEAD (the important one to get right here):
  The 4H HTF bias is built with `resample(..., label="left", closed="left")`
  then `.shift(1)` BEFORE reindexing onto the base 5-minute timeframe, so
  every 5-minute bar only ever sees the previous, FULLY CLOSED 4H bar's
  EMA values -- never the 4H bar it's currently inside of. This is the
  exact class of bug this app's own docs call out as its most common real
  lookahead source (a resampled bar is labeled by its START time, so a
  naive `htf.index < timestamp` filter still leaks up to 4 hours of
  not-yet-closed price action into every bar that isn't exactly on a 4H
  boundary). Swing pivots are confirmed-and-shifted the same way as in
  ict_fvg_liquidity_sweep.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "SMC Quant Engine (ported from PineScript)"

HTF_FREQ = "4h"
HTF_EMA_FAST, HTF_EMA_SLOW = 50, 200

PIVOT_BARS = 5
ATR_PERIOD = 14
ADX_PERIOD = 14
VOL_LOOKBACK = 20
VOL_MULT = 1.2
SESSION_START_UTC, SESSION_END_UTC = 7, 21

SL_ATR_MULT = 1.5
RR_TP1 = 1.5
MIN_SCORE = 5


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


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def _confirmed_pivot(series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
    cond = pd.Series(True, index=series.index)
    for k in range(1, left + 1):
        cond &= (series > series.shift(k)) if is_high else (series < series.shift(k))
    for k in range(1, right + 1):
        cond &= (series > series.shift(-k)) if is_high else (series < series.shift(-k))
    return series.where(cond).shift(right)


def _htf_bias(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Last-fully-closed-4H-bar EMA bias, safely reindexed onto the base timeframe."""
    ts = pd.to_datetime(df["timestamp"])
    htf_close = df["close"].set_axis(ts).resample(HTF_FREQ, label="left", closed="left").last()
    htf_ema_fast = _ema(htf_close, HTF_EMA_FAST).shift(1)   # last COMPLETED 4H bar only
    htf_ema_slow = _ema(htf_close, HTF_EMA_SLOW).shift(1)
    base_fast = htf_ema_fast.reindex(ts, method="ffill").set_axis(df.index)
    base_slow = htf_ema_slow.reindex(ts, method="ffill").set_axis(df.index)
    return base_fast, base_slow


def generate_signals(df: pd.DataFrame) -> pd.Series:
    close, open_, high, low, volume = df["close"], df["open"], df["high"], df["low"], df["volume"]

    atr = _atr(df, ATR_PERIOD)
    adx = _adx(df, ADX_PERIOD)

    htf_ema_fast, htf_ema_slow = _htf_bias(df)
    htf_bull = htf_ema_fast > htf_ema_slow
    htf_bear = htf_ema_fast < htf_ema_slow

    pivot_high = _confirmed_pivot(high, PIVOT_BARS, PIVOT_BARS, is_high=True).ffill()
    pivot_low = _confirmed_pivot(low, PIVOT_BARS, PIVOT_BARS, is_high=False).ffill()

    bull_bos = (close > pivot_high) & (close.shift(1) <= pivot_high.shift(1))
    bear_bos = (close < pivot_low) & (close.shift(1) >= pivot_low.shift(1))
    bull_sweep = (low < pivot_low) & (close > pivot_low)
    bear_sweep = (high > pivot_high) & (close < pivot_high)

    bull_fvg = (low > high.shift(2)) & (close > open_)
    bear_fvg = (high < low.shift(2)) & (close < open_)

    body = (close - open_).abs()
    displacement_up = (close > high.shift(1)) & (body > atr) & (close > open_)
    displacement_down = (close < low.shift(1)) & (body > atr) & (close < open_)
    bull_ob = displacement_up & (close.shift(1) < open_.shift(1))
    bear_ob = displacement_down & (close.shift(1) > open_.shift(1))

    vol_avg = volume.rolling(VOL_LOOKBACK).mean()
    vol_spike = volume > (vol_avg * VOL_MULT)
    adx_trend = adx > 20

    hour_utc = pd.to_datetime(df["timestamp"]).dt.hour
    session_ok = (hour_utc >= SESSION_START_UTC) & (hour_utc <= SESSION_END_UTC)

    score_long = (
        htf_bull.astype(int)
        + bull_sweep.astype(int) * 2
        + bull_bos.astype(int) * 2
        + bull_fvg.astype(int)
        + bull_ob.astype(int)
        + vol_spike.astype(int)
    )
    score_short = (
        htf_bear.astype(int)
        + bear_sweep.astype(int) * 2
        + bear_bos.astype(int) * 2
        + bear_fvg.astype(int)
        + bear_ob.astype(int)
        + vol_spike.astype(int)
    )

    long_trigger = (score_long >= MIN_SCORE) & session_ok & adx_trend
    short_trigger = (score_short >= MIN_SCORE) & session_ok & adx_trend

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[long_trigger] = 1.0
    raw[short_trigger] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    stop_loss_distance = atr * SL_ATR_MULT
    take_profit_distance = stop_loss_distance * RR_TP1

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance

    return signal
