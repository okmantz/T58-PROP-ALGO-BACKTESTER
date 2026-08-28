"""
Ported from: hedge-fund-smc-ea.mq5 ("Hedge Fund SMC EA", QuantAlgo /
Alexandre Albert Ndour). Original was an MT5 EA using iMA/iATR indicator
handles -- not expressible in T58's MQL5 parser subset (ATR isn't a
supported indicator there), so it's ported here as a Python strategy
instead.

STRATEGY LOGIC (unchanged from the original):
  Trend filter : EMA(50) vs EMA(200)
  Entry        : bullish trend (EMA50 > EMA200) AND close breaks above the
                 PREVIOUS bar's high  -> long
                 bearish trend (EMA50 < EMA200) AND close breaks below the
                 previous bar's low   -> short
  Stop loss    : ATR(14) * 2.0
  Take profit  : ATR(14) * 4.0   (TP2_Mult in the original; TP1 was unused
                 by the original's actual order-send call, so this keeps
                 the ONE target the EA actually placed)
  Trailing     : ATR(14) distance, trailing the stop once in profit
                 (matches the original's ManageTrailingStop -- always-on
                 trailing rather than breakeven-then-trail, same as source)

WHAT DIDN'T CARRY OVER, AND WHY:
  Nothing state-dependent here -- this was the simplest of the batch, so
  it's a faithful, complete port. Position sizing is handled by T58's own
  RiskConfig (risk % per trade) rather than re-implementing the EA's
  CalculateLotSize by hand -- same risk-per-trade concept, T58's engine
  owns the arithmetic.

NO LOOKAHEAD: every value used to decide bar i's signal is computed from
bars <= i (EMAs/ATR are trailing indicators; the "previous bar" high/low
is genuinely the prior, already-closed bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "Hedge Fund SMC EA (ported from MQL5)"

EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
SL_ATR_MULT = 2.0
TP_ATR_MULT = 4.0
TRAIL_ATR_MULT = 1.0


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
    ema_fast = _ema(df["close"], EMA_FAST)
    ema_slow = _ema(df["close"], EMA_SLOW)
    atr = _atr(df, ATR_PERIOD)

    bullish_trend = ema_fast > ema_slow
    bearish_trend = ema_fast < ema_slow

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    long_trigger = bullish_trend & (df["close"] > prev_high)
    short_trigger = bearish_trend & (df["close"] < prev_low)

    # Build a held long/flat/short signal: a trigger sets the position,
    # and it's carried forward (the engine's own SL/TP/trailing closes it)
    # until an opposite trigger fires. This mirrors the original EA, which
    # only ever opens a NEW trade when flat (`if(PositionSelect(...)) {
    # ManageTrailingStop(); return; }`) and otherwise lets the stop/target
    # do the work.
    raw = pd.Series(0, index=df.index, dtype=float)
    raw[long_trigger] = 1.0
    raw[short_trigger] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    # IMPORTANT: the engine re-enters on ANY bar where it is flat and the
    # (held/ffilled) signal is nonzero -- not only on the original trigger
    # bar. If a position gets stopped out while the signal is still held,
    # the very next bar can trigger a fresh entry. So the stop/target/
    # trailing distances must be supplied for EVERY bar (derived from that
    # bar's own ATR), not just the original trigger bars, or a same-
    # direction re-entry after a stop-out would silently fall back to the
    # engine's generic 1%-of-price stop instead of the real ATR-based one.
    stop_loss_distance = atr * SL_ATR_MULT
    take_profit_distance = atr * TP_ATR_MULT
    trailing_stop_distance = atr * TRAIL_ATR_MULT

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance
    signal.attrs["trailing_stop_distance"] = trailing_stop_distance

    return signal
