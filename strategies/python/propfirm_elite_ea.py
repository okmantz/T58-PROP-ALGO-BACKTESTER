"""
Ported from: propfirm-elite-ea.mq5 ("PropFirm Elite EA / V8 Elite SMC",
QuantAlgo / Alexandre Albert Ndour). Original was a multi-confirmation
scoring system: BOS + triple-EMA trend + a simplified Fair Value Gap +
EMA-crossover momentum + RSI + volume spike, each worth points, entering
once either side's score crosses a threshold.

STRATEGY LOGIC (unchanged from the original):
  Trend       : EMA(21) vs EMA(50) vs EMA(200) fully stacked  = 2 points
  BOS         : close breaks the prior N-bar high/low          = 3 points
  FVG         : simplified 3-bar gap (low[1] > high[3], etc.)  = 2 points
  EMA cross   : EMA21/EMA50 crossed in the last bar (momentum) = 1 point
  RSI         : RSI(14) inside a "not extreme" band            = 1 point
  Volume      : current volume > 1.3x the recent average       = 1 point
              (volume confirmation adds to BOTH sides, matching the
               original's slightly odd "volConfirm adds to both bullScore
               AND bearScore" -- kept as-is rather than silently "fixing"
               the source strategy's own logic)
  Entry       : first side to reach >= 5 points (bull or bear), MINIMUM SCORE
  Stop loss   : ATR(10) * 1.2
  Take profit : stop distance * 2.0 (Risk:Reward, matches InpRR_Ratio)
  Breakeven   : once profit >= 1.0x the stop distance, move stop to
                entry (+/- a few points)
  Trailing    : once past the breakeven trigger, trail by ATR(10) * 0.8

WHAT DIDN'T CARRY OVER, AND WHY:
  - Daily loss/profit limits, max-trades-per-day, and the multi-session
    time filter: same reasoning as propfirm_reactive_bos.py -- these are
    account-state or pure risk-desk concerns. Set RiskConfig.daily_loss_
    limit_pct and RiskConfig.max_trades_per_day to match InpDailyLossMax /
    InpMaxTrades instead of trying to re-implement them inside the
    strategy (they cannot see today's running P&L from inside
    generate_signals -- see the python.py adapter's docstring).
  - The London/NY session filter IS just a time-of-day mask (not account
    state), so it WAS kept, using the timestamp's own hour -- adjust
    SESSION_1/2 below if your data isn't already in the server's timezone.

NO LOOKAHEAD: BOS uses the prior N bars only (`.shift(1)` before the
rolling window); the "EMA crossover in the last bar" check compares this
bar's EMA order against the PREVIOUS bar's EMA order, never a future one;
volume/RSI/ATR are all standard trailing indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "PropFirm Elite EA (ported from MQL5)"

EMA_FAST, EMA_SLOW, EMA_TREND = 21, 50, 200
BOS_PERIOD = 12
ATR_PERIOD = 10
RSI_PERIOD = 7
RSI_OVERBOUGHT, RSI_OVERSOLD = 75, 25
VOLUME_MULT = 1.3
VOLUME_LOOKBACK = 3

SL_ATR_MULT = 1.2
RR_RATIO = 2.0
BE_TRIGGER_R = 1.0          # move to breakeven once profit >= 1.0x stop distance
TRAIL_ATR_MULT = 0.8

MIN_SCORE = 5

SESSION_1_START, SESSION_1_END = 8, 11    # London
SESSION_2_START, SESSION_2_END = 14, 17   # New York


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


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)
    ema_trend = _ema(close, EMA_TREND)
    atr = _atr(df, ATR_PERIOD)
    rsi = _rsi(close, RSI_PERIOD)

    bos_high = high.shift(1).rolling(BOS_PERIOD).max()
    bos_low = low.shift(1).rolling(BOS_PERIOD).min()
    bull_bos = close.shift(1) > bos_high.shift(1)
    bear_bos = close.shift(1) < bos_low.shift(1)

    # Simplified FVG, exactly as the original: a 2-bar-old gap.
    bull_fvg = low.shift(1) > high.shift(3)
    bear_fvg = high.shift(1) < low.shift(3)

    bull_trend = (ema_fast > ema_slow) & (ema_slow > ema_trend)
    bear_trend = (ema_fast < ema_slow) & (ema_slow < ema_trend)

    bull_cross = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    bear_cross = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    rsi_buy_ok = (rsi > RSI_OVERSOLD) & (rsi < 65)
    rsi_sell_ok = (rsi < RSI_OVERBOUGHT) & (rsi > 35)

    avg_vol = volume.rolling(VOLUME_LOOKBACK).mean()
    vol_confirm = volume > avg_vol * VOLUME_MULT

    bull_score = (
        bull_bos.astype(int) * 3
        + bull_trend.astype(int) * 2
        + bull_fvg.astype(int) * 2
        + bull_cross.astype(int) * 1
        + rsi_buy_ok.astype(int) * 1
        + vol_confirm.astype(int) * 1  # matches original: volume adds to BOTH sides
    )
    bear_score = (
        bear_bos.astype(int) * 3
        + bear_trend.astype(int) * 2
        + bear_fvg.astype(int) * 2
        + bear_cross.astype(int) * 1
        + rsi_sell_ok.astype(int) * 1
        + vol_confirm.astype(int) * 1
    )

    hour = pd.to_datetime(df["timestamp"]).dt.hour
    in_session = ((hour >= SESSION_1_START) & (hour < SESSION_1_END)) | (
        (hour >= SESSION_2_START) & (hour < SESSION_2_END)
    )

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
    signal.attrs["breakeven_trigger_r"] = BE_TRIGGER_R

    return signal
