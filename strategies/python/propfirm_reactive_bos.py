"""
Ported from: propfirm-reactive-ea.mq5 ("PropFirm Reactive EA", QuantAlgo /
Alexandre Albert Ndour). Original combined an EMA(200) trend filter with a
Break-of-Structure (N-bar high/low breakout) entry and an ATR-based
stop/target, plus prop-firm daily loss/target limits and a session filter.

STRATEGY LOGIC (unchanged from the original):
  Trend filter : close vs EMA(200)
  BOS entry    : bullish trend AND close breaks above the highest high of
                 the PRIOR `BOS_PERIOD` bars (excluding the current bar)
                 -> long; bearish trend AND close breaks below the lowest
                 low of the prior BOS_PERIOD bars -> short
  Stop loss    : ATR(14) * 1.5
  Take profit  : stop distance * 1.5 (Risk:Reward 1:1.5, matches original)
  Session      : only take NEW entries between InpStartHour-InpEndHour
                 (server-hour proxy -- see note below)

WHAT DIDN'T CARRY OVER, AND WHY:
  - Daily loss limit / daily profit target / per-day trade cap: these are
    ACCOUNT-STATE checks (today's running P&L), and T58's Python strategy
    interface calls generate_signals(df) once, statelessly, over the whole
    dataset before any P&L exists -- so a strategy literally cannot see
    "how much have I made/lost today" (see the python.py adapter's own
    docstring on this). T58 enforces the equivalent protection at the
    ENGINE level instead: set RiskConfig.daily_loss_limit_pct and the
    Prop Rules tab's daily-loss field to match the original's $45/day
    limit, and set RiskConfig.max_trades_per_day if you want a trade cap.
    That is the correct place for this, not inside the strategy.
  - "Server hour" session filter: the original's hours are relative to the
    broker's server clock, which isn't something this port can know. The
    filter below uses the timestamp's own hour-of-day as a stand-in --
    adjust SESSION_START_HOUR/SESSION_END_HOUR to match your data's
    timezone/session before trusting the results.

NO LOOKAHEAD: the BOS breakout level is the highest/lowest of bars
strictly BEFORE the current one (`.shift(1)` before the rolling window),
so a breakout is only ever measured against already-closed price action.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "PropFirm Reactive EA (ported from MQL5)"

EMA_TREND_PERIOD = 200
BOS_PERIOD = 15
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
RR_RATIO = 1.5

SESSION_START_HOUR = 9   # inclusive, matches InpStartHour in the original
SESSION_END_HOUR = 18    # exclusive, matches InpEndHour in the original


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
    ema_trend = _ema(df["close"], EMA_TREND_PERIOD)
    atr = _atr(df, ATR_PERIOD)

    bos_high = df["high"].shift(1).rolling(BOS_PERIOD).max()
    bos_low = df["low"].shift(1).rolling(BOS_PERIOD).min()

    bullish_trend = df["close"] > ema_trend
    bearish_trend = df["close"] < ema_trend

    hour = pd.to_datetime(df["timestamp"]).dt.hour
    in_session = (hour >= SESSION_START_HOUR) & (hour < SESSION_END_HOUR)

    long_trigger = bullish_trend & (df["close"] > bos_high) & in_session
    short_trigger = bearish_trend & (df["close"] < bos_low) & in_session

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[long_trigger] = 1.0
    raw[short_trigger] = -1.0
    signal = raw.replace(0.0, np.nan).ffill().fillna(0.0)

    # Supplied for every bar (see hedge_fund_smc_ea.py's note on why this
    # must not be limited to the original trigger bars only).
    stop_loss_distance = atr * SL_ATR_MULT
    take_profit_distance = stop_loss_distance * RR_RATIO

    signal.attrs["stop_loss_distance"] = stop_loss_distance
    signal.attrs["take_profit_distance"] = take_profit_distance

    return signal
