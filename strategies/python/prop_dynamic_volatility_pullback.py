import numpy as np
import pandas as pd

STRATEGY_NAME = "Prop Dynamic Volatility Pullback"
STOP_LOSS_PIPS = 25
TAKE_PROFIT_PIPS = 45


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Prop-oriented trend-pullback engine with dynamic ATR risk scaling.

    Only uses pandas and numpy; no lookahead.
    """
    signals = pd.Series(0, index=df.index, dtype=int)

    # 1. Trend & Regime Filters (EMAs)
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_mid = df["close"].ewm(span=50, adjust=False).mean()
    ema_slow = df["close"].ewm(span=200, adjust=False).mean()

    # 2. Volatility Calculation (ATR 14)
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat(
        [high_low, high_close_prev, low_close_prev], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window=14).mean().bfill()

    # 3. Regime Definitions
    bull_regime = (ema_mid > ema_slow) & (df["close"] > ema_slow)
    bear_regime = (ema_mid < ema_slow) & (df["close"] < ema_slow)

    # 4. Setup & Trigger Conditions (Displacement + Pullback Reversal)
    # Long setup: Price dipped near or below 20 EMA, then closes strongly above it in an uptrend
    long_pullback = (
        (df["low"].shift(1) <= ema_fast.shift(1))
        & (df["close"].shift(1) >= ema_mid.shift(1))
        & (df["close"] > ema_fast)
        & (df["close"] > df["open"])
    )

    # Short setup: Price pushed near or above 20 EMA, then closes strongly below it in a downtrend
    short_pullback = (
        (df["high"].shift(1) >= ema_fast.shift(1))
        & (df["close"].shift(1) <= ema_mid.shift(1))
        & (df["close"] < ema_fast)
        & (df["close"] < df["open"])
    )

    # Event-based entry trigger (avoid persistent holds)
    long_trigger = bull_regime & long_pullback
    short_trigger = bear_regime & short_pullback

    # 5. Populate Signals (-1 = Short, 0 = Flat, 1 = Long)
    signals[long_trigger] = 1
    signals[short_trigger] = -1

    # 6. Dynamic Risk Settings (Volatility-normalized)
    # Stop Loss: 1.5x ATR, Take Profit: 2.5x ATR (1:1.67 Risk/Reward)
    sl_distance = (1.5 * atr).values
    tp_distance = (2.5 * atr).values

    signals.attrs["stop_loss_distance"] = sl_distance
    signals.attrs["take_profit_distance"] = tp_distance
    signals.attrs["breakeven_trigger_r"] = 1.0  # Move SL to BE at +1.0R

    return signals
