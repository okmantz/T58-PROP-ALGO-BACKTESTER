import numpy as np
import pandas as pd


STRATEGY_NAME = "Prop ATR Breakout Continuation"

BREAKOUT_LOOKBACK = 20
FAST_EMA_LENGTH = 21
SLOW_EMA_LENGTH = 55
ATR_LENGTH = 14
MAX_ENTRIES_PER_DAY = 2
MAX_HOLD_BARS = 64


def generate_signals(df: pd.DataFrame) -> pd.Series:
    n = len(df)

    if n == 0:
        signals = pd.Series(dtype="int64", index=df.index, name="signal")
        signals.attrs["stop_loss_distance"] = np.array([], dtype=float)
        signals.attrs["take_profit_distance"] = np.array([], dtype=float)
        signals.attrs["trailing_stop_distance"] = np.array([], dtype=float)
        signals.attrs["breakeven_trigger_r"] = 1.0
        return signals

    open_price = pd.to_numeric(df["open"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        span=ATR_LENGTH,
        adjust=False,
        min_periods=ATR_LENGTH,
    ).mean()

    fast_ema = close.ewm(
        span=FAST_EMA_LENGTH,
        adjust=False,
        min_periods=FAST_EMA_LENGTH,
    ).mean()

    slow_ema = close.ewm(
        span=SLOW_EMA_LENGTH,
        adjust=False,
        min_periods=SLOW_EMA_LENGTH,
    ).mean()

    prior_channel_high = high.shift(1).rolling(
        BREAKOUT_LOOKBACK,
        min_periods=BREAKOUT_LOOKBACK,
    ).max()

    prior_channel_low = low.shift(1).rolling(
        BREAKOUT_LOOKBACK,
        min_periods=BREAKOUT_LOOKBACK,
    ).min()

    long_breakout = (
        (close > prior_channel_high)
        & (close.shift(1) <= prior_channel_high.shift(1))
    ).fillna(False)

    short_breakout = (
        (close < prior_channel_low)
        & (close.shift(1) >= prior_channel_low.shift(1))
    ).fillna(False)

    long_regime = (
        (fast_ema > slow_ema)
        & (slow_ema.diff() >= 0)
    ).fillna(False)

    short_regime = (
        (fast_ema < slow_ema)
        & (slow_ema.diff() <= 0)
    ).fillna(False)

    usable_volatility = (
        atr.notna()
        & atr.gt(0)
        & close.notna()
        & close.ne(0)
    )

    candidate_event = (
        ((long_breakout & long_regime) | (short_breakout & short_regime))
        & usable_volatility
    ).fillna(False)

    timestamp = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    day_key = pd.Series(
        timestamp.dt.normalize(),
        index=df.index,
    ).fillna(pd.Timestamp("1900-01-01"))

    daily_entry_count = (
        candidate_event.astype(np.int64)
        .groupby(day_key, sort=False)
        .cumsum()
    )

    daily_limit_ok = daily_entry_count.le(MAX_ENTRIES_PER_DAY)

    long_entry = (
        long_breakout
        & long_regime
        & usable_volatility
        & daily_limit_ok
    ).fillna(False)

    short_entry = (
        short_breakout
        & short_regime
        & usable_volatility
        & daily_limit_ok
    ).fillna(False)

    close_values = close.to_numpy(dtype=float)
    fast_values = fast_ema.to_numpy(dtype=float)
    long_entry_values = long_entry.to_numpy(dtype=bool)
    short_entry_values = short_entry.to_numpy(dtype=bool)

    signal_values = np.zeros(n, dtype=np.int8)

    position = 0
    bars_held = 0

    for i in range(n):
        if position == 0:
            if long_entry_values[i]:
                position = 1
                bars_held = 0
            elif short_entry_values[i]:
                position = -1
                bars_held = 0

        elif position == 1:
            bars_held += 1

            current_close = close_values[i]
            current_fast = fast_values[i]

            invalid_long = (
                not np.isfinite(current_close)
                or not np.isfinite(current_fast)
                or current_close < current_fast
                or short_entry_values[i]
                or bars_held >= MAX_HOLD_BARS
            )

            if invalid_long:
                position = 0
                bars_held = 0

        elif position == -1:
            bars_held += 1

            current_close = close_values[i]
            current_fast = fast_values[i]

            invalid_short = (
                not np.isfinite(current_close)
                or not np.isfinite(current_fast)
                or current_close > current_fast
                or long_entry_values[i]
                or bars_held >= MAX_HOLD_BARS
            )

            if invalid_short:
                position = 0
                bars_held = 0

        signal_values[i] = position

    signals = pd.Series(
        signal_values,
        index=df.index,
        dtype="int64",
        name="signal",
    )

    atr_values = atr.to_numpy(dtype=float)
    valid_atr = np.isfinite(atr_values) & (atr_values > 0)

    stop_loss_distance = np.where(
        valid_atr,
        1.20 * atr_values,
        0.0,
    )

    take_profit_distance = np.where(
        valid_atr,
        1.80 * atr_values,
        0.0,
    )

    trailing_stop_distance = np.where(
        valid_atr,
        1.00 * atr_values,
        0.0,
    )

    signals.attrs["stop_loss_distance"] = stop_loss_distance
    signals.attrs["take_profit_distance"] = take_profit_distance
    signals.attrs["trailing_stop_distance"] = trailing_stop_distance
    signals.attrs["breakeven_trigger_r"] = 1.0

    return signals
