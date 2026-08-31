import pandas as pd
import numpy as np

STRATEGY_NAME = "Regime-Gated Liquidity Reclaim"

# Searchable parameters. They are expressed in bars / ATR multiples rather
# than absolute price units so the strategy scales across instruments.
ATR_PERIOD = 20
FAST_EMA = 50
SLOW_EMA = 200
LIQUIDITY_LOOKBACK = 36
FVG_EXPIRY_BARS = 10
MIN_DISPLACEMENT_ATR = 0.80
MIN_CLOSE_LOCATION = 0.68
MIN_ATR_RATIO = 0.75
MAX_ATR_RATIO = 1.80
STOP_BUFFER_ATR = 0.15
TARGET_R = 1.50
BREAKEVEN_R = 0.90
MAX_HOLD_BARS = 18
COOLDOWN_BARS = 6
SESSION_START_ET = 9 * 60 + 30
SESSION_END_ET = 12 * 60


def generate_signals(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    idx = df.index
    out = np.zeros(n, dtype=int)

    ts = pd.to_datetime(df["timestamp"])
    # The bundled T58 intraday data is UTC. Convert to New York using pandas
    # so the session follows DST rather than hard-coding a UTC offset.
    if ts.dt.tz is None:
        et = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        et = ts.dt.tz_convert("America/New_York")
    minute = et.dt.hour * 60 + et.dt.minute
    in_session = (minute >= SESSION_START_ET) & (minute <= SESSION_END_ET)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    atr_baseline = atr.rolling(80, min_periods=80).median()
    atr_ratio = atr / atr_baseline.replace(0, np.nan)

    fast = c.ewm(span=FAST_EMA, adjust=False, min_periods=FAST_EMA).mean()
    slow = c.ewm(span=SLOW_EMA, adjust=False, min_periods=SLOW_EMA).mean()

    # Prior-bar liquidity levels. shift(1) is essential: the current bar is
    # never allowed to define the level it is then tested against.
    prior_high = h.shift(1).rolling(LIQUIDITY_LOOKBACK, min_periods=LIQUIDITY_LOOKBACK).max()
    prior_low = l.shift(1).rolling(LIQUIDITY_LOOKBACK, min_periods=LIQUIDITY_LOOKBACK).min()

    # A sweep is an event, not a permanent state.
    sweep_low = (l < prior_low) & (c > prior_low)
    sweep_high = (h > prior_high) & (c < prior_high)

    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    close_location = (c - l) / candle_range

    # Textbook three-candle FVGs: no overlap between bar i and bar i-2.
    bull_fvg = (
        (l > h.shift(2))
        & (c > o)
        & (body >= MIN_DISPLACEMENT_ATR * atr)
        & (close_location >= MIN_CLOSE_LOCATION)
    )
    bear_fvg = (
        (h < l.shift(2))
        & (c < o)
        & (body >= MIN_DISPLACEMENT_ATR * atr)
        & (close_location <= 1.0 - MIN_CLOSE_LOCATION)
    )

    # Each qualifying FVG becomes an active zone only after it is formed.
    # We pair it with a recent liquidity sweep and a directional trend regime.
    bull_zone_top = np.nan
    bull_zone_bottom = np.nan
    bull_zone_age = 10**9
    bear_zone_top = np.nan
    bear_zone_bottom = np.nan
    bear_zone_age = 10**9
    last_sweep_low_bar = -10**9
    last_sweep_high_bar = -10**9

    # Dynamic execution arrays: only the value on the actual entry bar is
    # consumed by T58's execution engine.
    sl_arr = np.full(n, np.nan, dtype=float)
    tp_arr = np.full(n, np.nan, dtype=float)

    position = 0
    entry_price = np.nan
    stop_price = np.nan
    target_price = np.nan
    entry_risk = np.nan
    bars_held = 0
    cooldown = 0

    for i in range(n):
        a = atr.iloc[i]
        hh = h.iloc[i]
        ll = l.iloc[i]
        cc = c.iloc[i]

        if not np.isfinite(a) or a <= 0:
            out[i] = 0
            continue

        if cooldown > 0:
            cooldown -= 1

        if bool(sweep_low.iloc[i]):
            last_sweep_low_bar = i
        if bool(sweep_high.iloc[i]):
            last_sweep_high_bar = i

        # Add new zones only after the current candle has closed.
        if bool(bull_fvg.iloc[i]):
            bull_zone_top = h.iloc[i - 2]
            bull_zone_bottom = l.iloc[i]
            bull_zone_age = 0
        elif np.isfinite(bull_zone_top):
            bull_zone_age += 1

        if bool(bear_fvg.iloc[i]):
            bear_zone_top = l.iloc[i - 2]
            bear_zone_bottom = h.iloc[i]
            bear_zone_age = 0
        elif np.isfinite(bear_zone_top):
            bear_zone_age += 1

        if bull_zone_age > FVG_EXPIRY_BARS:
            bull_zone_top = np.nan
            bull_zone_bottom = np.nan
        if bear_zone_age > FVG_EXPIRY_BARS:
            bear_zone_top = np.nan
            bear_zone_bottom = np.nan

        # Manage the strategy's virtual position state. This state uses only
        # current/past prices; it is NOT an account-P&L counter.
        if position != 0:
            if position == 1:
                if hh >= entry_price + BREAKEVEN_R * entry_risk:
                    stop_price = max(stop_price, entry_price)
                stop_hit = ll <= stop_price
                target_hit = hh >= target_price
                invalidation = cc < fast.iloc[i] - 0.25 * a
            else:
                if ll <= entry_price - BREAKEVEN_R * entry_risk:
                    stop_price = min(stop_price, entry_price)
                stop_hit = hh >= stop_price
                target_hit = ll <= target_price
                invalidation = cc > fast.iloc[i] + 0.25 * a

            # Conservative same-bar assumption: stop wins if both levels are
            # touched by the same candle.
            if stop_hit or target_hit or invalidation:
                position = 0
                bars_held = 0
                cooldown = COOLDOWN_BARS
                out[i] = 0
                continue

            # Do not carry exposure indefinitely or across the session close.
            bars_held += 1
            if bars_held >= MAX_HOLD_BARS or not bool(in_session.iloc[i]):
                position = 0
                bars_held = 0
                cooldown = COOLDOWN_BARS
                out[i] = 0
                continue

            out[i] = position
            sl_arr[i] = entry_risk
            tp_arr[i] = abs(target_price - entry_price)
            continue

        if cooldown > 0 or not bool(in_session.iloc[i]):
            out[i] = 0
            continue

        regime_ok = (
            (atr_ratio.iloc[i] >= MIN_ATR_RATIO)
            & (atr_ratio.iloc[i] <= MAX_ATR_RATIO)
        )

        # The sweep must be recent enough to belong to the same structural
        # setup as the displacement/FVG. This avoids treating an unrelated
        # sweep from hours earlier as confirmation.
        recent_low_sweep = 0 <= i - last_sweep_low_bar <= FVG_EXPIRY_BARS
        recent_high_sweep = 0 <= i - last_sweep_high_bar <= FVG_EXPIRY_BARS

        long_reclaim = (
            recent_low_sweep
            and np.isfinite(bull_zone_top)
            and np.isfinite(bull_zone_bottom)
            and bull_zone_age <= FVG_EXPIRY_BARS
            and fast.iloc[i] > slow.iloc[i]
            and fast.iloc[i] > fast.iloc[max(0, i - 5)]
            and regime_ok
            and ll <= bull_zone_top
            and cc > bull_zone_top
            and cc > o.iloc[i]
        )

        short_reclaim = (
            recent_high_sweep
            and np.isfinite(bear_zone_top)
            and np.isfinite(bear_zone_bottom)
            and bear_zone_age <= FVG_EXPIRY_BARS
            and fast.iloc[i] < slow.iloc[i]
            and fast.iloc[i] < fast.iloc[max(0, i - 5)]
            and regime_ok
            and hh >= bear_zone_top
            and cc < bear_zone_top
            and cc < o.iloc[i]
        )

        if long_reclaim:
            position = 1
            entry_price = cc
            sweep_low_price = l.iloc[last_sweep_low_bar] if last_sweep_low_bar >= 0 else bull_zone_bottom
            stop_price = min(bull_zone_bottom, sweep_low_price) - STOP_BUFFER_ATR * a
            entry_risk = max(abs(entry_price - stop_price), 0.35 * a)
            stop_price = entry_price - entry_risk
            target_price = entry_price + TARGET_R * entry_risk
            bars_held = 0
            sl_arr[i] = entry_risk
            tp_arr[i] = TARGET_R * entry_risk
            out[i] = 1
        elif short_reclaim:
            position = -1
            entry_price = cc
            sweep_high_price = h.iloc[last_sweep_high_bar] if last_sweep_high_bar >= 0 else bear_zone_top
            stop_price = max(bear_zone_bottom, sweep_high_price) + STOP_BUFFER_ATR * a
            entry_risk = max(abs(entry_price - stop_price), 0.35 * a)
            stop_price = entry_price + entry_risk
            target_price = entry_price - TARGET_R * entry_risk
            bars_held = 0
            sl_arr[i] = entry_risk
            tp_arr[i] = TARGET_R * entry_risk
            out[i] = -1
        else:
            out[i] = 0

    signals = pd.Series(out, index=idx)
    signals.attrs["stop_loss_distance"] = sl_arr
    signals.attrs["take_profit_distance"] = tp_arr
    signals.attrs["breakeven_trigger_r"] = BREAKEVEN_R
    return signals
