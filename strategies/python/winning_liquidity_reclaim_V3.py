import pandas as pd
import numpy as np

STRATEGY_NAME = "Prop Liquidity Reclaim V2"

# ============================================================
# SEARCHABLE PARAMETERS
# ============================================================

ATR_PERIOD = 20

FAST_EMA = 50
SLOW_EMA = 200

LIQUIDITY_LOOKBACK = 30
SWEEP_MAX_AGE = 8

FVG_EXPIRY_BARS = 12

# Lower than V1 to increase trade frequency while retaining
# meaningful displacement.
MIN_DISPLACEMENT_ATR = 0.65
MIN_CLOSE_LOCATION = 0.62

# Avoid extremely dead or extremely chaotic volatility regimes.
MIN_ATR_RATIO = 0.65
MAX_ATR_RATIO = 1.65

# Structural stop settings.
STOP_BUFFER_ATR = 0.12
MIN_STOP_ATR = 0.35

# Prop-oriented reward target.
TARGET_R = 1.35

# Move stop to breakeven after this much favorable movement.
BREAKEVEN_R = 0.85

# Avoid holding trades indefinitely.
MAX_HOLD_BARS = 14

# Prevent immediate re-entry after an exit.
COOLDOWN_BARS = 4

# New York trading session.
SESSION_START_ET = 8 * 60 + 30
SESSION_END_ET = 12 * 60


def generate_signals(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    idx = df.index

    out = np.zeros(n, dtype=int)

    # ========================================================
    # TIME / SESSION
    # ========================================================

    ts = pd.to_datetime(df["timestamp"])

    # T58 bundled intraday data is normally UTC.
    # Convert to New York so DST is handled correctly.
    if ts.dt.tz is None:
        et = (
            ts.dt
            .tz_localize("UTC")
            .dt
            .tz_convert("America/New_York")
        )
    else:
        et = ts.dt.tz_convert("America/New_York")

    minute = et.dt.hour * 60 + et.dt.minute

    in_session = (
        (minute >= SESSION_START_ET)
        & (minute <= SESSION_END_ET)
    )

    # ========================================================
    # PRICE DATA
    # ========================================================

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # ========================================================
    # ATR
    # ========================================================

    prev_close = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1.0 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD,
    ).mean()

    # Median ATR baseline gives us a relative volatility regime.
    atr_baseline = atr.rolling(
        80,
        min_periods=80,
    ).median()

    atr_ratio = (
        atr
        / atr_baseline.replace(0, np.nan)
    )

    # ========================================================
    # TREND REGIME
    # ========================================================

    fast = c.ewm(
        span=FAST_EMA,
        adjust=False,
        min_periods=FAST_EMA,
    ).mean()

    slow = c.ewm(
        span=SLOW_EMA,
        adjust=False,
        min_periods=SLOW_EMA,
    ).mean()

    # Five-bar slope of fast EMA.
    fast_slope = fast - fast.shift(5)

    # ========================================================
    # LIQUIDITY LEVELS
    # ========================================================

    # IMPORTANT:
    # shift(1) prevents the current candle from defining the
    # liquidity level that it is subsequently used to sweep.

    prior_high = h.shift(1).rolling(
        LIQUIDITY_LOOKBACK,
        min_periods=LIQUIDITY_LOOKBACK,
    ).max()

    prior_low = l.shift(1).rolling(
        LIQUIDITY_LOOKBACK,
        min_periods=LIQUIDITY_LOOKBACK,
    ).min()

    # ========================================================
    # LIQUIDITY SWEEPS
    # ========================================================

    # Bullish liquidity sweep:
    # price trades below prior lows and closes back above them.

    sweep_low = (
        (l < prior_low)
        & (c > prior_low)
    )

    # Bearish liquidity sweep:
    # price trades above prior highs and closes back below them.

    sweep_high = (
        (h > prior_high)
        & (c < prior_high)
    )

    # ========================================================
    # CANDLE QUALITY
    # ========================================================

    body = (c - o).abs()

    candle_range = (
        h - l
    ).replace(0, np.nan)

    close_location = (
        (c - l)
        / candle_range
    )

    # Bullish displacement candle.
    bull_displacement = (
        (c > o)
        & (
            body
            >= MIN_DISPLACEMENT_ATR * atr
        )
        & (
            close_location
            >= MIN_CLOSE_LOCATION
        )
    )

    # Bearish displacement candle.
    bear_displacement = (
        (c < o)
        & (
            body
            >= MIN_DISPLACEMENT_ATR * atr
        )
        & (
            close_location
            <= 1.0 - MIN_CLOSE_LOCATION
        )
    )

    # ========================================================
    # FAIR VALUE GAPS
    # ========================================================

    # Bullish FVG:
    # current low > high two candles ago.

    bull_fvg = (
        (l > h.shift(2))
        & bull_displacement
    )

    # Bearish FVG:
    # current high < low two candles ago.

    bear_fvg = (
        (h < l.shift(2))
        & bear_displacement
    )

    # ========================================================
    # STATE VARIABLES
    # ========================================================

    bull_zone_top = np.nan
    bull_zone_bottom = np.nan
    bull_zone_age = 10**9

    bear_zone_top = np.nan
    bear_zone_bottom = np.nan
    bear_zone_age = 10**9

    last_sweep_low_bar = -10**9
    last_sweep_high_bar = -10**9

    # ========================================================
    # EXECUTION METADATA
    # ========================================================

    sl_arr = np.full(
        n,
        np.nan,
        dtype=float,
    )

    tp_arr = np.full(
        n,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # VIRTUAL POSITION STATE
    # ========================================================

    position = 0

    entry_price = np.nan
    stop_price = np.nan
    target_price = np.nan
    entry_risk = np.nan

    bars_held = 0
    cooldown = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for i in range(n):

        a = atr.iloc[i]

        # ----------------------------------------------------
        # Need valid ATR before doing anything.
        # ----------------------------------------------------

        if not np.isfinite(a) or a <= 0:
            out[i] = 0
            continue

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if cooldown > 0:
            cooldown -= 1

        # ====================================================
        # RECORD NEW LIQUIDITY SWEEPS
        # ====================================================

        if bool(sweep_low.iloc[i]):
            last_sweep_low_bar = i

        if bool(sweep_high.iloc[i]):
            last_sweep_high_bar = i

        # ====================================================
        # CREATE / AGE BULLISH FVG
        # ====================================================

        if bool(bull_fvg.iloc[i]):

            # Bullish FVG:
            #
            # bottom = high from candle i-2
            # top    = low from current candle

            bull_zone_top = l.iloc[i]
            bull_zone_bottom = h.iloc[i - 2]

            bull_zone_age = 0

        elif np.isfinite(bull_zone_top):

            bull_zone_age += 1

        # ====================================================
        # CREATE / AGE BEARISH FVG
        # ====================================================

        if bool(bear_fvg.iloc[i]):

            # Bearish FVG:
            #
            # bottom = high of current candle
            # top    = low from candle i-2

            bear_zone_top = l.iloc[i - 2]
            bear_zone_bottom = h.iloc[i]

            bear_zone_age = 0

        elif np.isfinite(bear_zone_top):

            bear_zone_age += 1

        # ====================================================
        # EXPIRE OLD FVG ZONES
        # ====================================================

        if bull_zone_age > FVG_EXPIRY_BARS:

            bull_zone_top = np.nan
            bull_zone_bottom = np.nan

        if bear_zone_age > FVG_EXPIRY_BARS:

            bear_zone_top = np.nan
            bear_zone_bottom = np.nan

        # ====================================================
        # MANAGE EXISTING POSITION
        # ====================================================

        if position != 0:

            # =================================================
            # LONG POSITION
            # =================================================

            if position == 1:

                # Move stop to breakeven after sufficient
                # favorable movement.

                if (
                    h.iloc[i]
                    >= entry_price
                    + BREAKEVEN_R * entry_risk
                ):

                    stop_price = max(
                        stop_price,
                        entry_price,
                    )

                stop_hit = (
                    l.iloc[i]
                    <= stop_price
                )

                target_hit = (
                    h.iloc[i]
                    >= target_price
                )

                # Trend invalidation.
                invalidation = (
                    np.isfinite(fast.iloc[i])
                    and (
                        c.iloc[i]
                        < fast.iloc[i]
                        - 0.20 * a
                    )
                )

            # =================================================
            # SHORT POSITION
            # =================================================

            else:

                if (
                    l.iloc[i]
                    <= entry_price
                    - BREAKEVEN_R * entry_risk
                ):

                    stop_price = min(
                        stop_price,
                        entry_price,
                    )

                stop_hit = (
                    h.iloc[i]
                    >= stop_price
                )

                target_hit = (
                    l.iloc[i]
                    <= target_price
                )

                # Trend invalidation.
                invalidation = (
                    np.isfinite(fast.iloc[i])
                    and (
                        c.iloc[i]
                        > fast.iloc[i]
                        + 0.20 * a
                    )
                )

            # =================================================
            # EXIT
            # =================================================

            # Conservative assumption:
            # If stop and target are both touched on the same
            # candle, assume the stop happened first.

            if (
                stop_hit
                or target_hit
                or invalidation
            ):

                position = 0
                bars_held = 0
                cooldown = COOLDOWN_BARS

                out[i] = 0

                continue

            # ------------------------------------------------
            # Increment holding time.
            # ------------------------------------------------

            bars_held += 1

            # ------------------------------------------------
            # Time-based exit.
            # ------------------------------------------------

            if (
                bars_held >= MAX_HOLD_BARS
                or not bool(in_session.iloc[i])
            ):

                position = 0
                bars_held = 0
                cooldown = COOLDOWN_BARS

                out[i] = 0

                continue

            # ------------------------------------------------
            # Continue position.
            # ------------------------------------------------

            out[i] = position

            sl_arr[i] = entry_risk

            tp_arr[i] = abs(
                target_price
                - entry_price
            )

            continue

        # ====================================================
        # NO POSITION
        # ====================================================

        if (
            cooldown > 0
            or not bool(in_session.iloc[i])
        ):

            out[i] = 0
            continue

        # Need valid trend values.
        if (
            not np.isfinite(fast.iloc[i])
            or not np.isfinite(slow.iloc[i])
        ):

            out[i] = 0
            continue

        # ====================================================
        # VOLATILITY FILTER
        # ====================================================

        regime_ok = (
            np.isfinite(atr_ratio.iloc[i])
            and (
                atr_ratio.iloc[i]
                >= MIN_ATR_RATIO
            )
            and (
                atr_ratio.iloc[i]
                <= MAX_ATR_RATIO
            )
        )

        # ====================================================
        # RECENT SWEEP FILTER
        # ====================================================

        recent_low_sweep = (
            0
            <= i - last_sweep_low_bar
            <= SWEEP_MAX_AGE
        )

        recent_high_sweep = (
            0
            <= i - last_sweep_high_bar
            <= SWEEP_MAX_AGE
        )

        # ====================================================
        # LONG SETUP
        # ====================================================

        long_reclaim = (

            # Recent sell-side liquidity sweep.
            recent_low_sweep

            # Valid bullish FVG.
            and np.isfinite(bull_zone_top)
            and np.isfinite(bull_zone_bottom)

            and (
                bull_zone_age
                <= FVG_EXPIRY_BARS
            )

            # ----------------------------------------------
            # Trend regime
            # ----------------------------------------------

            and (
                fast.iloc[i]
                > slow.iloc[i]
            )

            and (
                fast_slope.iloc[i]
                > 0
            )

            # ----------------------------------------------
            # Volatility regime
            # ----------------------------------------------

            and regime_ok

            # ----------------------------------------------
            # Reclaim the FVG
            # ----------------------------------------------

            and (
                l.iloc[i]
                <= bull_zone_top
            )

            and (
                c.iloc[i]
                > bull_zone_top
            )

            # Confirmation candle closes bullish.
            and (
                c.iloc[i]
                > o.iloc[i]
            )
        )

        # ====================================================
        # SHORT SETUP
        # ====================================================

        short_reclaim = (

            # Recent buy-side liquidity sweep.
            recent_high_sweep

            # Valid bearish FVG.
            and np.isfinite(bear_zone_top)
            and np.isfinite(bear_zone_bottom)

            and (
                bear_zone_age
                <= FVG_EXPIRY_BARS
            )

            # ----------------------------------------------
            # Trend regime
            # ----------------------------------------------

            and (
                fast.iloc[i]
                < slow.iloc[i]
            )

            and (
                fast_slope.iloc[i]
                < 0
            )

            # ----------------------------------------------
            # Volatility regime
            # ----------------------------------------------

            and regime_ok

            # ----------------------------------------------
            # Reclaim bearish FVG
            # ----------------------------------------------

            and (
                h.iloc[i]
                >= bear_zone_bottom
            )

            and (
                c.iloc[i]
                < bear_zone_bottom
            )

            # Confirmation candle closes bearish.
            and (
                c.iloc[i]
                < o.iloc[i]
            )
        )

        # ====================================================
        # LONG EXECUTION
        # ====================================================

        if long_reclaim:

            position = 1

            entry_price = c.iloc[i]

            # The liquidity sweep is the structural
            # invalidation point.

            sweep_price = l.iloc[
                last_sweep_low_bar
            ]

            raw_stop = (
                sweep_price
                - STOP_BUFFER_ATR * a
            )

            # Never allow an abnormally tiny stop.
            risk = max(
                abs(
                    entry_price
                    - raw_stop
                ),
                MIN_STOP_ATR * a,
            )

            entry_risk = risk

            stop_price = (
                entry_price
                - risk
            )

            target_price = (
                entry_price
                + TARGET_R * risk
            )

            bars_held = 0

            sl_arr[i] = entry_risk

            tp_arr[i] = (
                TARGET_R
                * entry_risk
            )

            out[i] = 1

        # ====================================================
        # SHORT EXECUTION
        # ====================================================

        elif short_reclaim:

            position = -1

            entry_price = c.iloc[i]

            # Structural invalidation above the liquidity
            # sweep.

            sweep_price = h.iloc[
                last_sweep_high_bar
            ]

            raw_stop = (
                sweep_price
                + STOP_BUFFER_ATR * a
            )

            # Never allow an abnormally tiny stop.

            risk = max(
                abs(
                    raw_stop
                    - entry_price
                ),
                MIN_STOP_ATR * a,
            )

            entry_risk = risk

            stop_price = (
                entry_price
                + risk
            )

            target_price = (
                entry_price
                - TARGET_R * risk
            )

            bars_held = 0

            sl_arr[i] = entry_risk

            tp_arr[i] = (
                TARGET_R
                * entry_risk
            )

            out[i] = -1

        else:

            out[i] = 0

    # ========================================================
    # RETURN SIGNAL SERIES
    # ========================================================

    signals = pd.Series(
        out,
        index=idx,
    )

    # T58 execution metadata.
    signals.attrs[
        "stop_loss_distance"
    ] = sl_arr

    signals.attrs[
        "take_profit_distance"
    ] = tp_arr

    signals.attrs[
        "breakeven_trigger_r"
    ] = BREAKEVEN_R

    return signals
