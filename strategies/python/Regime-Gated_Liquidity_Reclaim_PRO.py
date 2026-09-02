import pandas as pd
import numpy as np

STRATEGY_NAME = "Regime-Gated Liquidity Reclaim PRO"

# ============================================================
# CORE PARAMETERS
# ============================================================

ATR_PERIOD = 20

FAST_EMA = 50
SLOW_EMA = 200

LIQUIDITY_LOOKBACK = 36

FVG_EXPIRY_BARS = 8
SWEEP_RECENCY_BARS = 12

# Stronger displacement requirement
MIN_DISPLACEMENT_ATR = 0.80
MIN_CLOSE_LOCATION = 0.68

# Volatility regime
MIN_ATR_RATIO = 0.80
MAX_ATR_RATIO = 1.75

# Stop / target
STOP_BUFFER_ATR = 0.12
MAX_STOP_ATR = 1.50

TARGET_R = 1.40
BREAKEVEN_R = 0.75

MAX_HOLD_BARS = 16

# ============================================================
# PROP-FIRM RISK GOVERNOR
# ============================================================

# These are expressed in R, not dollars.
# Actual position sizing should be handled by T58.

MAX_TRADES_PER_DAY = 3
MAX_LOSSES_PER_DAY = 2

MAX_DAILY_LOSS_R = 1.00

# Once a good day is achieved, stop trading.
DAILY_PROFIT_LOCK_R = 1.50

# Stop after a losing streak.
MAX_CONSECUTIVE_LOSSES = 2

# Recent equity deterioration protection.
EQUITY_LOOKBACK_TRADES = 8
MAX_RECENT_LOSS_R = 2.50

# Cooldown after any completed trade
COOLDOWN_BARS = 6

# Additional cooldown after a losing trade
LOSS_COOLDOWN_BARS = 12

# ============================================================
# SESSION WINDOWS
# ============================================================

SESSION_1_START_ET = 9 * 60 + 30
SESSION_1_END_ET = 11 * 60 + 30

SESSION_2_START_ET = 13 * 60 + 30
SESSION_2_END_ET = 15 * 60 + 30


def generate_signals(df: pd.DataFrame) -> pd.Series:

    n = len(df)
    idx = df.index

    out = np.zeros(n, dtype=int)

    sl_arr = np.full(n, np.nan, dtype=float)
    tp_arr = np.full(n, np.nan, dtype=float)

    # ========================================================
    # TIME
    # ========================================================

    ts = pd.to_datetime(df["timestamp"])

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

    session_1 = (
        (minute >= SESSION_1_START_ET)
        & (minute <= SESSION_1_END_ET)
    )

    session_2 = (
        (minute >= SESSION_2_START_ET)
        & (minute <= SESSION_2_END_ET)
    )

    in_session = session_1 | session_2

    # Session identifier.
    session_id = np.where(
        session_1,
        1,
        np.where(session_2, 2, 0)
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

    atr = (
        tr.ewm(
            alpha=1.0 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD
        ).mean()
    )

    atr_baseline = (
        atr
        .rolling(
            80,
            min_periods=80
        )
        .median()
    )

    atr_ratio = (
        atr /
        atr_baseline.replace(0, np.nan)
    )

    # ========================================================
    # TREND
    # ========================================================

    fast = (
        c.ewm(
            span=FAST_EMA,
            adjust=False,
            min_periods=FAST_EMA
        ).mean()
    )

    slow = (
        c.ewm(
            span=SLOW_EMA,
            adjust=False,
            min_periods=SLOW_EMA
        ).mean()
    )

    # EMA slope.
    fast_slope = fast - fast.shift(5)
    slow_slope = slow - slow.shift(10)

    # ========================================================
    # LIQUIDITY
    # ========================================================

    prior_high = (
        h.shift(1)
        .rolling(
            LIQUIDITY_LOOKBACK,
            min_periods=LIQUIDITY_LOOKBACK
        )
        .max()
    )

    prior_low = (
        l.shift(1)
        .rolling(
            LIQUIDITY_LOOKBACK,
            min_periods=LIQUIDITY_LOOKBACK
        )
        .min()
    )

    sweep_low = (
        (l < prior_low)
        &
        (c > prior_low)
    )

    sweep_high = (
        (h > prior_high)
        &
        (c < prior_high)
    )

    # ========================================================
    # CANDLE QUALITY
    # ========================================================

    body = (c - o).abs()

    candle_range = (
        h - l
    ).replace(0, np.nan)

    close_location = (
        (c - l) /
        candle_range
    )

    body_ratio = (
        body /
        candle_range
    )

    # ========================================================
    # FVG
    # ========================================================

    bull_fvg = (
        (l > h.shift(2))
        &
        (c > o)
        &
        (body >= MIN_DISPLACEMENT_ATR * atr)
        &
        (close_location >= MIN_CLOSE_LOCATION)
        &
        (body_ratio >= 0.55)
    )

    bear_fvg = (
        (h < l.shift(2))
        &
        (c < o)
        &
        (body >= MIN_DISPLACEMENT_ATR * atr)
        &
        (close_location <= 1.0 - MIN_CLOSE_LOCATION)
        &
        (body_ratio >= 0.55)
    )

    # ========================================================
    # STATE
    # ========================================================

    bull_zone_top = np.nan
    bull_zone_bottom = np.nan
    bull_zone_age = 10**9

    bear_zone_top = np.nan
    bear_zone_bottom = np.nan
    bear_zone_age = 10**9

    last_sweep_low_bar = -10**9
    last_sweep_high_bar = -10**9

    position = 0

    entry_price = np.nan
    stop_price = np.nan
    target_price = np.nan
    entry_risk = np.nan

    bars_held = 0

    cooldown = 0

    # ========================================================
    # PROP RISK STATE
    # ========================================================

    current_day = None

    daily_r = 0.0
    daily_trades = 0
    daily_losses = 0

    consecutive_losses = 0

    completed_trade_results = []

    session_trade_1 = False
    session_trade_2 = False

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for i in range(n):

        a = atr.iloc[i]

        if not np.isfinite(a) or a <= 0:
            out[i] = 0
            continue

        # ====================================================
        # NEW DAY RESET
        # ====================================================

        day = et.iloc[i].date()

        if current_day is None or day != current_day:

            current_day = day

            daily_r = 0.0
            daily_trades = 0
            daily_losses = 0

            session_trade_1 = False
            session_trade_2 = False

        # ====================================================
        # COOLDOWN
        # ====================================================

        if cooldown > 0:
            cooldown -= 1

        # ====================================================
        # UPDATE SWEEPS
        # ====================================================

        if bool(sweep_low.iloc[i]):
            last_sweep_low_bar = i

        if bool(sweep_high.iloc[i]):
            last_sweep_high_bar = i

        # ====================================================
        # UPDATE FVG ZONES
        # ====================================================

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

        # Expire zones.
        if bull_zone_age > FVG_EXPIRY_BARS:

            bull_zone_top = np.nan
            bull_zone_bottom = np.nan

        if bear_zone_age > FVG_EXPIRY_BARS:

            bear_zone_top = np.nan
            bear_zone_bottom = np.nan

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        if position != 0:

            if position == 1:

                # Breakeven
                if hh := h.iloc[i]:

                    if hh >= (
                        entry_price +
                        BREAKEVEN_R * entry_risk
                    ):
                        stop_price = max(
                            stop_price,
                            entry_price
                        )

                stop_hit = l.iloc[i] <= stop_price

                target_hit = h.iloc[i] >= target_price

                invalidation = (
                    c.iloc[i] <
                    fast.iloc[i] -
                    0.25 * a
                )

            else:

                if l.iloc[i] <= (
                    entry_price -
                    BREAKEVEN_R * entry_risk
                ):
                    stop_price = min(
                        stop_price,
                        entry_price
                    )

                stop_hit = h.iloc[i] >= stop_price

                target_hit = l.iloc[i] <= target_price

                invalidation = (
                    c.iloc[i] >
                    fast.iloc[i] +
                    0.25 * a
                )

            # ================================================
            # EXIT
            # ================================================

            if stop_hit or target_hit or invalidation:

                # Conservative assumption:
                # stop wins if both stop and target are touched.
                if stop_hit:

                    trade_r = -1.0

                elif target_hit:

                    trade_r = TARGET_R

                else:

                    # Invalidation is treated as a small loss.
                    if position == 1:

                        pnl_distance = (
                            c.iloc[i] -
                            entry_price
                        )

                    else:

                        pnl_distance = (
                            entry_price -
                            c.iloc[i]
                        )

                    trade_r = (
                        pnl_distance /
                        max(entry_risk, 1e-12)
                    )

                    trade_r = max(
                        -1.0,
                        min(
                            TARGET_R,
                            trade_r
                        )
                    )

                # Record result.
                completed_trade_results.append(
                    trade_r
                )

                daily_r += trade_r

                if trade_r < 0:

                    daily_losses += 1
                    consecutive_losses += 1

                    cooldown = LOSS_COOLDOWN_BARS

                else:

                    consecutive_losses = 0
                    cooldown = COOLDOWN_BARS

                daily_trades += 1

                position = 0
                bars_held = 0

                out[i] = 0

                continue

            # =================================================
            # MAX HOLD
            # =================================================

            bars_held += 1

            if (
                bars_held >= MAX_HOLD_BARS
                or not bool(in_session.iloc[i])
            ):

                # Time exit.
                if position == 1:

                    trade_r = (
                        c.iloc[i] -
                        entry_price
                    ) / max(
                        entry_risk,
                        1e-12
                    )

                else:

                    trade_r = (
                        entry_price -
                        c.iloc[i]
                    ) / max(
                        entry_risk,
                        1e-12
                    )

                trade_r = max(
                    -1.0,
                    min(
                        TARGET_R,
                        trade_r
                    )
                )

                daily_r += trade_r

                completed_trade_results.append(
                    trade_r
                )

                if trade_r < 0:

                    daily_losses += 1
                    consecutive_losses += 1
                    cooldown = LOSS_COOLDOWN_BARS

                else:

                    consecutive_losses = 0
                    cooldown = COOLDOWN_BARS

                daily_trades += 1

                position = 0
                bars_held = 0

                out[i] = 0

                continue

            out[i] = position

            sl_arr[i] = entry_risk

            tp_arr[i] = abs(
                target_price -
                entry_price
            )

            continue

        # ====================================================
        # ENTRY RISK GOVERNOR
        # ====================================================

        if cooldown > 0:
            out[i] = 0
            continue

        if not bool(in_session.iloc[i]):
            out[i] = 0
            continue

        # Maximum trades.
        if daily_trades >= MAX_TRADES_PER_DAY:
            out[i] = 0
            continue

        # Maximum losses.
        if daily_losses >= MAX_LOSSES_PER_DAY:
            out[i] = 0
            continue

        # Daily loss lock.
        if daily_r <= -MAX_DAILY_LOSS_R:
            out[i] = 0
            continue

        # Daily profit lock.
        if daily_r >= DAILY_PROFIT_LOCK_R:
            out[i] = 0
            continue

        # Consecutive loss lock.
        if (
            consecutive_losses >=
            MAX_CONSECUTIVE_LOSSES
        ):
            out[i] = 0
            continue

        # ====================================================
        # RECENT EQUITY CURVE FILTER
        # ====================================================

        if len(completed_trade_results) >= EQUITY_LOOKBACK_TRADES:

            recent = completed_trade_results[
                -EQUITY_LOOKBACK_TRADES:
            ]

            recent_r = sum(recent)

            if recent_r <= -MAX_RECENT_LOSS_R:
                out[i] = 0
                continue

        # ====================================================
        # VOLATILITY REGIME
        # ====================================================

        ratio = atr_ratio.iloc[i]

        regime_ok = (
            np.isfinite(ratio)
            and ratio >= MIN_ATR_RATIO
            and ratio <= MAX_ATR_RATIO
        )

        if not regime_ok:
            out[i] = 0
            continue

        # ====================================================
        # SWEEP RECENCY
        # ====================================================

        recent_low_sweep = (
            0 <=
            i - last_sweep_low_bar
            <= SWEEP_RECENCY_BARS
        )

        recent_high_sweep = (
            0 <=
            i - last_sweep_high_bar
            <= SWEEP_RECENCY_BARS
        )

        # ====================================================
        # SESSION LIMIT
        # ====================================================

        current_session = session_id[i]

        if current_session == 1 and session_trade_1:
            out[i] = 0
            continue

        if current_session == 2 and session_trade_2:
            out[i] = 0
            continue

        # ====================================================
        # LONG SETUP
        # ====================================================

        long_trend = (
            fast.iloc[i] > slow.iloc[i]
            and fast_slope.iloc[i] > 0
            and slow_slope.iloc[i] >= 0
        )

        long_reclaim = (

            recent_low_sweep

            and np.isfinite(bull_zone_top)

            and np.isfinite(bull_zone_bottom)

            and bull_zone_age <= FVG_EXPIRY_BARS

            and long_trend

            and l.iloc[i] <= bull_zone_top

            and c.iloc[i] > bull_zone_top

            and c.iloc[i] > o.iloc[i]

            and close_location.iloc[i] >= 0.65

            and body_ratio.iloc[i] >= 0.45
        )

        # ====================================================
        # SHORT SETUP
        # ====================================================

        short_trend = (
            fast.iloc[i] < slow.iloc[i]
            and fast_slope.iloc[i] < 0
            and slow_slope.iloc[i] <= 0
        )

        short_reclaim = (

            recent_high_sweep

            and np.isfinite(bear_zone_top)

            and np.isfinite(bear_zone_bottom)

            and bear_zone_age <= FVG_EXPIRY_BARS

            and short_trend

            and h.iloc[i] >= bear_zone_top

            and c.iloc[i] < bear_zone_top

            and c.iloc[i] < o.iloc[i]

            and close_location.iloc[i] <= 0.35

            and body_ratio.iloc[i] >= 0.45
        )

        # ====================================================
        # LONG ENTRY
        # ====================================================

        if long_reclaim:

            sweep_price = (
                l.iloc[last_sweep_low_bar]
                if last_sweep_low_bar >= 0
                else bull_zone_bottom
            )

            structural_stop = (
                min(
                    bull_zone_bottom,
                    sweep_price
                )
                -
                STOP_BUFFER_ATR * a
            )

            risk = abs(
                c.iloc[i] -
                structural_stop
            )

            # Reject abnormally large stops.
            if risk > MAX_STOP_ATR * a:
                out[i] = 0
                continue

            risk = max(
                risk,
                0.35 * a
            )

            entry_price = c.iloc[i]

            stop_price = (
                entry_price -
                risk
            )

            target_price = (
                entry_price +
                TARGET_R * risk
            )

            entry_risk = risk

            position = 1

            bars_held = 0

            daily_trades += 1

            session_trade_1 |= (
                current_session == 1
            )

            session_trade_2 |= (
                current_session == 2
            )

            sl_arr[i] = risk

            tp_arr[i] = (
                TARGET_R *
                risk
            )

            out[i] = 1

        # ====================================================
        # SHORT ENTRY
        # ====================================================

        elif short_reclaim:

            sweep_price = (
                h.iloc[last_sweep_high_bar]
                if last_sweep_high_bar >= 0
                else bear_zone_top
            )

            structural_stop = (
                max(
                    bear_zone_bottom,
                    sweep_price
                )
                +
                STOP_BUFFER_ATR * a
            )

            risk = abs(
                c.iloc[i] -
                structural_stop
            )

            if risk > MAX_STOP_ATR * a:
                out[i] = 0
                continue

            risk = max(
                risk,
                0.35 * a
            )

            entry_price = c.iloc[i]

            stop_price = (
                entry_price +
                risk
            )

            target_price = (
                entry_price -
                TARGET_R * risk
            )

            entry_risk = risk

            position = -1

            bars_held = 0

            daily_trades += 1

            session_trade_1 |= (
                current_session == 1
            )

            session_trade_2 |= (
                current_session == 2
            )

            sl_arr[i] = risk

            tp_arr[i] = (
                TARGET_R *
                risk
            )

            out[i] = -1

        else:

            out[i] = 0

    # ========================================================
    # RETURN SIGNALS
    # ========================================================

    signals = pd.Series(
        out,
        index=idx
    )

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
