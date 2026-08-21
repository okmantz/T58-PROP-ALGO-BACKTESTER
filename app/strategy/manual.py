"""
T58 Manual Strategy Engine.

The manual strategy format is deliberately data-driven so the visual builder
can create complex strategies without generating Python source code.

Backward compatibility:
- The original expression-based config still works.
- The original ``indicators`` + ``long_entry``/``short_entry`` fields still work.
- New visual-builder configs use ``entry_conditions`` / ``exit_conditions``
  and ``risk_management``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult, signals_from_conditions
from app.strategy.expr import safe_eval_bool
from app.strategy.indicators import build_indicator_series, INDICATOR_FUNCS


class ManualStrategy(Strategy):
    source_type = "manual"

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise StrategyError("Manual strategy configuration must be a dictionary.")
        self.config = config

    # ------------------------------------------------------------------
    # Legacy indicator support
    # ------------------------------------------------------------------
    def _build_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for ind in self.config.get("indicators", []):
            itype = str(ind.get("type", "")).lower()
            if itype not in INDICATOR_FUNCS:
                raise StrategyError(
                    f"Unsupported indicator type '{itype}'. Supported: {list(INDICATOR_FUNCS)}"
                )
            col = ind.get("column", "close")
            if col not in work.columns:
                raise StrategyError(f"Indicator references unknown column '{col}'.")
            period = max(int(ind.get("period", 14)), 1)
            alias = ind.get("as") or f"{itype}_{period}_{col}"
            work[alias] = INDICATOR_FUNCS[itype](work[col], period)
        return work

    # ------------------------------------------------------------------
    # Visual-builder condition evaluation
    # ------------------------------------------------------------------
    def _series_from_operand(self, work: pd.DataFrame, operand: Any, side: str = "left") -> pd.Series:
        """Resolve a visual-builder operand into a numeric/boolean Series."""
        if isinstance(operand, (int, float, np.number)):
            return pd.Series(float(operand), index=work.index)
        if operand is None:
            return pd.Series(np.nan, index=work.index)

        if isinstance(operand, str):
            name = operand.lower().strip()
            if name in work.columns:
                return work[name]
            if name in {"open", "high", "low", "close", "volume"}:
                if name not in work.columns:
                    raise StrategyError(f"Market data does not contain '{name}'.")
                return work[name]
            try:
                return pd.Series(float(operand), index=work.index)
            except ValueError as exc:
                raise StrategyError(f"Unknown operand '{operand}'.") from exc

        if not isinstance(operand, dict):
            raise StrategyError(f"Invalid {side} operand: {operand!r}")

        kind = str(operand.get("type", operand.get("source", "close"))).lower().strip()
        field = str(operand.get("field", "close")).lower().strip()
        period = max(int(operand.get("period", 14) or 14), 1)
        lookback = max(int(operand.get("lookback", period) or period), 1)
        direction = str(operand.get("direction", "both")).lower().strip()

        if kind in {"value", "constant", "number"}:
            try:
                return pd.Series(float(operand.get("value", 0)), index=work.index)
            except (TypeError, ValueError) as exc:
                raise StrategyError("A numeric condition value is required.") from exc

        if kind in {"price", "open", "high", "low", "close", "volume"}:
            col = field if kind == "price" else kind
            if col not in work.columns:
                raise StrategyError(f"Market data does not contain '{col}'.")
            return work[col]

        if kind in {"ema", "sma", "wma", "rsi", "vwap", "macd", "macd_signal", "macd_histogram", "atr",
                    "bollinger_mid", "bollinger_upper", "bollinger_lower", "highest_high", "lowest_low",
                    "average_volume", "candle_range", "percentage_change"}:
            return build_indicator_series(work, kind, period=period, column=field, lookback=lookback)

        if kind == "candle_direction":
            bullish = work["close"] > work["open"]
            bearish = work["close"] < work["open"]
            if direction == "bearish":
                return bearish.astype(int)
            if direction == "bullish":
                return bullish.astype(int)
            return (bullish.astype(int) - bearish.astype(int))

        if kind in {"swing_high", "swing_low"}:
            left = work["high"] if kind == "swing_high" else work["low"]
            if kind == "swing_high":
                return (left == left.rolling(lookback * 2 + 1, center=True, min_periods=lookback + 1).max()).astype(int)
            return (left == left.rolling(lookback * 2 + 1, center=True, min_periods=lookback + 1).min()).astype(int)

        if kind in {"liquidity_sweep", "break_of_structure", "bos", "change_of_character", "choch", "fair_value_gap", "fvg", "order_block"}:
            return self._advanced_boolean(work, kind, lookback, direction).astype(int)

        if kind in {"session_high", "session_low", "previous_day_high", "previous_day_low", "previous_day_close",
                    "opening_range_high", "opening_range_low"}:
            return self._session_series(work, kind, operand)

        if kind in {"atr_regime", "volatility_regime"}:
            return self._regime_series(work, kind, period, operand)

        raise StrategyError(f"Unsupported visual-builder condition source '{kind}'.")

    def _advanced_boolean(self, work: pd.DataFrame, kind: str, lookback: int, direction: str) -> pd.Series:
        high, low, close = work["high"], work["low"], work["close"]
        prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=lookback).min()

        if kind == "liquidity_sweep":
            sweep_high = (high > prior_high) & (close < prior_high)
            sweep_low = (low < prior_low) & (close > prior_low)
            if direction == "bullish":
                return sweep_low.fillna(False)
            if direction == "bearish":
                return sweep_high.fillna(False)
            return (sweep_high | sweep_low).fillna(False)

        if kind in {"break_of_structure", "bos"}:
            up = close > prior_high
            down = close < prior_low
            if direction == "bullish":
                return up.fillna(False)
            if direction == "bearish":
                return down.fillna(False)
            return (up | down).fillna(False)

        if kind in {"change_of_character", "choch"}:
            hh = high.diff(lookback) > 0
            ll = low.diff(lookback) < 0
            bull_shift = (close > high.shift(lookback)) & ll.shift(1).fillna(False)
            bear_shift = (close < low.shift(lookback)) & hh.shift(1).fillna(False)
            if direction == "bullish":
                return bull_shift.fillna(False)
            if direction == "bearish":
                return bear_shift.fillna(False)
            return (bull_shift | bear_shift).fillna(False)

        if kind in {"fair_value_gap", "fvg"}:
            bull = low > high.shift(2)
            bear = high < low.shift(2)
            if direction == "bullish":
                return bull.fillna(False)
            if direction == "bearish":
                return bear.fillna(False)
            return (bull | bear).fillna(False)

        # Simple, deterministic order-block proxy: the last opposite candle
        # immediately preceding a displacement candle.
        bull_displacement = close > high.shift(1) + (high - low).rolling(lookback, min_periods=1).mean()
        bear_displacement = close < low.shift(1) - (high - low).rolling(lookback, min_periods=1).mean()
        bull_ob = (close.shift(1) < work["open"].shift(1)) & bull_displacement
        bear_ob = (close.shift(1) > work["open"].shift(1)) & bear_displacement
        if direction == "bullish":
            return bull_ob.fillna(False)
        if direction == "bearish":
            return bear_ob.fillna(False)
        return (bull_ob | bear_ob).fillna(False)

    def _session_mask(self, index: pd.Series, start: str, end: str) -> pd.Series:
        t = pd.to_datetime(index)
        start_t = pd.to_datetime(start).time()
        end_t = pd.to_datetime(end).time()
        times = t.dt.time
        if start_t <= end_t:
            return (times >= start_t) & (times <= end_t)
        return (times >= start_t) | (times <= end_t)

    def _session_series(self, work: pd.DataFrame, kind: str, operand: dict[str, Any]) -> pd.Series:
        ts = pd.to_datetime(work["timestamp"])
        day = ts.dt.normalize()
        session_start = operand.get("session_start", "08:30")
        session_end = operand.get("session_end", "15:00")
        mask = self._session_mask(ts, session_start, session_end)

        if kind in {"session_high", "session_low"}:
            source = work["high"] if kind == "session_high" else work["low"]
            # Expanding within the active session avoids future-bar leakage.
            grouped = source.where(mask).groupby(day)
            result = grouped.cummax() if kind == "session_high" else grouped.cummin()
            return result.ffill()

        daily = work.groupby(day)
        if kind == "previous_day_high":
            daily_val = daily["high"].max().shift(1)
            return day.map(daily_val)
        if kind == "previous_day_low":
            daily_val = daily["low"].min().shift(1)
            return day.map(daily_val)
        if kind == "previous_day_close":
            daily_val = daily["close"].last().shift(1)
            return day.map(daily_val)

        # Opening-range levels become available only after the opening window
        # has completed, preventing look-ahead bias inside the opening range.
        start_t = pd.to_datetime(session_start).time()
        end_t = pd.to_datetime(session_end).time()
        if start_t <= end_t:
            opening_mask = (ts.dt.time >= start_t) & (ts.dt.time <= end_t)
            after_open = ts.dt.time > end_t
        else:
            opening_mask = (ts.dt.time >= start_t) | (ts.dt.time <= end_t)
            after_open = ts.dt.time > end_t
        opening_high = work["high"].where(opening_mask).groupby(day).transform("max")
        opening_low = work["low"].where(opening_mask).groupby(day).transform("min")
        result = opening_high if kind == "opening_range_high" else opening_low
        return result.where(after_open).ffill()

    def _regime_series(self, work: pd.DataFrame, kind: str, period: int, operand: dict[str, Any]) -> pd.Series:
        atr = build_indicator_series(work, "atr", period, "close")
        if kind == "atr_regime":
            baseline = atr.rolling(max(period * 3, period + 1), min_periods=period).mean()
            out = pd.Series(0, index=work.index, dtype=int)
            out[atr > baseline * 1.25] = 1
            out[atr < baseline * 0.75] = -1
            return out
        returns = work["close"].pct_change()
        vol = returns.rolling(period, min_periods=period).std()
        baseline = vol.rolling(max(period * 3, period + 1), min_periods=period).mean()
        out = pd.Series(0, index=work.index, dtype=int)
        out[vol > baseline * 1.25] = 1
        out[vol < baseline * 0.75] = -1
        return out

    @staticmethod
    def _compare(left: pd.Series, operator: str, right: pd.Series) -> pd.Series:
        op = operator.strip().lower()
        if op in {">", "greater than", "gt"}:
            return left > right
        if op in {">=", "greater than or equal", "gte"}:
            return left >= right
        if op in {"<", "less than", "lt"}:
            return left < right
        if op in {"<=", "less than or equal", "lte"}:
            return left <= right
        if op in {"==", "equal to", "equals", "eq"}:
            return left == right
        if op in {"!=", "not equal", "neq"}:
            return left != right
        if op in {"cross above", "crosses above"}:
            return (left > right) & (left.shift(1) <= right.shift(1))
        if op in {"cross below", "crosses below"}:
            return (left < right) & (left.shift(1) >= right.shift(1))
        if op in {"is true", "true"}:
            return left.astype(bool)
        if op in {"is false", "false"}:
            return ~left.astype(bool)
        raise StrategyError(f"Unsupported condition operator '{operator}'.")

    def _evaluate_condition(self, work: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
        if not isinstance(condition, dict):
            raise StrategyError(f"Invalid condition: {condition!r}")
        left = self._series_from_operand(work, condition.get("left", condition.get("source", "close")), "left")
        right_operand = condition.get("right", condition.get("value", 0))
        right = self._series_from_operand(work, right_operand, "right")
        result = self._compare(left, str(condition.get("operator", ">")), right)
        return result.fillna(False).astype(bool)

    def _combine_conditions(self, work: pd.DataFrame, conditions: list[dict[str, Any]], connectors: list[str] | None = None) -> pd.Series:
        if not conditions:
            return pd.Series(False, index=work.index)
        result = self._evaluate_condition(work, conditions[0])
        connectors = connectors or []
        for i, condition in enumerate(conditions[1:], start=1):
            current = self._evaluate_condition(work, condition)
            connector = str(connectors[i - 1] if i - 1 < len(connectors) else "AND").upper()
            result = result & current if connector == "AND" else result | current
        return result.fillna(False).astype(bool)

    def _build_visual_signals(self, work: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        entries = self.config.get("entry_conditions", {})
        exits = self.config.get("exit_conditions", {})

        long_entry = self._combine_conditions(work, entries.get("long", []), entries.get("long_connectors"))
        short_entry = self._combine_conditions(work, entries.get("short", []), entries.get("short_connectors"))
        long_exit = self._combine_conditions(work, exits.get("long", []), exits.get("long_connectors"))
        short_exit = self._combine_conditions(work, exits.get("short", []), exits.get("short_connectors"))

        allowed = str(self.config.get("market", {}).get("direction", self.config.get("direction", "Both"))).lower()
        if allowed == "long":
            short_entry[:] = False
            short_exit[:] = False
        elif allowed == "short":
            long_entry[:] = False
            long_exit[:] = False

        return long_entry, long_exit, short_entry, short_exit

    def _apply_signal_exits(self, signals: pd.Series, work: pd.DataFrame) -> pd.Series:
        """Apply exits that can be represented by the engine's signal model."""
        rm = self.config.get("risk_management", {})
        out = signals.copy().astype(int)

        max_bars = rm.get("max_bars_in_trade")
        if max_bars:
            try:
                max_bars = max(int(max_bars), 1)
            except (TypeError, ValueError):
                max_bars = None
        if max_bars:
            vals = out.to_numpy(copy=True)
            position = 0
            bars = 0
            for i in range(len(vals)):
                if position == 0 and vals[i] != 0:
                    position = vals[i]
                    bars = 0
                elif position != 0:
                    if vals[i] != position:
                        position = vals[i]
                        bars = 0 if position != 0 else 0
                    else:
                        bars += 1
                        if bars >= max_bars:
                            vals[i] = 0
                            position = 0
                            bars = 0
            out = pd.Series(vals, index=signals.index)

        # Clock-time exit. Once the configured time is reached, flatten the
        # current signal for that bar and prevent re-entry for the remainder
        # of that trading day.
        time_cfg = rm.get("time_based_exit", {})
        if time_cfg.get("enabled") and time_cfg.get("time"):
            try:
                exit_time = pd.to_datetime(str(time_cfg["time"])).time()
                ts = pd.to_datetime(work["timestamp"])
                day = ts.dt.normalize()
                flattened = out.copy()
                for d in day.drop_duplicates():
                    mask = (day == d) & (ts.dt.time >= exit_time)
                    flattened.loc[mask] = 0
                out = flattened
            except (TypeError, ValueError):
                raise StrategyError("Time-Based Exit must use HH:MM format.")

        return out

    # ------------------------------------------------------------------
    # Risk management: ATR-aware stop loss / take profit / trailing stop
    # ------------------------------------------------------------------
    def _atr_series(self, work: pd.DataFrame, cache: dict[int, pd.Series], period: int) -> pd.Series:
        period = max(int(period or 14), 1)
        if period not in cache:
            cache[period] = build_indicator_series(work, "atr", period=period, column="close")
        return cache[period]

    def _build_risk_management(self, work: pd.DataFrame) -> dict[str, Any]:
        """
        Resolve the `risk_management` block (plus legacy top-level
        stop_loss_pips/take_profit_pips) into the values StrategyResult /
        the execution engine need. Every field is optional: anything not
        configured is simply left out, so a user who only wants a fixed
        stop loss and nothing else never has to touch ATR, trailing stop,
        or break-even settings.
        """
        cfg = self.config
        rm = cfg.get("risk_management", {}) or {}
        atr_cache: dict[int, pd.Series] = {}

        stop_loss_pips = None
        stop_loss_distance = None
        stop_type = str(rm.get("stop_type", "")).lower()
        if stop_type == "fixed" and rm.get("stop_value") not in (None, ""):
            stop_loss_pips = float(rm["stop_value"])
        elif stop_type == "atr" and rm.get("stop_value") not in (None, ""):
            mult = float(rm["stop_value"])
            stop_loss_distance = self._atr_series(work, atr_cache, rm.get("stop_atr_period", 14)) * mult
        elif cfg.get("stop_loss_pips") not in (None, ""):
            stop_loss_pips = float(cfg["stop_loss_pips"])

        take_profit_pips = None
        take_profit_distance = None
        target_type = str(rm.get("target_type", "")).lower()
        if target_type == "fixed" and rm.get("target_value") not in (None, ""):
            take_profit_pips = float(rm["target_value"])
        elif target_type == "atr" and rm.get("target_value") not in (None, ""):
            mult = float(rm["target_value"])
            take_profit_distance = self._atr_series(work, atr_cache, rm.get("target_atr_period", 14)) * mult
        elif cfg.get("take_profit_pips") not in (None, ""):
            take_profit_pips = float(cfg["take_profit_pips"])

        trailing_stop_distance = None
        ts = rm.get("trailing_stop", {}) or {}
        if ts.get("enabled") and ts.get("value") not in (None, ""):
            mult = float(ts["value"])
            trailing_stop_distance = self._atr_series(work, atr_cache, ts.get("atr_period", 14)) * mult

        breakeven_trigger_r = None
        be = rm.get("break_even", {}) or {}
        if be.get("enabled") and be.get("trigger_r") not in (None, ""):
            try:
                breakeven_trigger_r = max(float(be["trigger_r"]), 0.0)
            except (TypeError, ValueError):
                breakeven_trigger_r = None

        return {
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips,
            "stop_loss_distance": stop_loss_distance,
            "take_profit_distance": take_profit_distance,
            "trailing_stop_distance": trailing_stop_distance,
            "breakeven_trigger_r": breakeven_trigger_r,
        }

    # ------------------------------------------------------------------
    # Public strategy API
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> StrategyResult:
        cfg = self.config
        work = self._build_indicators(df)

        has_visual = bool(cfg.get("entry_conditions"))
        if has_visual:
            long_entry_sig, long_exit_sig, short_entry_sig, short_exit_sig = self._build_visual_signals(work)
        else:
            long_entry = cfg.get("long_entry")
            long_exit = cfg.get("long_exit")
            short_entry = cfg.get("short_entry")
            short_exit = cfg.get("short_exit")
            if not long_entry and not short_entry:
                raise StrategyError("At least one of 'long_entry' or 'short_entry' must be defined.")
            long_entry_sig = safe_eval_bool(work, long_entry, "long_entry") if long_entry else pd.Series(False, index=work.index)
            long_exit_sig = safe_eval_bool(work, long_exit, "long_exit") if long_exit else pd.Series(False, index=work.index)
            short_entry_sig = safe_eval_bool(work, short_entry, "short_entry") if short_entry else pd.Series(False, index=work.index)
            short_exit_sig = safe_eval_bool(work, short_exit, "short_exit") if short_exit else pd.Series(False, index=work.index)

        rm_cfg = cfg.get("risk_management", {}) or {}
        opposite_signal_exit = bool(rm_cfg.get("opposite_signal_exit", True))

        raw_signals = signals_from_conditions(
            work.index, long_entry_sig, long_exit_sig, short_entry_sig, short_exit_sig,
            allow_opposite_signal_flip=opposite_signal_exit,
        )
        signals = self._apply_signal_exits(raw_signals, work)
        signals = self._validate_signals(signals, df)

        risk = self._build_risk_management(work)

        return StrategyResult(
            name=cfg.get("name", "Manual Strategy"),
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=risk["stop_loss_pips"],
            take_profit_pips=risk["take_profit_pips"],
            stop_loss_distance=risk["stop_loss_distance"],
            take_profit_distance=risk["take_profit_distance"],
            trailing_stop_distance=risk["trailing_stop_distance"],
            breakeven_trigger_r=risk["breakeven_trigger_r"],
        )
