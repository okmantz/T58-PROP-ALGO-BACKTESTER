"""
Python Strategy Adapter.

Loads a user-supplied .py file which must define a top-level function:

    def generate_signals(df: pd.DataFrame) -> pd.Series:
        ...  # returns a series of -1/0/1 the same length as df

Optionally the module may also define fixed, whole-backtest exit parameters:
    STOP_LOSS_PIPS = <float>
    TAKE_PROFIT_PIPS = <float>
    STRATEGY_NAME = "<str>"

DYNAMIC (per-trade) STOPS AND TARGETS
--------------------------------------
Many real strategies compute a stop/target that depends on the specific
setup (an ATR multiple, a zone boundary, a swing level) rather than a fixed
pip count. A fixed STOP_LOSS_PIPS cannot express this. To support it without
breaking the single-return-value contract above, the returned signal Series
may carry the following OPTIONAL keys in its `.attrs` dict, each a
Series/array-like of the same length as `df` (raw price units, NaN/absent on
non-entry bars — only the value on the entry bar itself is read):

    signals.attrs["stop_loss_distance"]       -> |entry - stop|
    signals.attrs["take_profit_distance"]     -> |entry - target|
    signals.attrs["trailing_stop_distance"]   -> raw-price trailing distance
    signals.attrs["breakeven_trigger_r"]      -> scalar float (e.g. 1.0 == "+1R")

When present, these take precedence over STOP_LOSS_PIPS/TAKE_PROFIT_PIPS for
the bars where they're defined. This is the ONLY way a Python strategy's own
computed stop/target actually reaches the execution engine — if your
strategy computes a stop or target and never attaches it here, the engine
has no way to know about it and will size/protect the trade using its own
generic fallback instead, silently discarding your intended risk management.

MULTI-TIMEFRAME BIAS FILTERS — A COMMON LOOKAHEAD TRAP
-------------------------------------------------------
If your strategy resamples the input data to a higher timeframe (e.g. a 1H
trend filter for a 15m entry strategy) and then filters it with something
like `htf[htf.index < timestamp]` to get "the last closed HTF bar", THIS IS
LIKELY A LOOKAHEAD BUG. A resampled bar is labeled by its START time, so
`htf.index < timestamp` includes the still-forming CURRENT bar for any
`timestamp` that isn't exactly on the HTF boundary — which was computed
using the full bar's data, including bars later than `timestamp` that
haven't happened yet. Use `app.strategy.mtf.completed_bars()` /
`last_completed_bar()` instead, which correctly requires the bar to have
fully closed (bar_start + timeframe <= timestamp) before using it. This
exact mistake was found in a real uploaded strategy and was, on its own,
responsible for the strategy's entire apparent edge (see app/strategy/mtf.py
docstring for the before/after numbers).

STRATEGIES CANNOT IMPLEMENT THEIR OWN "MAX DAILY LOSSES" GATE
----------------------------------------------------------------
generate_signals(df) is called ONCE, statelessly, over the entire dataset
before any execution or P&L exists. A strategy has no way to know whether
its own earlier trades that day won or lost, so any internal
"stop trading after N daily losses" counter a strategy tries to maintain is
silently a no-op — it can count trades taken, but never trades lost. If you
need a real daily-loss circuit breaker, use the engine's own
RiskConfig-level enforcement (see app/backtest/risk.py) rather than trying
to build one inside generate_signals().

The module is imported in isolation via importlib so a bad/malicious upload
cannot silently corrupt the running app's own modules; execution errors are
caught and surfaced as a clear StrategyError per the spec's requirement that
unsupported/invalid strategy code must fail loudly rather than produce a
silently inaccurate backtest.
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult


class PythonStrategy(Strategy):
    source_type = "python"

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise StrategyError(f"Python strategy file not found: {self.file_path}")
        if self.file_path.suffix != ".py":
            raise StrategyError("Python strategy must be a .py file.")

    def _load_module(self):
        module_name = f"user_strategy_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, self.file_path)
        if spec is None or spec.loader is None:
            raise StrategyError(f"Could not load Python module from {self.file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(f"Error executing uploaded Python strategy: {exc}") from exc
        finally:
            sys.modules.pop(module_name, None)
        return module

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        module = self._load_module()

        if not hasattr(module, "generate_signals"):
            raise StrategyError(
                "Uploaded Python strategy must define a top-level function "
                "`generate_signals(df) -> pd.Series`."
            )

        try:
            raw_signals = module.generate_signals(df.copy())
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(f"Uploaded strategy raised an exception: {exc}") from exc

        # Preserve .attrs (dynamic stop/target info) before any coercion below,
        # since pd.Series(raw_signals, index=...) or a fresh Series does not
        # necessarily carry the original object's .attrs forward.
        source_attrs = getattr(raw_signals, "attrs", {}) or {}

        if not isinstance(raw_signals, pd.Series):
            try:
                raw_signals = pd.Series(raw_signals, index=df.index)
            except Exception as exc:  # noqa: BLE001
                raise StrategyError(
                    f"generate_signals() must return a pandas Series (or array-like) of -1/0/1: {exc}"
                ) from exc

        signals = self._validate_signals(raw_signals, df)

        stop_loss_distance = self._extract_distance_attr(
            source_attrs, "stop_loss_distance", df
        )
        take_profit_distance = self._extract_distance_attr(
            source_attrs, "take_profit_distance", df
        )
        trailing_stop_distance = self._extract_distance_attr(
            source_attrs, "trailing_stop_distance", df
        )
        breakeven_trigger_r = source_attrs.get("breakeven_trigger_r")
        if breakeven_trigger_r is not None:
            try:
                breakeven_trigger_r = float(breakeven_trigger_r)
            except (TypeError, ValueError):
                breakeven_trigger_r = None

        return StrategyResult(
            name=getattr(module, "STRATEGY_NAME", self.file_path.stem),
            source_type=self.source_type,
            signals=signals,
            stop_loss_pips=getattr(module, "STOP_LOSS_PIPS", None),
            take_profit_pips=getattr(module, "TAKE_PROFIT_PIPS", None),
            stop_loss_distance=stop_loss_distance,
            take_profit_distance=take_profit_distance,
            trailing_stop_distance=trailing_stop_distance,
            breakeven_trigger_r=breakeven_trigger_r,
        )

    @staticmethod
    def _extract_distance_attr(source_attrs: dict, key: str, df: pd.DataFrame) -> pd.Series | None:
        """
        Pull an optional per-bar distance series out of the strategy's
        returned signal .attrs, coercing it to a numeric pd.Series aligned
        to df's index. Returns None if the key is absent, empty, or fails
        to coerce cleanly (the strategy falls back to fixed pips / the
        engine's default stop in that case, rather than crashing the run).
        """
        raw = source_attrs.get(key)
        if raw is None:
            return None
        try:
            series = pd.Series(raw)
            if len(series) != len(df):
                raise ValueError(
                    f"'{key}' has length {len(series)}, expected {len(df)} (one value per input row)"
                )
            series.index = df.index
            series = pd.to_numeric(series, errors="coerce")
        except Exception as exc:  # noqa: BLE001
            raise StrategyError(
                f"Uploaded strategy's signals.attrs['{key}'] could not be used: {exc}"
            ) from exc
        # An all-NaN / all-non-positive distance series is equivalent to not
        # providing one at all -- don't pass it through only to have every
        # bar silently ignored downstream.
        if not np.isfinite(series.values).any():
            return None
        return series
