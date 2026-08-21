"""
Standardized strategy representation.

Regardless of source (manual rule builder, Python, PineScript, MQL5), every
strategy is reduced to a function that consumes an OHLCV DataFrame and
produces a standardized signal series. This lets the backtest engine remain
completely agnostic to strategy origin.

Signal convention:
    1  -> enter/hold long
    -1 -> enter/hold short
    0  -> flat / no position
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


class StrategyError(Exception):
    """Raised when a strategy is invalid or cannot be converted to signals."""


@dataclass
class StrategyResult:
    name: str
    source_type: str  # "manual" | "python" | "pinescript" | "mql5"
    signals: pd.Series  # indexed like the input dataframe, values in {-1, 0, 1}
    stop_loss_pips: float | None = None
    take_profit_pips: float | None = None
    # Optional per-bar stop/target distances in raw price units (e.g. an
    # ATR-multiple stop). When set, these take precedence over the fixed
    # pip-based fields above. Indexed like `signals`.
    stop_loss_distance: pd.Series | None = None
    take_profit_distance: pd.Series | None = None
    # Optional per-bar trailing-stop distance in raw price units (e.g. an
    # ATR-multiple trailing stop, fixed at trade entry).
    trailing_stop_distance: pd.Series | None = None
    # Move the stop to break-even once open profit reaches this multiple
    # of the trade's initial risk (e.g. 1.0 == "+1R").
    breakeven_trigger_r: float | None = None


class Strategy:
    """Base class all strategy adapters implement."""

    source_type: str = "base"

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        raise NotImplementedError

    @staticmethod
    def _validate_signals(signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        if len(signals) != len(df):
            raise StrategyError(
                f"Strategy produced {len(signals)} signals for {len(df)} bars; lengths must match."
            )
        signals = signals.fillna(0).clip(-1, 1).round().astype(int)
        return signals


def signals_from_conditions(
    index: pd.Index,
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
    allow_opposite_signal_flip: bool = True,
) -> pd.Series:
    """
    Shared stateful long/flat/short position loop used by every strategy
    adapter (Manual, PineScript, MQL5) once each has reduced its rules down
    to four boolean condition series. Keeping this in one place guarantees
    all strategy sources behave identically given the same conditions.

    allow_opposite_signal_flip: when True (default, and the only behavior
    prior versions had), an opposite-direction entry signal while a
    position is open immediately reverses it. When False ("Opposite Signal
    Exit" turned off in the Manual Strategy Builder), an opposite entry
    signal is ignored while a position is open; the position can only be
    closed by its own exit conditions, stop loss, take profit, a
    time-based exit, or the max-bars-in-trade limit.
    """
    le, lx = long_entry.values, long_exit.values
    se, sx = short_entry.values, short_exit.values

    position = 0
    out = np.zeros(len(index), dtype=int)
    for i in range(len(index)):
        if position == 0:
            if le[i]:
                position = 1
            elif se[i]:
                position = -1
        elif position == 1:
            if lx[i]:
                position = 0
            elif allow_opposite_signal_flip and se[i]:
                position = -1
        elif position == -1:
            if sx[i]:
                position = 0
            elif allow_opposite_signal_flip and le[i]:
                position = 1
        out[i] = position

    return pd.Series(out, index=index)
