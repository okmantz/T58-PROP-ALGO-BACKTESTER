"""
Two-instrument alignment for pairs / relative-value strategies.

This app's engine (app.backtest.engine.run_backtest) is single-instrument:
one OHLCV DataFrame in, one trade list out (see app/portfolio/portfolio.py's
module docstring for the same constraint on the multi-asset side). A real
statistical-pairs strategy needs to compare two instruments' prices bar by
bar, so rather than rewriting the engine into a two-instrument-aware
event loop, this module does the honest, minimal thing: merge the SECOND
instrument's close price into the FIRST instrument's DataFrame as an
ordinary extra column, aligned by timestamp. From the engine's point of
view it is still backtesting one instrument with one extra data column --
which is exactly what it already knows how to do -- while the strategy
itself (app.strategy.manual's "pair_ratio" / "pair_zscore" operand kinds,
or a Python strategy that reads the same column directly) gets genuine
two-instrument context.

Known, explicit limitation: this merge is as-of/forward-filled, not a
strict "both bars closed at the exact same instant" join -- correct for
two instruments on the same or compatible timeframes, but a pair traded
across very different timeframes (e.g. a 1m instrument paired against a
1d instrument) will effectively have the slower series held constant for
long stretches. Fine for the intended use (two similarly-liquid,
similar-timeframe instruments); not a tick-accurate cross-timeframe join.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_PAIR_COLUMN = "pair_close"


class PairDataError(Exception):
    """Raised when two instruments' data cannot be aligned for a pairs strategy."""


def merge_pair_series(
    df: pd.DataFrame,
    pair_df: pd.DataFrame,
    column_name: str = DEFAULT_PAIR_COLUMN,
) -> pd.DataFrame:
    """
    Returns a COPY of `df` with an extra column (`column_name`, default
    "pair_close") holding `pair_df`'s close price as-of each `df` bar's
    timestamp -- i.e. the most recent pair-instrument close known at or
    before that bar, forward-filled. This is a strictly backward-looking
    join: a `df` bar can never see a `pair_df` bar that hadn't printed yet,
    so it introduces no lookahead of its own.

    Bars at the very start of `df` that precede `pair_df`'s first bar get
    NaN (rolling z-score/ratio operands already handle NaN input the same
    way every other warm-up-period indicator in this app does -- they
    simply don't fire until enough history exists).
    """
    if "timestamp" not in df.columns or "timestamp" not in pair_df.columns:
        raise PairDataError("Both DataFrames must have a 'timestamp' column to align on.")
    if "close" not in pair_df.columns:
        raise PairDataError("The pair instrument's DataFrame must have a 'close' column.")

    left = df.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"])
    right = pair_df[["timestamp", "close"]].copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"])
    right = right.sort_values("timestamp").rename(columns={"close": column_name})

    left_sorted = left.sort_values("timestamp")
    merged = pd.merge_asof(left_sorted, right, on="timestamp", direction="backward")
    # merge_asof requires sorted-by-key inputs but does not itself guarantee
    # the ORIGINAL row order is preserved for output -- restore it so every
    # other part of the pipeline (which assumes df's original bar order)
    # keeps working unmodified.
    merged = merged.set_index(left_sorted.index).reindex(left.index)
    return merged
