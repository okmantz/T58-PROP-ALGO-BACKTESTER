"""
Multi-timeframe alignment helpers.

Strategies frequently need "the last COMPLETED higher-timeframe bar" relative
to the current lower-timeframe bar being evaluated -- e.g. a 15m strategy
gating entries on a 1H trend bias. Getting this exactly right requires two
different things that look superficially similar and are easy to conflate:

    1. df.resample(freq)  -- groups rows into bins LABELED BY THEIR START.
       A row labeled "10:00" after resample("1h") covers [10:00, 11:00) and
       is only fully formed once real time reaches 11:00.

    2. htf[htf.index < timestamp]  -- looks like it excludes "the future",
       but only excludes bars whose LABEL is >= timestamp. It does NOT
       exclude a bar whose label is in the past but whose bin hasn't closed
       yet. For any `timestamp` that isn't exactly on an hour boundary
       (i.e. almost every 15m/5m bar), `htf.index < timestamp` INCLUDES the
       still-forming current-hour bar -- which was computed over the FULL
       hour, including bars later than `timestamp` that haven't happened
       yet in a live/real-time sense. That is a genuine lookahead bug: the
       "1H bias" ends up computed partly from the future.

    This exact mistake was found (and is fixed here) in a real uploaded
    strategy (strategy_01_ny_liquidity_fvg): its 1H EMA bias filter used
    `h1[h1.index < timestamp]` and was silently peeking at up to ~59 minutes
    of not-yet-happened price action for every bar that wasn't exactly on
    the hour (~78% of its session bars). Removing the leak flipped the
    strategy from profitable (net_profit +$36k, Sharpe +3.2 over the same
    dataset/config) to a clear loser (net_profit -$12k, Sharpe -2.7) --
    the entire reported edge came from the leak, not a real one.

Use `completed_bars()` below instead of hand-rolling the filter. It is the
one place this logic is implemented and tested, so every strategy that
needs "the last fully-closed HTF candle" gets it right by construction.
"""
from __future__ import annotations

import pandas as pd


def completed_bars(htf: pd.DataFrame, timestamp: pd.Timestamp, freq: str) -> pd.DataFrame:
    """
    Return only the higher-timeframe rows of `htf` (indexed by bar-start
    timestamp, e.g. the output of `df.resample(freq).agg(...)`) that are
    fully CLOSED as of `timestamp` -- i.e. bar_start + freq <= timestamp.

    This is deliberately stricter than `htf.index < timestamp`, which
    admits the still-forming current bar for any `timestamp` that isn't
    exactly on a bar boundary.
    """
    bar_length = pd.tseries.frequencies.to_offset(freq)
    return htf[htf.index + bar_length <= timestamp]


def last_completed_bar(htf: pd.DataFrame, timestamp: pd.Timestamp, freq: str) -> pd.Series | None:
    """Convenience wrapper: the single most recent completed HTF bar, or
    None if no HTF bar has closed yet as of `timestamp`."""
    closed = completed_bars(htf, timestamp, freq)
    if closed.empty:
        return None
    return closed.iloc[-1]
