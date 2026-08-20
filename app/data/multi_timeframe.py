"""
Multi-timeframe merge utility.

Lets a strategy see higher-timeframe context (e.g. a 1h "bias" timeframe,
a 15m "zone" timeframe) alongside the base timeframe it actually trades on
(e.g. 5m "entry"), without changing how indicators or conditions are
written.

The finest (smallest bar-interval) dataframe becomes the "base" timeframe
and keeps its plain column names: 'open', 'high', 'low', 'close', 'volume'.
Every additional, coarser dataframe is merged onto the base timestamps
(using an as-of / backward merge, so no future data ever leaks into an
earlier base bar) and its columns are namespaced with a 'tfNN_' prefix,
where NN is the inferred bar size in minutes, e.g. 'tf60_close',
'tf15_high'. Those prefixed columns can be referenced directly as the
'column' of an indicator, or as an operand in a manual-strategy condition,
exactly like 'close' or 'sma_fast' would be.
"""
from __future__ import annotations

import pandas as pd


def infer_timeframe_minutes(df: pd.DataFrame) -> float:
    """Infer the typical bar spacing of a dataframe, in minutes."""
    ts = pd.to_datetime(df["timestamp"])
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return 0.0
    return float(diffs.median())


def merge_multi_timeframe(
    dataframes: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge 1+ OHLCV dataframes of (possibly) different timeframes into one,
    aligned on the finest (smallest-interval) timeframe.

    Returns (merged_df, labels):
      - merged_df: the base dataframe, plus 'tfNN_*' columns for every
        additional, coarser timeframe that was merged in.
      - labels: a human-readable description of what became the base and
        what became each additional timeframe, in the order merged, e.g.
        ["base (5m)", "tf15 (15m)", "tf60 (60m)"].
    """
    if not dataframes:
        raise ValueError("No dataframes provided.")

    if len(dataframes) == 1:
        df = dataframes[0].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, ["base"]

    ranked = sorted(dataframes, key=infer_timeframe_minutes)

    base = ranked[0].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    base = base.sort_values("timestamp").reset_index(drop=True)

    labels = [f"base ({infer_timeframe_minutes(ranked[0]):.0f}m)"]
    merged = base
    used_prefixes: set[str] = set()

    for htf_df in ranked[1:]:
        minutes = infer_timeframe_minutes(htf_df)
        base_prefix = f"tf{round(minutes)}"
        prefix = base_prefix
        n = 2
        while prefix in used_prefixes:
            prefix = f"{base_prefix}_{n}"
            n += 1
        used_prefixes.add(prefix)

        htf = htf_df.copy()
        htf["timestamp"] = pd.to_datetime(htf["timestamp"])
        htf = htf.sort_values("timestamp").reset_index(drop=True)
        htf = htf.rename(columns={c: f"{prefix}_{c}" for c in htf.columns if c != "timestamp"})

        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            htf,
            on="timestamp",
            direction="backward",
        )
        labels.append(f"{prefix} ({minutes:.0f}m)")

    return merged.reset_index(drop=True), labels
