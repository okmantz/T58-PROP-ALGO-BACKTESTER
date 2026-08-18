"""
Market data import and validation.

Accepts CSV files containing OHLC(V) time series data, auto-detects common
column naming conventions, normalizes them into a standard schema, and
validates timestamps / OHLC integrity / duplicates / gaps before the data
is allowed to enter the backtest engine.

Standard internal schema (columns, in order):
    timestamp (pandas.Timestamp, UTC-naive, sorted ascending)
    open, high, low, close (float)
    volume (float, optional -> filled with 0.0 if missing)
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

STANDARD_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Common aliases seen across broker / vendor CSV exports.
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "datetime", "date_time", "local time"],
    "open": ["open", "o", "open price"],
    "high": ["high", "h", "high price"],
    "low": ["low", "l", "low price"],
    "close": ["close", "c", "close price", "adj close", "price"],
    "volume": ["volume", "vol", "v", "tick volume", "tickvol"],
}


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str


@dataclass
class ImportResult:
    dataframe: Optional[pd.DataFrame]
    issues: list[ValidationIssue] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.dataframe is not None and not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[str]:
        return [i.message for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [i.message for i in self.issues if i.level == "warning"]


def _auto_map_columns(columns: list[str]) -> dict[str, str]:
    """Map raw CSV column names -> standard column names."""
    lower_map = {c.lower().strip(): c for c in columns}
    mapping: dict[str, str] = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                mapping[standard] = lower_map[alias]
                break
    return mapping


def import_csv(path_or_buffer, manual_mapping: Optional[dict[str, str]] = None) -> ImportResult:
    """
    Import a CSV file (path or file-like object) of OHLC data.

    manual_mapping: optional dict of {standard_column: raw_csv_column_name}
    to override/complete auto-detection when headers don't match known
    aliases (e.g. {"timestamp": "Local time", "close": "Bid"}).
    """
    issues: list[ValidationIssue] = []
    try:
        raw = pd.read_csv(path_or_buffer)
    except Exception as exc:  # noqa: BLE001
        return ImportResult(dataframe=None, issues=[ValidationIssue("error", f"Could not parse CSV: {exc}")])

    if raw.empty:
        return ImportResult(dataframe=None, issues=[ValidationIssue("error", "CSV file contains no rows.")])

    mapping = _auto_map_columns(list(raw.columns))
    if manual_mapping:
        mapping.update(manual_mapping)

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [r for r in required if r not in mapping]
    if missing:
        return ImportResult(
            dataframe=None,
            issues=[ValidationIssue(
                "error",
                f"Could not identify required column(s): {missing}. "
                f"Detected columns: {list(raw.columns)}. Provide manual_mapping to resolve.",
            )],
            column_mapping=mapping,
        )

    df = pd.DataFrame()
    for std_col in STANDARD_COLUMNS:
        if std_col in mapping:
            df[std_col] = raw[mapping[std_col]]
        elif std_col == "volume":
            df[std_col] = 0.0

    # --- Timestamp validation ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        issues.append(ValidationIssue("warning", f"Dropped {bad_ts} row(s) with unparsable timestamps."))
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        return ImportResult(dataframe=None, issues=issues + [ValidationIssue("error", "No valid rows remain after timestamp validation.")])

    # --- Numeric OHLCV validation ---
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_ohlc = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    if bad_ohlc:
        issues.append(ValidationIssue("warning", f"Dropped {bad_ohlc} row(s) with non-numeric OHLC values."))
        df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)

    # --- Sort + duplicate handling ---
    df = df.sort_values("timestamp")
    dupes = df.duplicated(subset=["timestamp"]).sum()
    if dupes:
        issues.append(ValidationIssue("warning", f"Removed {dupes} duplicate timestamp row(s) (kept first)."))
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # --- OHLC logical integrity (high >= low, high >= open/close, low <= open/close) ---
    bad_logic = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    n_bad_logic = int(bad_logic.sum())
    if n_bad_logic:
        issues.append(ValidationIssue(
            "warning",
            f"Removed {n_bad_logic} row(s) that failed OHLC logical integrity checks (e.g. high < low).",
        ))
        df = df[~bad_logic]

    if df.empty:
        return ImportResult(dataframe=None, issues=issues + [ValidationIssue("error", "No valid rows remain after integrity checks.")])

    # --- Gap detection (informational only) ---
    diffs = df["timestamp"].diff().dropna()
    if not diffs.empty:
        median_gap = diffs.median()
        large_gaps = int((diffs > median_gap * 5).sum())
        if large_gaps:
            issues.append(ValidationIssue(
                "warning",
                f"Detected {large_gaps} time gap(s) larger than 5x the median bar interval "
                f"(median interval: {median_gap}).",
            ))

    df = df.reset_index(drop=True)
    return ImportResult(dataframe=df, issues=issues, column_mapping=mapping)


def import_csv_bytes(content: bytes, manual_mapping: Optional[dict[str, str]] = None) -> ImportResult:
    """Convenience wrapper for importing from raw bytes (e.g. uploaded file)."""
    return import_csv(io.BytesIO(content), manual_mapping=manual_mapping)
