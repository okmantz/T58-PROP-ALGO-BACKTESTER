"""
Market data import and validation.

Accepts common OHLC(V) market-data formats, including:

- CSV files with headers
- CSV files without headers
- Comma-separated files
- Tab-separated files
- Semicolon-separated files
- Common broker/vendor column names
- Headerless 6-column OHLCV files

Standard internal schema:

    timestamp
    open
    high
    low
    close
    volume
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


STANDARD_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


COLUMN_ALIASES = {
    "timestamp": [
        "timestamp",
        "time",
        "date",
        "datetime",
        "date_time",
        "local time",
        "local_time",
        "datetime utc",
        "date time",
    ],
    "open": [
        "open",
        "o",
        "open price",
        "open_price",
    ],
    "high": [
        "high",
        "h",
        "high price",
        "high_price",
    ],
    "low": [
        "low",
        "l",
        "low price",
        "low_price",
    ],
    "close": [
        "close",
        "c",
        "close price",
        "close_price",
        "adj close",
        "adjusted close",
        "price",
    ],
    "volume": [
        "volume",
        "vol",
        "v",
        "tick volume",
        "tickvol",
        "tick_volume",
    ],
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
        return (
            self.dataframe is not None
            and not any(i.level == "error" for i in self.issues)
        )

    @property
    def errors(self) -> list[str]:
        return [
            i.message
            for i in self.issues
            if i.level == "error"
        ]

    @property
    def warnings(self) -> list[str]:
        return [
            i.message
            for i in self.issues
            if i.level == "warning"
        ]


def _detect_separator(path_or_buffer) -> str:
    """
    Detect the most likely delimiter.

    Supports:
        comma
        tab
        semicolon
        pipe
    """

    try:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
            sample = path_or_buffer.read(8192)

            if isinstance(sample, bytes):
                sample = sample.decode("utf-8-sig", errors="replace")

            path_or_buffer.seek(0)
        else:
            with open(path_or_buffer, "rb") as f:
                sample = f.read(8192).decode(
                    "utf-8-sig",
                    errors="replace",
                )

    except Exception:
        return ","

    candidates = {
        ",": sample.count(","),
        "\t": sample.count("\t"),
        ";": sample.count(";"),
        "|": sample.count("|"),
    }

    separator = max(candidates, key=candidates.get)

    if candidates[separator] == 0:
        return ","

    return separator


def _looks_like_header(row: list[str]) -> bool:
    """
    Determine whether the first parsed row looks like a header.
    """

    normalized = {
        str(value).strip().lower()
        for value in row
    }

    known_names = set()

    for aliases in COLUMN_ALIASES.values():
        known_names.update(aliases)

    # If at least one recognized column name appears,
    # treat the row as a header.
    if normalized.intersection(known_names):
        return True

    # If the first field looks like a date/time value,
    # it is probably data rather than a header.
    if row:
        first = str(row[0]).strip()

        parsed = pd.to_datetime(
            first,
            errors="coerce",
        )

        if not pd.isna(parsed):
            return False

    return True


def _read_raw_file(path_or_buffer) -> pd.DataFrame:
    """
    Read a market-data file while automatically detecting:

    - delimiter
    - header/no-header
    """

    separator = _detect_separator(path_or_buffer)

    # First read the first row without assuming a header.
    try:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)

        first_row = pd.read_csv(
            path_or_buffer,
            sep=separator,
            header=None,
            nrows=1,
            dtype=str,
        )

        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)

    except Exception as exc:
        raise ValueError(f"Could not inspect CSV file: {exc}") from exc

    if first_row.empty:
        raise ValueError("CSV file contains no rows.")

    first_values = [
        str(value).strip()
        for value in first_row.iloc[0].tolist()
    ]

    has_header = _looks_like_header(first_values)

    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)

    if has_header:
        raw = pd.read_csv(
            path_or_buffer,
            sep=separator,
        )
    else:
        raw = pd.read_csv(
            path_or_buffer,
            sep=separator,
            header=None,
        )

        # Handle the common headerless OHLCV layout:
        #
        # timestamp, open, high, low, close, volume
        #
        # This is the format used by the supplied EURUSD15.csv.
        if raw.shape[1] >= 5:
            if raw.shape[1] == 5:
                raw.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            elif raw.shape[1] >= 6:
                raw = raw.iloc[:, :6]
                raw.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]

    return raw


def _auto_map_columns(columns: list[str]) -> dict[str, str]:
    """
    Map raw column names to standard column names.
    """

    lower_map = {
        str(c).lower().strip(): c
        for c in columns
    }

    mapping: dict[str, str] = {}

    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                mapping[standard] = lower_map[alias]
                break

    return mapping


def import_csv(
    path_or_buffer,
    manual_mapping: Optional[dict[str, str]] = None,
) -> ImportResult:
    """
    Import a market-data file.

    Automatically handles common CSV formats and headerless
    OHLCV data.
    """

    issues: list[ValidationIssue] = []

    try:
        raw = _read_raw_file(path_or_buffer)

    except Exception as exc:
        return ImportResult(
            dataframe=None,
            issues=[
                ValidationIssue(
                    "error",
                    f"Could not parse CSV: {exc}",
                )
            ],
        )

    if raw.empty:
        return ImportResult(
            dataframe=None,
            issues=[
                ValidationIssue(
                    "error",
                    "CSV file contains no rows.",
                )
            ],
        )

    mapping = _auto_map_columns(
        list(raw.columns)
    )

    if manual_mapping:
        mapping.update(manual_mapping)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in mapping
    ]

    if missing:
        return ImportResult(
            dataframe=None,
            issues=[
                ValidationIssue(
                    "error",
                    f"Could not identify required column(s): {missing}. "
                    f"Detected columns: {list(raw.columns)}. "
                    f"Expected timestamp/open/high/low/close data.",
                )
            ],
            column_mapping=mapping,
        )

    df = pd.DataFrame()

    for standard_column in STANDARD_COLUMNS:

        if standard_column in mapping:
            df[standard_column] = raw[
                mapping[standard_column]
            ]

        elif standard_column == "volume":
            df[standard_column] = 0.0

    # ---------------------------------------------------------
    # Timestamp validation
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=False,
    )

    bad_ts = int(
        df["timestamp"].isna().sum()
    )

    if bad_ts:
        issues.append(
            ValidationIssue(
                "warning",
                f"Dropped {bad_ts} row(s) with "
                f"unparsable timestamps.",
            )
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    if df.empty:
        return ImportResult(
            dataframe=None,
            issues=issues + [
                ValidationIssue(
                    "error",
                    "No valid rows remain after "
                    "timestamp validation.",
                )
            ],
        )

    # ---------------------------------------------------------
    # Numeric OHLCV validation
    # ---------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    bad_ohlc = int(
        df[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    if bad_ohlc:
        issues.append(
            ValidationIssue(
                "warning",
                f"Dropped {bad_ohlc} row(s) with "
                f"non-numeric OHLC values.",
            )
        )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

    df["volume"] = df["volume"].fillna(
        0.0
    )

    # ---------------------------------------------------------
    # Sort + duplicate handling
    # ---------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    )

    dupes = int(
        df.duplicated(
            subset=["timestamp"]
        ).sum()
    )

    if dupes:
        issues.append(
            ValidationIssue(
                "warning",
                f"Removed {dupes} duplicate "
                f"timestamp row(s) (kept first).",
            )
        )

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )

    # ---------------------------------------------------------
    # OHLC integrity
    # ---------------------------------------------------------

    bad_logic = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    n_bad_logic = int(
        bad_logic.sum()
    )

    if n_bad_logic:
        issues.append(
            ValidationIssue(
                "warning",
                f"Removed {n_bad_logic} row(s) "
                f"that failed OHLC logical integrity "
                f"checks.",
            )
        )

        df = df[~bad_logic]

    if df.empty:
        return ImportResult(
            dataframe=None,
            issues=issues + [
                ValidationIssue(
                    "error",
                    "No valid rows remain after "
                    "integrity checks.",
                )
            ],
        )

    # ---------------------------------------------------------
    # Gap detection
    # ---------------------------------------------------------

    diffs = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    if not diffs.empty:

        median_gap = diffs.median()

        if median_gap > pd.Timedelta(0):

            large_gaps = int(
                (
                    diffs
                    > median_gap * 5
                ).sum()
            )

            if large_gaps:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"Detected {large_gaps} "
                        f"time gap(s) larger than "
                        f"5x the median bar interval "
                        f"(median interval: "
                        f"{median_gap}).",
                    )
                )

    # ---------------------------------------------------------
    # Final cleanup
    # ---------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    return ImportResult(
        dataframe=df,
        issues=issues,
        column_mapping=mapping,
    )


def import_csv_bytes(
    content: bytes,
    manual_mapping: Optional[dict[str, str]] = None,
) -> ImportResult:
    """
    Convenience wrapper for importing raw bytes.
    """

    return import_csv(
        io.BytesIO(content),
        manual_mapping=manual_mapping,
    )
