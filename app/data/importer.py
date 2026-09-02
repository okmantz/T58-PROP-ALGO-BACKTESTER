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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
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
        "ts",
        "time",
        "date",
        "datetime",
        "date_time",
        "dt",
        "bar_time",
        "bartime",
        "period",
        "local time",
        "local_time",
        "datetime utc",
        "date time",
        "date (utc)",
        "time (utc)",
        "utc",
        "utc time",
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


SUPPORTED_ARCHIVE_EXTENSIONS = {".zip", ".7z"}
SUPPORTED_DATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet"}


def _name_of(path_or_buffer) -> str:
    """Best-effort filename for a path, Path, or file-like/buffer object
    (Tkinter/Flask both hand this function open file handles or werkzeug
    FileStorage objects, not always plain path strings)."""
    if isinstance(path_or_buffer, (str, Path)):
        return str(path_or_buffer)
    return str(getattr(path_or_buffer, "name", "") or getattr(path_or_buffer, "filename", "") or "")


def _extension_of(path_or_buffer) -> str:
    return Path(_name_of(path_or_buffer)).suffix.lower()


def _bytes_of(path_or_buffer) -> bytes:
    if isinstance(path_or_buffer, (str, Path)):
        return Path(path_or_buffer).read_bytes()
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    data = path_or_buffer.read()
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    return data


def _pick_data_member(names: list[str]) -> str:
    """Given the member names inside a zip/7z archive, pick the one that's
    actually the market-data file -- skips directories, macOS junk
    (__MACOSX/.DS_Store), and anything without a recognized data
    extension. If more than one candidate remains, the pattern in
    practice (a vendor export zip) is one real data file plus small
    readme/metadata files, so the LARGEST-by-name-heuristic candidate
    isn't knowable from names alone -- callers pass this list already
    filtered to real candidates and this just picks the first, but if
    several match we prefer the one that looks most like a data export
    (contains a digit, e.g. an OHLCV period suffix) over a generic name
    like 'data.csv'.
    """
    candidates = [
        n for n in names
        if not n.endswith("/") and "__MACOSX" not in n and not Path(n).name.startswith(".")
        and Path(n).suffix.lower() in SUPPORTED_DATA_EXTENSIONS
    ]
    if not candidates:
        raise ValueError(
            f"No .csv/.tsv/.txt/.parquet file found inside the archive (contents: {names[:10]})."
        )
    if len(candidates) == 1:
        return candidates[0]
    with_digit = [n for n in candidates if any(ch.isdigit() for ch in Path(n).stem)]
    return sorted(with_digit or candidates, key=len)[0]


def _flatten_parquet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens a MultiIndex column structure into plain strings.

    Parquet files exported from things like a multi-ticker yfinance/
    pandas dump (columns like ('Close', 'AAPL')) or a pivoted OHLCV
    export commonly keep a 2-level column MultiIndex. _auto_map_columns
    only ever looks at plain string column names, so a MultiIndex column
    set would otherwise never match any alias and the import would fail
    with a confusing "could not identify required column(s)" error even
    though the data is perfectly good OHLCV data.

    If every column shares the exact same non-field-name level (a
    single-instrument export, e.g. every column's second level is
    "XAUUSD"), that level is dropped entirely rather than appended --
    it's pure noise for a single-instrument backtester and would
    otherwise stop "Close_XAUUSD" from matching the "close" alias.
    Only when columns genuinely differ (a real multi-ticker file) are
    the levels joined with "_", field name first, so the existing
    lower-case/substring alias matching in _auto_map_columns still has a
    fighting chance of recognizing e.g. "close" inside "close_aapl".
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        n_levels = df.columns.nlevels
        drop_levels = [
            lvl for lvl in range(1, n_levels)
            if df.columns.get_level_values(lvl).nunique() <= 1
        ]
        flat_index = df.columns.droplevel(drop_levels) if drop_levels else df.columns
        if isinstance(flat_index, pd.MultiIndex):
            df.columns = [
                "_".join(str(level) for level in col if str(level) and str(level).lower() != "nan")
                for col in flat_index.to_flat_index()
            ]
        else:
            df.columns = [str(c) for c in flat_index]
    return df


def _promote_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Turns a DatetimeIndex (or an index that's clearly a timestamp, e.g.
    named 'Date'/'Datetime'/'time') into a real 'timestamp' column.

    Parquet, unlike CSV, very often carries the bar timestamp as the
    DataFrame's index rather than as a plain column -- this is the default
    for anything written via `df.to_parquet()` off a time-indexed frame
    (pandas-datareader/yfinance-style exports, most quant research
    pipelines). Without this, _auto_map_columns would never find a
    timestamp column at all since it only ever looks at df.columns.
    """
    index_name = str(df.index.name).strip().lower() if df.index.name is not None else ""
    looks_like_time_index = (
        isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex))
        or index_name in ("date", "datetime", "timestamp", "time", "index", "gmt time", "ts", "dt")
    )
    if looks_like_time_index and not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
        # reset_index() names an unnamed DatetimeIndex "index" -- give it
        # an actually-recognizable name so _auto_map_columns's exact-alias
        # pass (not just its fuzzy fallback) can find it.
        first_col = df.columns[0]
        if str(first_col).strip().lower() in ("index", ""):
            df = df.rename(columns={first_col: "timestamp"})
    return df


def _read_parquet_ohlcv(path_or_buffer) -> pd.DataFrame:
    """Reads a .parquet file OR a partitioned parquet directory (a folder
    of part-*.parquet files, as written by Spark/Dask/most data-lake
    pipelines) into a plain OHLCV-shaped DataFrame.

    Handles the ways real-world parquet exports differ from this app's
    original single-flat-file assumption:
      - a directory of partitioned parquet parts instead of one file
      - the timestamp living in the index instead of a column
      - a MultiIndex column structure (multi-ticker exports)
    Epoch-integer timestamp columns (seconds/ms/us/ns since epoch, common
    in parquet since it has no universal "the timestamp column" marker)
    are handled later in import_csv's timestamp-validation step via
    _coerce_timestamp_series, once the timestamp column has actually been
    identified -- not here, since at this point it isn't picked out yet.
    """
    try:
        if isinstance(path_or_buffer, (str, Path)) and Path(path_or_buffer).is_dir():
            df = pd.read_parquet(path_or_buffer)
        else:
            df = pd.read_parquet(io.BytesIO(_bytes_of(path_or_buffer)))
    except ImportError as exc:
        raise ValueError(
            "Reading .parquet files requires the 'pyarrow' package (pip install pyarrow)."
        ) from exc
    df = _flatten_parquet_columns(df)
    df = _promote_datetime_index(df)
    return df


def _read_member_bytes(member_name: str, member_bytes: bytes) -> pd.DataFrame:
    ext = Path(member_name).suffix.lower()
    if ext == ".parquet":
        return _read_parquet_ohlcv(io.BytesIO(member_bytes))
    # csv/tsv/txt member -- reuse the normal CSV path (delimiter/header
    # detection) on the extracted bytes.
    return _read_csv_like(io.BytesIO(member_bytes))


def _read_zip(path_or_buffer) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(_bytes_of(path_or_buffer))) as zf:
        member = _pick_data_member(zf.namelist())
        return _read_member_bytes(member, zf.read(member))


def _read_7z(path_or_buffer) -> pd.DataFrame:
    try:
        import py7zr
    except ImportError as exc:
        raise ValueError(
            "Reading .7z files requires the 'py7zr' package (pip install py7zr)."
        ) from exc
    import tempfile
    with py7zr.SevenZipFile(io.BytesIO(_bytes_of(path_or_buffer)), mode="r") as archive:
        names = archive.getnames()
        member = _pick_data_member(names)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive.extract(path=tmp_dir, targets=[member])
            extracted_path = Path(tmp_dir) / member
            return _read_member_bytes(member, extracted_path.read_bytes())


def _read_raw_file(path_or_buffer) -> pd.DataFrame:
    """
    Read a market-data file, dispatching on extension first:

    - a directory              -> treated as a partitioned parquet dataset
                                   (a folder of part-*.parquet files) if it
                                   contains any .parquet files; pd.read_parquet
                                   reads the whole partition set as one frame.
    - .parquet             -> pd.read_parquet directly (already tabular,
                               no delimiter/header guessing needed)
    - .zip / .7z            -> opens the archive and reads whichever member
                               inside it looks like the actual OHLCV data
                               file (.csv/.tsv/.txt/.parquet), skipping
                               folders and OS junk like __MACOSX/.DS_Store
    - anything else (.csv, .tsv, .txt, or no extension at all -- e.g. a
      Tkinter/Flask file handle that doesn't expose one) falls through to
      the existing delimiter/header-detecting CSV reader below.
    """
    if isinstance(path_or_buffer, (str, Path)) and Path(path_or_buffer).is_dir():
        directory = Path(path_or_buffer)
        if any(directory.rglob("*.parquet")):
            return _read_parquet_ohlcv(directory)
        raise ValueError(f"'{directory}' is a directory with no .parquet files inside it.")
    ext = _extension_of(path_or_buffer)
    if ext == ".parquet":
        return _read_parquet_ohlcv(path_or_buffer)
    if ext == ".zip":
        return _read_zip(path_or_buffer)
    if ext == ".7z":
        return _read_7z(path_or_buffer)
    return _read_csv_like(path_or_buffer)


def _read_csv_like(path_or_buffer) -> pd.DataFrame:
    """
    Read a market-data file while automatically detecting:

    - delimiter
    - header/no-header
    \"\"\" - see _read_raw_file's docstring for the extension dispatch above this."""
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


def _auto_map_columns(columns: list[str], raw: "pd.DataFrame | None" = None) -> dict[str, str]:
    """
    Map raw column names to standard column names.

    Falls back to fuzzy timestamp detection when no exact alias matches --
    real vendor/broker exports use all kinds of naming ("Gmt time",
    "Timestamp (ms)", "bar_start", "trade_date") that a fixed alias list
    can never fully enumerate. The fallback only fires for the timestamp
    column (the one Owen actually hit -- a "ts" column with no exact
    alias) and only when the exact-match pass found nothing, so a file
    that already matches cleanly is never second-guessed.
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

    if "timestamp" not in mapping:
        already_used = set(mapping.values())
        candidates = [
            lower_map[key] for key in lower_map
            if lower_map[key] not in already_used
            and ("date" in key or "time" in key or key in ("ts", "dt"))
        ]
        if len(candidates) == 1:
            mapping["timestamp"] = candidates[0]
        elif len(candidates) > 1 and raw is not None:
            # Prefer whichever candidate pandas can actually parse as a
            # datetime on a real sample of the data, rather than guessing
            # from the name alone.
            for cand in candidates:
                try:
                    parsed = pd.to_datetime(raw[cand].head(20), errors="coerce")
                    if parsed.notna().sum() >= max(1, int(len(parsed) * 0.8)):
                        mapping["timestamp"] = cand
                        break
                except Exception:
                    continue

    return mapping


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    """Parses a raw timestamp column/index into real datetimes, handling
    epoch-integer timestamps as well as ordinary date strings.

    Parquet has no universal marker for "this column is a datetime
    stored as an epoch integer" the way a CSV's ISO-8601 string does --
    plenty of real exports (MetaTrader, some data-lake pipelines, a
    DataFrame that got cast to int64 before saving) store the bar time as
    plain seconds/milliseconds/microseconds/nanoseconds since epoch.
    Blindly calling pd.to_datetime() on an int64 column assumes
    nanoseconds and silently produces nonsense dates (e.g. a 2024
    epoch-seconds value parsed as nanoseconds lands in 1970) instead of
    erroring, so it has to be handled explicitly rather than left to the
    generic parser below.

    Magnitude bands below are picked so any date from roughly 2001-2286
    is classified correctly regardless of which unit it's actually in --
    there's no ambiguity between the bands for real-world trading data.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=False)

    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna()
        if not sample.empty:
            magnitude = float(sample.abs().median())
            if magnitude > 1e17:
                unit = "ns"
            elif magnitude > 1e14:
                unit = "us"
            elif magnitude > 1e11:
                unit = "ms"
            elif magnitude > 1e8:
                unit = "s"
            else:
                unit = None  # too small to be a plausible epoch timestamp
            if unit is not None:
                parsed = pd.to_datetime(series, unit=unit, errors="coerce", utc=False)
                # Sanity check: a real epoch value converts to a normal
                # calendar year. If it doesn't, this wasn't actually an
                # epoch timestamp (e.g. a numeric bar index) -- fall
                # through to the generic parser instead of returning
                # garbage dates.
                valid = parsed.dropna()
                if not valid.empty and valid.dt.year.between(1990, 2200).mean() > 0.9:
                    return parsed

    return pd.to_datetime(series, errors="coerce", utc=False)


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
        list(raw.columns), raw
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

    df["timestamp"] = _coerce_timestamp_series(df["timestamp"])

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
    # Split / bad-tick artifact detection
    #
    # The single most common way a backtest lies to you: an unadjusted
    # stock split (or a bad print) creates a single-bar move of hundreds or
    # thousands of percent that no real trade could have captured, and any
    # strategy that happens to hold through it "passes" on a fabricated
    # gain. Flag any single-bar return this extreme so it gets a manual
    # eyeball before the backtest results are trusted.
    # ---------------------------------------------------------

    bar_returns = df["close"].pct_change().abs()
    SUSPICIOUS_BAR_RETURN = 0.20  # 20% in one bar
    suspicious = bar_returns[bar_returns > SUSPICIOUS_BAR_RETURN]

    if not suspicious.empty:
        worst_idx = suspicious.idxmax()
        worst_pct = suspicious.max() * 100
        worst_ts = df.loc[worst_idx, "timestamp"]
        issues.append(
            ValidationIssue(
                "warning",
                f"Detected {len(suspicious)} single-bar move(s) over "
                f"{SUSPICIOUS_BAR_RETURN * 100:.0f}% (largest: {worst_pct:,.1f}% "
                f"at {worst_ts}). This is the classic signature of an "
                "unadjusted stock split or a bad tick, not a real tradeable "
                "move — eyeball this bar before trusting any backtest that "
                "profits from it.",
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
