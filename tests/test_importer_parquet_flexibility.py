"""Regression tests for the flexible-parquet import support in
app/data/importer.py: real-world parquet exports rarely look like a
plain flat file with a "timestamp" column and float OHLCV columns the
way this app's CSV path always assumed. These cover the specific shapes
that used to make a perfectly good parquet file fail to import:

  - the bar time living in a DatetimeIndex instead of a column
  - epoch-integer timestamps (seconds/milliseconds since epoch)
  - a MultiIndex column structure from a single-instrument export
  - a genuine multi-ticker MultiIndex export (should not crash, and
    should still expose per-ticker columns for manual mapping)
  - a partitioned parquet directory (multiple part-*.parquet files)
"""
import numpy as np
import pandas as pd
import pytest

from app.data.importer import _read_raw_file, import_csv

pytest.importorskip("pyarrow")


def _ohlcv_frame(idx, seed=0):
    rng = np.random.default_rng(seed)
    base = 1900.0 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    return pd.DataFrame({
        "Open": base, "High": base + 0.5, "Low": base - 0.5, "Close": base, "Volume": 10.0,
    }, index=idx)


def test_datetime_index_promoted_to_timestamp_column(tmp_path):
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", name="Date")
    df = _ohlcv_frame(idx)
    p = tmp_path / "indexed.parquet"
    df.to_parquet(p)

    result = import_csv(str(p))

    assert result.is_valid, result.issues
    assert len(result.dataframe) == 200
    assert pd.api.types.is_datetime64_any_dtype(result.dataframe["timestamp"])
    assert result.dataframe["timestamp"].iloc[0] == idx[0]


def test_epoch_seconds_timestamp_column(tmp_path):
    idx = pd.date_range("2024-01-01", periods=100, freq="1h")
    epoch_s = (idx.view("int64") // 10**6)  # this pandas build's date_range is us-resolution
    df = pd.DataFrame({
        "timestamp": epoch_s, "open": 1900.0, "high": 1901.0, "low": 1899.0, "close": 1900.5, "volume": 5.0,
    })
    p = tmp_path / "epoch_s.parquet"
    df.to_parquet(p, index=False)

    result = import_csv(str(p))

    assert result.is_valid, result.issues
    assert result.dataframe["timestamp"].iloc[0] == idx[0]
    assert result.dataframe["timestamp"].iloc[-1] == idx[-1]


def test_epoch_milliseconds_timestamp_column(tmp_path):
    idx = pd.date_range("2024-01-01", periods=100, freq="1h")
    epoch_ms = (idx.view("int64") // 10**3)
    df = pd.DataFrame({
        "timestamp": epoch_ms, "open": 1900.0, "high": 1901.0, "low": 1899.0, "close": 1900.5, "volume": 5.0,
    })
    p = tmp_path / "epoch_ms.parquet"
    df.to_parquet(p, index=False)

    result = import_csv(str(p))

    assert result.is_valid, result.issues
    assert result.dataframe["timestamp"].iloc[0] == idx[0]


def test_single_instrument_multiindex_columns_flatten_cleanly(tmp_path):
    idx = pd.date_range("2024-01-01", periods=150, freq="15min", name="Date")
    base = _ohlcv_frame(idx)
    base.columns = pd.MultiIndex.from_product([base.columns, ["XAUUSD"]])
    p = tmp_path / "single_ticker_multiindex.parquet"
    base.to_parquet(p)

    result = import_csv(str(p))

    # The redundant single-ticker level should be dropped entirely so
    # "Open_XAUUSD" doesn't fail to match the "open" alias.
    assert result.is_valid, result.issues
    assert len(result.dataframe) == 150


def test_multi_ticker_multiindex_does_not_crash_and_exposes_columns(tmp_path):
    idx = pd.date_range("2024-01-01", periods=50, freq="15min")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["XAUUSD", "EURUSD"]])
    df = pd.DataFrame(np.random.default_rng(1).random((50, 10)) + 100, columns=cols, index=idx)
    p = tmp_path / "multi_ticker.parquet"
    df.to_parquet(p)

    raw = _read_raw_file(str(p))

    # Genuinely ambiguous (which ticker?) -- must not crash, and must
    # produce plain flattened column names a person could manually map.
    assert not isinstance(raw.columns, pd.MultiIndex)
    assert "Open_XAUUSD" in raw.columns and "Open_EURUSD" in raw.columns


def test_partitioned_parquet_directory(tmp_path):
    idx = pd.date_range("2024-01-01", periods=200, freq="15min")
    epoch_s = (idx.view("int64") // 10**6)
    df = pd.DataFrame({
        "timestamp": epoch_s, "open": 1900.0, "high": 1901.0, "low": 1899.0, "close": 1900.5, "volume": 5.0,
    })
    part_dir = tmp_path / "partitioned"
    part_dir.mkdir()
    df.iloc[:100].to_parquet(part_dir / "part-0.parquet", index=False)
    df.iloc[100:].to_parquet(part_dir / "part-1.parquet", index=False)

    result = import_csv(str(part_dir))

    assert result.is_valid, result.issues
    assert len(result.dataframe) == 200


def test_directory_with_no_parquet_files_raises_clear_error(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("not a data file")

    result = import_csv(str(empty_dir))

    assert not result.is_valid
    assert any("no .parquet files" in issue.message for issue in result.issues)
