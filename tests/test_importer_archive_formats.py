"""Regression tests for parquet/.zip/.7z market-data import support
(app/data/importer.py). CSV import itself is covered elsewhere -- these
only cover the new extension-dispatch paths."""
import zipfile

import pandas as pd
import pytest

from app.data.importer import import_csv

py7zr = pytest.importorskip("py7zr")
pytest.importorskip("pyarrow")


def _sample_df(n=50):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = [1900.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": 10.0,
    })


def test_import_parquet(tmp_path):
    df = _sample_df()
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    result = import_csv(str(p))
    assert result.is_valid
    assert len(result.dataframe) == len(df)


def test_import_zip_containing_csv(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / "inner.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="inner.csv")
    result = import_csv(str(zip_path))
    assert result.is_valid
    assert len(result.dataframe) == len(df)


def test_import_zip_skips_macos_junk(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / "inner.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="inner.csv")
        zf.writestr("__MACOSX/._inner.csv", b"junk")
        zf.writestr(".DS_Store", b"junk")
    result = import_csv(str(zip_path))
    assert result.is_valid
    assert len(result.dataframe) == len(df)


def test_import_7z_containing_parquet(tmp_path):
    df = _sample_df()
    parquet_path = tmp_path / "inner.parquet"
    df.to_parquet(parquet_path, index=False)
    archive_path = tmp_path / "data.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(str(parquet_path), arcname="inner.parquet")
    result = import_csv(str(archive_path))
    assert result.is_valid
    assert len(result.dataframe) == len(df)


def test_import_7z_vendor_schema_ts_symbol_column(tmp_path):
    """Regression test for a real failure: a futures-data vendor export
    (parquet inside a .7z) using 'ts' for the timestamp column and an
    extra 'symbol' column T58 doesn't otherwise use. Before the 'ts'
    alias + fuzzy-timestamp fallback were added, this failed with
    "Could not identify required column(s): ['timestamp']" even though
    the archive opened and the OHLCV data was perfectly readable."""
    n = 50
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    price = [3200.0 + i * 0.25 for i in range(n)]
    vendor_df = pd.DataFrame({
        "ts": ts, "symbol": "ES.F", "open": price, "high": price,
        "low": price, "close": price, "volume": 100.0,
    })
    parquet_path = tmp_path / "futures_ES.F_1m.parquet"
    vendor_df.to_parquet(parquet_path, index=False)
    archive_path = tmp_path / "ES_7zip.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(str(parquet_path), arcname="futures_ES.F_1m.parquet")
    result = import_csv(str(archive_path))
    assert result.is_valid, result.errors
    assert result.column_mapping["timestamp"] == "ts"
    assert len(result.dataframe) == n


def test_auto_map_columns_fuzzy_timestamp_fallback():
    """A column named something no alias list will ever fully enumerate
    (e.g. a broker's 'bar_start') should still be picked up via the
    fuzzy date/time-name fallback when it's the only candidate."""
    from app.data.importer import _auto_map_columns

    mapping = _auto_map_columns(["candle_time", "open", "high", "low", "close", "volume"])
    assert mapping["timestamp"] == "candle_time"
