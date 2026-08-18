import io

from app.data.importer import import_csv


def test_import_valid_csv():
    csv_text = (
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.1000,1.1005,1.0995,1.1002,100\n"
        "2024-01-01 00:05:00,1.1002,1.1008,1.1000,1.1006,120\n"
    )
    result = import_csv(io.StringIO(csv_text))
    assert result.is_valid
    assert len(result.dataframe) == 2
    assert list(result.dataframe.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_import_missing_required_column():
    csv_text = "open,high,low,close\n1.1,1.2,1.0,1.15\n"
    result = import_csv(io.StringIO(csv_text))
    assert not result.is_valid
    assert result.errors


def test_import_drops_bad_ohlc_logic():
    csv_text = (
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.10,1.20,1.05,1.15,100\n"  # valid
        "2024-01-01 00:05:00,1.10,1.00,1.05,1.15,100\n"  # invalid: high < low
    )
    result = import_csv(io.StringIO(csv_text))
    assert result.is_valid
    assert len(result.dataframe) == 1
    assert result.warnings


def test_import_deduplicates_timestamps():
    csv_text = (
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00,1.10,1.20,1.05,1.15,100\n"
        "2024-01-01 00:00:00,1.10,1.20,1.05,1.16,100\n"
    )
    result = import_csv(io.StringIO(csv_text))
    assert result.is_valid
    assert len(result.dataframe) == 1


def test_import_empty_csv():
    result = import_csv(io.StringIO(""))
    assert not result.is_valid
