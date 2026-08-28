from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data import alpaca_source


@pytest.fixture(autouse=True)
def isolate_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(alpaca_source, "get_raw_data_dir", lambda: tmp_path)
    yield


def _fake_bars_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume": [1000, 1100, 1200],
    })


def test_normalize_bars_selects_and_sorts_standard_columns():
    raw = _fake_bars_df().sample(frac=1, random_state=1)  # shuffled order
    out = alpaca_source._normalize_bars(raw, "AAPL")
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out["timestamp"].is_monotonic_increasing


def test_normalize_bars_raises_on_empty():
    with pytest.raises(alpaca_source.AlpacaFetchError):
        alpaca_source._normalize_bars(pd.DataFrame(), "AAPL")


def test_normalize_bars_raises_on_missing_columns():
    with pytest.raises(alpaca_source.AlpacaFetchError):
        alpaca_source._normalize_bars(pd.DataFrame({"timestamp": [1], "open": [1]}), "AAPL")


def test_timeframe_object_rejects_unknown_label():
    with pytest.raises(alpaca_source.AlpacaFetchError):
        alpaca_source._timeframe_object("3Weeks")


def test_save_bars_as_csv_writes_into_symbol_subfolder(tmp_path):
    df = _fake_bars_df()
    dest = alpaca_source.save_bars_as_csv(df, "eurusd", "1Hour")
    assert dest.exists()
    assert dest.parent.name == "EURUSD"
    assert dest.name == "EURUSD_1Hour_alpaca.csv"
    reloaded = pd.read_csv(dest)
    assert len(reloaded) == len(df)


def test_save_bars_as_csv_sanitizes_unsafe_symbol_characters():
    df = _fake_bars_df()
    dest = alpaca_source.save_bars_as_csv(df, "BTC/USD", "1Day")
    assert dest.parent.name == "BTCUSD"


def test_fetch_stock_bars_calls_alpaca_client_with_expected_request():
    fake_client_cls = MagicMock()
    fake_client_instance = fake_client_cls.return_value
    fake_client_instance.get_stock_bars.return_value.df = _fake_bars_df()

    with patch("alpaca.data.historical.StockHistoricalDataClient", fake_client_cls):
        df = alpaca_source.fetch_stock_bars(
            "key", "secret", "AAPL", "1Day", "2026-01-01", "2026-01-05",
        )

    fake_client_cls.assert_called_once_with("key", "secret")
    assert fake_client_instance.get_stock_bars.called
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_bars_dispatches_to_crypto_for_crypto_asset_class():
    with patch.object(alpaca_source, "fetch_crypto_bars", return_value="crypto-result") as mock_crypto, \
         patch.object(alpaca_source, "fetch_stock_bars", return_value="stock-result") as mock_stock:
        result = alpaca_source.fetch_bars(
            "key", "secret", "BTC/USD", "Crypto", "1Day", "2026-01-01", "2026-01-05",
        )
    assert result == "crypto-result"
    mock_crypto.assert_called_once()
    mock_stock.assert_not_called()


def test_fetch_bars_dispatches_to_stock_for_stock_asset_class():
    with patch.object(alpaca_source, "fetch_crypto_bars", return_value="crypto-result") as mock_crypto, \
         patch.object(alpaca_source, "fetch_stock_bars", return_value="stock-result") as mock_stock:
        result = alpaca_source.fetch_bars(
            "key", "secret", "AAPL", "Stock", "1Day", "2026-01-01", "2026-01-05",
        )
    assert result == "stock-result"
    mock_stock.assert_called_once()
    mock_crypto.assert_not_called()


def test_require_alpaca_raises_friendly_error_when_not_installed():
    with patch.dict("sys.modules", {"alpaca": None}):
        with pytest.raises(alpaca_source.AlpacaImportError):
            alpaca_source._require_alpaca()
