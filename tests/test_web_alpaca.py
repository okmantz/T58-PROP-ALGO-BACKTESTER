from unittest.mock import patch

import pandas as pd
import pytest

from app.data import alpaca_credentials, storage
from app.web import server as server_module
from app.web.server import app


@pytest.fixture(autouse=True)
def isolate_data_dirs(tmp_path, monkeypatch):
    """Same isolation strategy as tests/test_storage.py -- redirects both
    the raw-data dir (where fetched CSVs land) and the Alpaca credentials
    store away from the real ones on disk for the duration of each test."""
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(alpaca_credentials, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(alpaca_credentials, "_try_keyring", lambda: None)
    yield


def _fake_bars_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
        "open": [1.0, 2.0, 3.0, 4.0],
        "high": [1.1, 2.1, 3.1, 4.1],
        "low": [0.9, 1.9, 2.9, 3.9],
        "close": [1.05, 2.05, 3.05, 4.05],
        "volume": [100, 200, 300, 400],
    })


def test_index_shows_alpaca_fetch_panel():
    client = app.test_client()
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Or fetch data from Alpaca" in html
    assert 'name="alpaca_symbols"' in html


def test_fetch_without_keys_redirects_with_error_notice():
    client = app.test_client()
    r = client.post("/data/alpaca/fetch", data={"alpaca_symbols": "AAPL"})
    assert r.status_code == 302
    assert "alpaca_notice_kind=error" in r.headers["Location"]


def test_fetch_without_symbol_redirects_with_error_notice():
    client = app.test_client()
    r = client.post(
        "/data/alpaca/fetch",
        data={"alpaca_api_key": "k", "alpaca_secret_key": "s", "alpaca_symbols": ""},
    )
    assert r.status_code == 302
    assert "alpaca_notice_kind=error" in r.headers["Location"]


def test_successful_fetch_saves_csv_and_appears_in_dataset_list():
    client = app.test_client()
    with patch.object(server_module, "fetch_bars", return_value=_fake_bars_df()):
        r = client.post("/data/alpaca/fetch", data={
            "alpaca_api_key": "AKFAKE", "alpaca_secret_key": "SECFAKE",
            "alpaca_symbols": "TESTSYM", "alpaca_asset_class": "Stock",
            "alpaca_timeframe": "1Day", "alpaca_start": "2026-01-01", "alpaca_end": "2026-01-05",
            "alpaca_feed": "iex", "alpaca_adjustment": "raw",
        })
    assert r.status_code == 302
    assert "alpaca_notice_kind=success" in r.headers["Location"]

    names = [ds.name for ds in storage.list_stored_datasets()]
    assert any("TESTSYM" in n for n in names)


def test_save_keys_checkbox_persists_credentials_for_next_fetch():
    client = app.test_client()
    assert alpaca_credentials.has_saved_credentials() is False

    with patch.object(server_module, "fetch_bars", return_value=_fake_bars_df()):
        client.post("/data/alpaca/fetch", data={
            "alpaca_api_key": "AKSAVE", "alpaca_secret_key": "SECSAVE",
            "alpaca_save_keys": "on", "alpaca_symbols": "SAVETEST",
            "alpaca_asset_class": "Stock", "alpaca_timeframe": "1Day",
            "alpaca_start": "2026-01-01", "alpaca_end": "2026-01-05",
        })

    assert alpaca_credentials.has_saved_credentials() is True
    creds = alpaca_credentials.load_credentials()
    assert creds.api_key == "AKSAVE"


def test_blank_keys_reuse_previously_saved_credentials():
    alpaca_credentials.save_credentials("AKREUSE", "SECREUSE")
    client = app.test_client()

    with patch.object(server_module, "fetch_bars", return_value=_fake_bars_df()) as mock_fetch:
        client.post("/data/alpaca/fetch", data={
            "alpaca_symbols": "REUSETEST", "alpaca_asset_class": "Stock",
            "alpaca_timeframe": "1Day", "alpaca_start": "2026-01-01", "alpaca_end": "2026-01-05",
        })

    called_args = mock_fetch.call_args.args
    assert called_args[0] == "AKREUSE"
    assert called_args[1] == "SECREUSE"


def test_forget_keys_route_clears_saved_credentials():
    alpaca_credentials.save_credentials("k", "s")
    client = app.test_client()
    r = client.post("/data/alpaca/forget")
    assert r.status_code == 302
    assert alpaca_credentials.has_saved_credentials() is False


def test_fetch_error_from_alpaca_is_surfaced_in_notice():
    from app.data.alpaca_source import AlpacaFetchError

    client = app.test_client()
    with patch.object(server_module, "fetch_bars", side_effect=AlpacaFetchError("bad symbol")):
        r = client.post("/data/alpaca/fetch", data={
            "alpaca_api_key": "k", "alpaca_secret_key": "s", "alpaca_symbols": "BADSYM",
            "alpaca_asset_class": "Stock", "alpaca_timeframe": "1Day",
            "alpaca_start": "2026-01-01", "alpaca_end": "2026-01-05",
        })
    assert "alpaca_notice_kind=error" in r.headers["Location"]
    assert "bad+symbol" in r.headers["Location"] or "bad%20symbol" in r.headers["Location"]
