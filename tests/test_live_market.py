"""Tests for app.web.live_market -- covers the pieces that don't require a
real MT5 terminal or live Alpaca credentials (replay cursor behavior and
trade-marker construction from the journal)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.forward_test.journal import ForwardTestJournal
from app.web import live_market


def _write_dataset(tmp_path, monkeypatch, name="TEST/TEST1.csv", n=50):
    monkeypatch.setattr("app.web.live_market.get_raw_data_dir", lambda: tmp_path)
    dataset_path = tmp_path / name
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n):
        rows.append([start + pd.Timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 10 + i])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(dataset_path, index=False)
    return name


@pytest.fixture(autouse=True)
def _reset_replay_cache():
    live_market._replay_cache.clear()
    yield
    live_market._replay_cache.clear()


def test_mt5_status_reports_unavailable_in_this_sandbox():
    status = live_market.mt5_status()
    assert status["available"] is False
    assert status["connected"] is False


def test_fetch_mt5_bars_returns_empty_when_unavailable():
    assert live_market.fetch_mt5_bars("XAUUSD", 15) == []


def test_fetch_alpaca_bars_returns_empty_without_saved_credentials(monkeypatch):
    monkeypatch.setattr("app.data.alpaca_credentials.load_credentials", lambda: None)
    assert live_market.fetch_alpaca_bars("AAPL", "Stock", 15) == []


def test_fetch_alpaca_bars_with_real_credentials_object_does_not_crash(monkeypatch):
    """Regression test: load_credentials() returns an AlpacaCredentials
    dataclass instance, not a tuple -- subscripting it (creds[0]) used to
    raise TypeError and surface as an Internal Server Error on the Live
    Market page for anyone with saved Alpaca keys."""
    from app.data.alpaca_credentials import AlpacaCredentials

    monkeypatch.setattr(
        "app.data.alpaca_credentials.load_credentials",
        lambda: AlpacaCredentials(api_key="fake", secret_key="fake"),
    )
    # Network call will fail (fake keys) -- the point is that it fails
    # gracefully (returns []) rather than raising before it even tries.
    assert live_market.fetch_alpaca_bars("AAPL", "Stock", 15) == []


def test_replay_seeds_from_the_end_of_the_dataset(tmp_path, monkeypatch):
    name = _write_dataset(tmp_path, monkeypatch, n=400)
    bars, finished = live_market.fetch_replay_bars(name, advance=False)
    assert not finished
    assert len(bars) > 0
    # The seed window starts _REPLAY_WINDOW bars back from the end of the
    # dataset (leaving room to advance forward, bar by bar, toward the
    # dataset's true final row -- otherwise there'd be nothing left to
    # "replay").
    start_index = 400 - live_market._REPLAY_WINDOW
    assert bars[-1]["close"] == pytest.approx(100.5 + start_index)


def test_replay_advance_false_does_not_move_the_cursor(tmp_path, monkeypatch):
    name = _write_dataset(tmp_path, monkeypatch, n=50)
    bars1, _ = live_market.fetch_replay_bars(name, advance=False)
    bars2, _ = live_market.fetch_replay_bars(name, advance=False)
    assert bars1 == bars2


def test_replay_advance_true_moves_forward_and_eventually_finishes(tmp_path, monkeypatch):
    name = _write_dataset(tmp_path, monkeypatch, n=10)
    finished = False
    last_bars = None
    for _ in range(30):
        last_bars, finished = live_market.fetch_replay_bars(name, advance=True)
        if finished:
            break
    assert finished
    assert last_bars[-1]["close"] == pytest.approx(100.5 + 9)


def test_reset_replay_clears_cached_state(tmp_path, monkeypatch):
    name = _write_dataset(tmp_path, monkeypatch, n=20)
    live_market.fetch_replay_bars(name, advance=True)
    assert name in live_market._replay_cache
    live_market.reset_replay(name)
    assert name not in live_market._replay_cache


def test_list_replay_datasets_reflects_stored_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_raw_data_dir", lambda: tmp_path)
    name = _write_dataset(tmp_path, monkeypatch, name="EURUSD/EURUSD15.csv", n=5)
    names = live_market.list_replay_datasets()
    assert name in names


def test_recent_trade_markers_empty_when_no_sessions(tmp_path, monkeypatch):
    db_path = tmp_path / "forward_test.db"
    monkeypatch.setattr("app.forward_test.journal._db_path", lambda: db_path)
    assert live_market.recent_trade_markers("XAUUSD") == []


def test_recent_trade_markers_builds_entry_and_exit_markers(tmp_path, monkeypatch):
    db_path = tmp_path / "forward_test.db"
    monkeypatch.setattr("app.forward_test.journal._db_path", lambda: db_path)

    journal = ForwardTestJournal()
    session_id = journal.start_session("python", "strat.py", "XAUUSD", 15, "123", "Demo-Server")
    trade_id = journal.record_open(session_id, mt5_ticket=1, direction=1, volume=0.1,
                                    entry_price=2000.0, sl_price=1990.0, tp_price=2020.0)
    journal.record_close(trade_id, exit_price=2010.0, pnl=100.0)

    markers = live_market.recent_trade_markers("XAUUSD")
    assert len(markers) == 2
    entry, exit_marker = markers
    assert entry["shape"] == "arrowUp"
    assert "BUY" in entry["text"]
    assert exit_marker["shape"] == "arrowDown"
    assert "EXIT +100.00" in exit_marker["text"]


def test_recent_trade_markers_short_trade_has_reversed_shapes(tmp_path, monkeypatch):
    db_path = tmp_path / "forward_test.db"
    monkeypatch.setattr("app.forward_test.journal._db_path", lambda: db_path)

    journal = ForwardTestJournal()
    session_id = journal.start_session("python", "strat.py", "EURUSD", 15, "123", "Demo-Server")
    trade_id = journal.record_open(session_id, mt5_ticket=2, direction=-1, volume=0.1,
                                    entry_price=1.1, sl_price=1.105, tp_price=1.09)
    journal.record_close(trade_id, exit_price=1.095, pnl=-50.0)

    markers = live_market.recent_trade_markers("EURUSD")
    entry, exit_marker = markers
    assert entry["shape"] == "arrowDown"
    assert "SELL" in entry["text"]
    assert exit_marker["shape"] == "arrowUp"
    assert "EXIT -50.00" in exit_marker["text"]
