"""Tests for app.forward_test -- covers the pieces that don't require an
actual MT5 terminal (settings persistence, the SQLite journal, and the
engine's stop/target distance resolution and daily-loss-breach logic)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.forward_test import mt5_connector, mt5_settings
from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
from app.forward_test.journal import ForwardTestJournal


def test_mt5_unavailable_reports_clearly():
    # This sandbox never has the real MetaTrader5 package installed.
    assert mt5_connector.is_available() is False
    assert "MetaTrader5" in mt5_connector.unavailable_reason() or "Windows" in mt5_connector.unavailable_reason()


def test_mt5_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_settings, "_config_dir", lambda: tmp_path)
    settings = mt5_settings.MT5Settings(
        login="555111", server="Broker-Demo", password="s3cret",
        symbol="EURUSD", timeframe_minutes=5,
    )
    mt5_settings.save_settings(settings)
    loaded = mt5_settings.load_settings()
    assert loaded.login == "555111"
    assert loaded.server == "Broker-Demo"
    assert loaded.symbol == "EURUSD"
    assert loaded.timeframe_minutes == 5
    # Password fallback path is used when no OS keyring backend is present
    # in this environment -- either way it must round-trip.
    assert loaded.password == "s3cret"


def test_mt5_settings_is_usable():
    assert mt5_settings.MT5Settings(login="1", server="s", password="p").is_usable
    assert not mt5_settings.MT5Settings(login="", server="s", password="p").is_usable
    assert not mt5_settings.MT5Settings(login="1", server="s", password="").is_usable


def test_journal_records_open_and_close(tmp_path):
    j = ForwardTestJournal(db_path=tmp_path / "ft.db")
    sid = j.start_session("python", "strat.py", "XAUUSD", 15, "123", "Demo")
    trade_id = j.record_open(sid, mt5_ticket=42, direction=1, volume=0.1, entry_price=2000.0, sl_price=1990.0, tp_price=2020.0)
    assert j.open_trades(sid)[0].id == trade_id
    j.record_close(trade_id, exit_price=2015.0, pnl=150.0)
    stats = j.closed_trade_stats(sid)
    assert stats["n_trades"] == 1
    assert stats["win_rate"] == 100.0
    assert stats["net_pnl"] == 150.0
    assert j.open_trades(sid) == []
    j.end_session(sid)


def test_journal_events_logged(tmp_path):
    j = ForwardTestJournal(db_path=tmp_path / "ft2.db")
    sid = j.start_session("python", "strat.py", "XAUUSD", 15, "123", "Demo")
    j.log_event(sid, "warn", "daily loss limit hit")
    events = j.recent_events(sid)
    assert events[0][2] == "daily loss limit hit"


class _FakeResult:
    """Minimal stand-in for StrategyResult, only the fields the engine reads."""
    def __init__(self, signals, stop_loss_distance=None, take_profit_distance=None,
                 stop_loss_pips=None, take_profit_pips=None):
        self.signals = pd.Series(signals)
        self.stop_loss_distance = pd.Series(stop_loss_distance) if stop_loss_distance is not None else None
        self.take_profit_distance = pd.Series(take_profit_distance) if take_profit_distance is not None else None
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips


def _make_session(risk=None):
    risk = risk or RiskConfig(initial_balance=50_000.0, risk_value=1.0, pip_size=0.0001)
    cfg = ForwardTestConfig(symbol="EURUSD", timeframe_minutes=15, risk=risk)
    # connector/journal/strategy aren't exercised by the pure resolution
    # helpers below, so None stand-ins are fine here.
    return ForwardTestSession(
        strategy=None, strategy_type="python", strategy_filename="x.py",
        connector=None, journal=None, config=cfg,
    )


def test_resolve_stop_distance_prefers_dynamic_distance():
    session = _make_session()
    result = _FakeResult(signals=[1], stop_loss_distance=[0.0025], stop_loss_pips=999)
    assert session._resolve_stop_distance(result, price=1.1000) == pytest.approx(0.0025)


def test_resolve_stop_distance_falls_back_to_fixed_pips():
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, pip_size=0.0001)
    session = _make_session(risk)
    result = _FakeResult(signals=[1], stop_loss_pips=20)
    assert session._resolve_stop_distance(result, price=1.1000) == pytest.approx(20 * 0.0001)


def test_resolve_stop_distance_falls_back_to_pct_of_price_when_no_stop_defined():
    session = _make_session()
    result = _FakeResult(signals=[1])
    distance = session._resolve_stop_distance(result, price=2000.0)
    assert distance == pytest.approx(2000.0 * 0.01)  # DEFAULT_STOP_PCT_OF_PRICE


def test_daily_loss_breach_detection():
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, daily_loss_limit_pct=2.0)
    session = _make_session(risk)
    session._daily_realized_pnl = -900.0
    assert not session._daily_loss_breached()  # 2% of 50k = 1000
    session._daily_realized_pnl = -1000.0
    assert session._daily_loss_breached()


def test_daily_loss_breach_disabled_when_limit_is_none():
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, daily_loss_limit_pct=None)
    session = _make_session(risk)
    session._daily_realized_pnl = -50_000.0
    assert not session._daily_loss_breached()
