"""
Live Market data feed for the Live Market tab -- symbol/timeframe discovery
and OHLCV bar retrieval, across three possible sources:

  - MT5    -- a real live/demo MetaTrader 5 terminal connection, reusing the
              exact same saved settings as the Live Demo Test tab. Bars
              include the currently-forming candle, so polling this
              repeatedly and calling the chart's series.update() on the
              most recent bar is exactly TradingView Lightweight Charts'
              own documented real-time-update pattern -- no full redraw.
  - ALPACA -- Alpaca's REST market data API (stocks/crypto), reusing the
              same saved API keys as the Market Data tab's Alpaca fetch.
              Free-tier stock data is exchange-delayed (not real-time);
              this is surfaced honestly to the page rather than claimed
              as live.
  - REPLAY -- no live connection available (or none configured yet):
              replays an already-imported local CSV bar-by-bar at a
              steady pace, so the page always has something real to draw
              and the whole real-time-update code path can be exercised
              without needing a broker connection.

Kept out of app/web/server.py to keep that file's route list readable --
server.py only wires these functions to a handful of thin Flask routes.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

import pandas as pd

from app.data import alpaca_credentials
from app.data.storage import get_raw_data_dir, list_stored_datasets
from app.forward_test import mt5_connector as mt5_connector_module
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.mt5_settings import load_settings as load_mt5_settings

MT5_TIMEFRAME_NAMES = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1"}
ALPACA_TIMEFRAME_LABELS = {1: "1Min", 5: "5Min", 15: "15Min", 30: "30Min", 60: "1Hour", 1440: "1Day"}
TIMEFRAME_CHOICES_MINUTES = [1, 5, 15, 30, 60, 240, 1440]

_mt5_lock = threading.Lock()
_mt5_ready = False
_mt5_last_attempt = 0.0
_MT5_RETRY_COOLDOWN_SECONDS = 15.0  # don't retry a broken connection on every single poll


def mt5_status() -> dict:
    """Best-effort read of whether a live MT5 connection is usable right
    now, without blocking on a fresh connection attempt just to check.
    Never raises -- a corrupt settings file or an unexpected error from
    the MetaTrader5 package here must not take down the whole Live
    Market page along with it."""
    try:
        settings = load_mt5_settings()
        return {
            "available": mt5_connector_module.is_available(),
            "configured": settings.is_usable,
            "connected": _mt5_ready,
            "default_symbol": settings.symbol,
            "default_timeframe_minutes": settings.timeframe_minutes,
        }
    except Exception:
        return {
            "available": False, "configured": False, "connected": False,
            "default_symbol": "XAUUSD", "default_timeframe_minutes": 15,
        }


def _ensure_mt5_connected() -> bool:
    global _mt5_ready, _mt5_last_attempt
    if _mt5_ready:
        return True
    if not mt5_connector_module.is_available():
        return False
    settings = load_mt5_settings()
    if not settings.is_usable:
        return False
    now = time.time()
    if now - _mt5_last_attempt < _MT5_RETRY_COOLDOWN_SECONDS:
        return False
    with _mt5_lock:
        if _mt5_ready:
            return True
        _mt5_last_attempt = time.time()
        try:
            connector = mt5_connector_module.MT5Connector(
                settings.login, settings.password, settings.server, settings.terminal_path,
            )
            result = connector.connect()
            _mt5_ready = bool(result.ok)
        except Exception:
            # The underlying MetaTrader5 package occasionally raises
            # rather than returning a clean failure (seen from a
            # background thread, which is exactly how this gets called --
            # the Flask dev server this app starts runs in its own
            # thread). Treat that identically to "failed to connect"
            # rather than letting it become an unhandled 500 for the
            # whole Live Market page.
            _mt5_ready = False
        return _mt5_ready


def fetch_mt5_bars(symbol: str, timeframe_minutes: int, count: int = 300) -> list[dict]:
    if not _ensure_mt5_connected():
        return []
    try:
        import MetaTrader5 as mt5  # safe: _ensure_mt5_connected() already confirmed importability

        tf_name = MT5_TIMEFRAME_NAMES.get(timeframe_minutes, "M15")
        tf_const = getattr(mt5, f"TIMEFRAME_{tf_name}", mt5.TIMEFRAME_M15)
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if rates is None:
            return []
        bars = []
        for r in rates:
            volume = float(r["real_volume"]) if r["real_volume"] else float(r["tick_volume"])
            bars.append({
                "time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]), "volume": volume,
            })
        return bars
    except Exception:
        return []


def fetch_alpaca_bars(symbol: str, asset_class: str, timeframe_minutes: int, count: int = 300) -> list[dict]:
    creds = alpaca_credentials.load_credentials()
    if not creds or not creds.is_usable:
        return []
    from app.data.alpaca_source import fetch_bars as alpaca_fetch_bars

    tf_label = ALPACA_TIMEFRAME_LABELS.get(timeframe_minutes, "15Min")
    end = pd.Timestamp.now("UTC")
    # Generous lookback so `count` bars actually exist at this timeframe
    # even across weekends/market closures -- trimmed to the last `count`
    # rows below.
    lookback_days = max(3, (timeframe_minutes * count) // (60 * 24) + 3)
    start = end - pd.Timedelta(days=lookback_days)
    try:
        df = alpaca_fetch_bars(
            creds.api_key, creds.secret_key, symbol, asset_class, tf_label,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        )
    except Exception:
        return []
    if df is None or df.empty:
        return []
    df = df.tail(count)
    bars = []
    for row in df.itertuples():
        bars.append({
            "time": int(pd.Timestamp(row.timestamp).timestamp()), "open": float(row.open),
            "high": float(row.high), "low": float(row.low), "close": float(row.close),
            "volume": float(getattr(row, "volume", 0) or 0),
        })
    return bars


@dataclass
class _ReplayState:
    df: pd.DataFrame
    cursor: int
    finished: bool = False


_replay_cache: dict[str, _ReplayState] = {}
_replay_lock = threading.Lock()
_REPLAY_WINDOW = 300


def list_replay_datasets() -> list[str]:
    try:
        return [d.name for d in list_stored_datasets()]
    except Exception:
        return []


def fetch_replay_bars(dataset_name: str, advance: bool = True) -> tuple[list[dict], bool]:
    """Returns (bars, finished). `advance=False` re-reads the current
    window without moving the cursor forward -- used for the very first
    request after picking a dataset, so simply opening the page doesn't
    immediately burn a bar off the replay.

    Never raises: a dataset that fails to parse (corrupt file, unexpected
    encoding, a genuinely bad row) is treated as "finished, no bars" --
    the page shows UNAVAILABLE for that dataset instead of the whole
    request blowing up with a 500.
    """
    if not dataset_name:
        return [], True
    try:
        with _replay_lock:
            state = _replay_cache.get(dataset_name)
            if state is None:
                from app.data.importer import import_csv

                path = get_raw_data_dir() / dataset_name
                if not path.exists():
                    return [], True
                result = import_csv(path)
                if result.dataframe is None or result.dataframe.empty:
                    return [], True
                start_cursor = max(0, len(result.dataframe) - _REPLAY_WINDOW)
                state = _ReplayState(df=result.dataframe, cursor=start_cursor)
                _replay_cache[dataset_name] = state

            if advance and not state.finished:
                state.cursor = min(state.cursor + 1, len(state.df) - 1)
                if state.cursor >= len(state.df) - 1:
                    state.finished = True

            window_start = max(0, state.cursor - _REPLAY_WINDOW + 1)
            window = state.df.iloc[window_start: state.cursor + 1]
            bars = []
            for row in window.itertuples():
                bars.append({
                    "time": int(pd.Timestamp(row.timestamp).timestamp()), "open": float(row.open),
                    "high": float(row.high), "low": float(row.low), "close": float(row.close),
                    "volume": float(getattr(row, "volume", 0) or 0),
                })
            return bars, state.finished
    except Exception:
        return [], True


def reset_replay(dataset_name: str) -> None:
    with _replay_lock:
        _replay_cache.pop(dataset_name, None)


def recent_trade_markers(symbol: str, limit: int = 100) -> list[dict]:
    """Buy/sell markers for the Live Market chart's candlestick series,
    pulled from the Live Demo Test journal -- the most recent session run
    against this exact symbol, if any. Read-only; safe to call while a
    session is actively writing to the same database file. Never raises."""
    try:
        journal = ForwardTestJournal()
        session_id = journal.latest_session_for_symbol(symbol)
        if session_id is None:
            return []
        trades = journal.all_trades(session_id)[:limit]
    except Exception:
        return []

    markers = []
    for t in trades:
        is_long = t.direction == 1
        markers.append({
            "time": int(t.entry_time),
            "position": "belowBar" if is_long else "aboveBar",
            "shape": "arrowUp" if is_long else "arrowDown",
            "color": "#3ED685" if is_long else "#F0596A",
            "text": f"{'BUY' if is_long else 'SELL'} {t.entry_price:.5g}",
        })
        if t.exit_time:
            pnl = t.pnl or 0.0
            profit = pnl >= 0
            markers.append({
                "time": int(t.exit_time),
                "position": "aboveBar" if is_long else "belowBar",
                "shape": "arrowDown" if is_long else "arrowUp",
                "color": "#3ED685" if profit else "#F0596A",
                "text": f"EXIT {'+' if profit else ''}{pnl:.2f}",
            })
    markers.sort(key=lambda m: m["time"])
    return markers
