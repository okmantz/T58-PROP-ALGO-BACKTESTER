"""
Local SQLite journal for forward-test trades and session events.

One row per position (opened here, updated in place when it closes) plus a
lightweight event log (connects, signals seen, warnings, circuit-breaker
trips) so a session's full history survives an app restart and can be
audited later against the backtest report it came from.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.data.storage import get_app_base_dir


def _db_path() -> Path:
    d = get_app_base_dir() / "data" / "forward_test"
    d.mkdir(parents=True, exist_ok=True)
    return d / "forward_test.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    strategy_type TEXT NOT NULL,
    strategy_filename TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe_minutes INTEGER NOT NULL,
    mt5_login TEXT,
    mt5_server TEXT,
    ended_at REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    mt5_ticket INTEGER,
    direction INTEGER NOT NULL,
    volume REAL NOT NULL,
    entry_time REAL NOT NULL,
    entry_price REAL NOT NULL,
    sl_price REAL,
    tp_price REAL,
    exit_time REAL,
    exit_price REAL,
    pnl REAL,
    status TEXT NOT NULL DEFAULT 'open',
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""


@dataclass
class TradeRecord:
    id: int
    session_id: int
    mt5_ticket: Optional[int]
    direction: int
    volume: float
    entry_time: float
    entry_price: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    exit_time: Optional[float]
    exit_price: Optional[float]
    pnl: Optional[float]
    status: str


class ForwardTestJournal:
    def __init__(self, db_path: Optional[Path] = None):
        self.path = db_path or _db_path()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_session(
        self, strategy_type: str, strategy_filename: str, symbol: str,
        timeframe_minutes: int, mt5_login: str, mt5_server: str,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at, strategy_type, strategy_filename, symbol, "
            "timeframe_minutes, mt5_login, mt5_server) VALUES (?,?,?,?,?,?,?)",
            (time.time(), strategy_type, strategy_filename, symbol, timeframe_minutes, mt5_login, mt5_server),
        )
        self._conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        self._conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))
        self._conn.commit()

    def log_event(self, session_id: int, level: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO events (session_id, ts, level, message) VALUES (?,?,?,?)",
            (session_id, time.time(), level, message),
        )
        self._conn.commit()

    def record_open(
        self, session_id: int, mt5_ticket: Optional[int], direction: int, volume: float,
        entry_price: float, sl_price: Optional[float], tp_price: Optional[float],
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO trades (session_id, mt5_ticket, direction, volume, entry_time, "
            "entry_price, sl_price, tp_price, status) VALUES (?,?,?,?,?,?,?,?, 'open')",
            (session_id, mt5_ticket, direction, volume, time.time(), entry_price, sl_price, tp_price),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_close(self, trade_id: int, exit_price: float, pnl: float) -> None:
        self._conn.execute(
            "UPDATE trades SET exit_time=?, exit_price=?, pnl=?, status='closed' WHERE id=?",
            (time.time(), exit_price, pnl, trade_id),
        )
        self._conn.commit()

    def open_trades(self, session_id: int) -> list[TradeRecord]:
        return self._query("SELECT * FROM trades WHERE session_id=? AND status='open'", (session_id,))

    def all_trades(self, session_id: int) -> list[TradeRecord]:
        return self._query(
            "SELECT * FROM trades WHERE session_id=? ORDER BY entry_time DESC", (session_id,)
        )

    def latest_session_for_symbol(self, symbol: str) -> Optional[int]:
        """Used by the Live Market page to find which session's trades to
        show as chart markers -- the most recent session run against this
        exact symbol, regardless of whether it has ended."""
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE symbol=? ORDER BY started_at DESC LIMIT 1", (symbol,),
        ).fetchone()
        return row[0] if row else None

    def recent_events(self, session_id: int, limit: int = 200) -> list[tuple]:
        cur = self._conn.execute(
            "SELECT ts, level, message FROM events WHERE session_id=? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        )
        return cur.fetchall()

    def closed_trade_stats(self, session_id: int) -> dict:
        rows = self._conn.execute(
            "SELECT pnl FROM trades WHERE session_id=? AND status='closed' AND pnl IS NOT NULL",
            (session_id,),
        ).fetchall()
        pnls = [r[0] for r in rows]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n_trades": n,
            "win_rate": (wins / n * 100.0) if n else None,
            "net_pnl": sum(pnls) if n else 0.0,
            "avg_pnl": (sum(pnls) / n) if n else None,
        }

    def _query(self, sql: str, params: tuple) -> list[TradeRecord]:
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [TradeRecord(**dict(zip(cols, row))) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
