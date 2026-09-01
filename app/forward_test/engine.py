"""
ForwardTestSession -- the polling loop that turns closed bars into signals
into MT5 demo orders.

Deliberately reuses, rather than re-implements, the app's own decision
logic:
  - signal generation:  strategy.generate(df)                  (app.strategy.base)
  - stop/target sizing: the same distance-resolution rule execution.py uses
                         (dynamic distance -> fixed pips -> 1%-of-price fallback)
  - position sizing:    risk.position_size(equity, stop_loss_pips)  (app.backtest.risk)
  - daily loss breaker: risk.daily_loss_limit_pct, same semantics as the backtest engine

This keeps forward-test behavior honestly comparable to whatever the
strategy's backtest report already says -- if the numbers diverge, that's
real information about the strategy or the market, not an artifact of two
different implementations of "how big should this trade be."

Threading model: one background thread per session, started by `.start()`
and stopped by `.stop()` (graceful) or `.flatten_all_and_stop()` (also
closes every open position first -- the kill switch). All state changes are
funneled through this one thread; the UI only ever reads snapshots.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from app.backtest.execution import DEFAULT_STOP_PCT_OF_PRICE
from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.forward_test.mt5_connector import MT5Connector
from app.strategy.base import Strategy

LogCallback = Callable[[str, str], None]  # (level, message)


@dataclass
class ForwardTestStatus:
    running: bool = False
    connected: bool = False
    last_bar_time: Optional[pd.Timestamp] = None
    last_signal: int = 0
    open_position_ticket: Optional[int] = None
    balance: Optional[float] = None
    equity: Optional[float] = None
    n_trades_closed: int = 0
    win_rate: Optional[float] = None
    net_pnl: float = 0.0
    halted_reason: Optional[str] = None
    baseline_win_rate: Optional[float] = None
    drift_flag: Optional[str] = None


@dataclass
class ForwardTestConfig:
    symbol: str
    timeframe_minutes: int
    risk: RiskConfig
    poll_seconds: int = 20
    history_bars: int = 1500          # rolling window fed to strategy.generate() each poll
    min_drift_sample: int = 20        # trades needed before comparing to baseline_win_rate
    drift_tolerance_pts: float = 20.0  # percentage points of win-rate deviation to flag
    baseline_win_rate: Optional[float] = None  # from the strategy's own backtest report, if known


class ForwardTestSession:
    def __init__(
        self,
        strategy: Strategy,
        strategy_type: str,
        strategy_filename: str,
        connector: MT5Connector,
        journal: ForwardTestJournal,
        config: ForwardTestConfig,
        on_log: Optional[LogCallback] = None,
        on_status: Optional[Callable[[ForwardTestStatus], None]] = None,
    ):
        self.strategy = strategy
        self.strategy_type = strategy_type
        self.strategy_filename = strategy_filename
        self.connector = connector
        self.journal = journal
        self.cfg = config
        self._on_log = on_log or (lambda level, msg: None)
        self._on_status = on_status or (lambda status: None)

        self.status = ForwardTestStatus()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._session_id: Optional[int] = None
        self._open_trade_row_id: Optional[int] = None
        self._daily_realized_pnl = 0.0
        self._daily_key: Optional[str] = None
        self._mt5_down = False  # tracks reconnect state across polls, for the recovered-log message

    # -- public controls ----------------------------------------------------

    def start(self) -> tuple[bool, str]:
        if self.status.running:
            return False, "Already running."
        conn = self.connector.connect()
        if not conn.ok:
            self._log("error", conn.message)
            return False, conn.message
        self.status.connected = True
        self.status.balance = conn.balance
        self.status.equity = conn.equity

        self._session_id = self.journal.start_session(
            self.strategy_type, self.strategy_filename, self.cfg.symbol,
            self.cfg.timeframe_minutes, self.connector.login, self.connector.server,
        )
        self.status.baseline_win_rate = self.cfg.baseline_win_rate
        self._reconcile_existing_position()

        self._stop_flag.clear()
        self.status.running = True
        self.status.halted_reason = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log("info", f"Forward test started: {self.strategy_filename} on {self.cfg.symbol} "
                           f"({self.cfg.timeframe_minutes}m), account {conn.account_login}@{conn.account_server}.")
        return True, "Started."

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.poll_seconds + 10)
        self.status.running = False
        if self._session_id is not None:
            self.journal.end_session(self._session_id)
        self.connector.disconnect()
        self.status.connected = False
        self._log("info", "Forward test stopped.")

    def flatten_all_and_stop(self) -> None:
        """The kill switch: close every open position on this symbol, then stop."""
        self._log("warn", "Kill switch pressed -- flattening all open positions.")
        try:
            results = self.connector.close_all(self.cfg.symbol)
            for r in results:
                self._log("info" if r.ok else "error", r.message)
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"Error while flattening positions: {exc}")
        self.stop()

    # -- internals ------------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        if self._session_id is not None:
            try:
                self.journal.log_event(self._session_id, level, message)
            except Exception:
                pass
        self._on_log(level, message)

    def _reconcile_existing_position(self) -> None:
        """Adopt whatever MT5 says is actually open, rather than assuming
        flat -- important after an app restart mid-position."""
        try:
            positions = self.connector.get_open_positions(self.cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"Could not check for existing open positions: {exc}")
            return
        if not positions:
            return
        p = positions[0]
        self.status.open_position_ticket = p.ticket
        self.status.last_signal = p.direction
        self._log("info", f"Found an existing open position on {p.symbol} (ticket {p.ticket}) "
                           "-- adopting it instead of opening a duplicate.")

    def _today_key(self, ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    def _reset_daily_counter_if_needed(self, ts: pd.Timestamp) -> None:
        key = self._today_key(ts)
        if key != self._daily_key:
            self._daily_key = key
            self._daily_realized_pnl = 0.0
            self.status.halted_reason = None

    def _daily_loss_breached(self) -> bool:
        if self.cfg.risk.daily_loss_limit_pct is None:
            return False
        limit_amount = self.cfg.risk.initial_balance * (self.cfg.risk.daily_loss_limit_pct / 100.0)
        return self._daily_realized_pnl <= -abs(limit_amount)

    def _resolve_stop_distance(self, result, price: float) -> float:
        """Same precedence order as app.backtest.execution: per-bar dynamic
        distance, then fixed pips, then a 1%-of-price fallback -- never a
        divide-by-zero, never an unprotected trade."""
        if result.stop_loss_distance is not None:
            val = float(result.stop_loss_distance.iloc[-1])
            if val > 0:
                return val
        if result.stop_loss_pips:
            return float(result.stop_loss_pips) * self.cfg.risk.pip_size
        return abs(price) * DEFAULT_STOP_PCT_OF_PRICE

    def _resolve_target_distance(self, result, price: float) -> Optional[float]:
        if result.take_profit_distance is not None:
            val = float(result.take_profit_distance.iloc[-1])
            if val > 0:
                return val
        if result.take_profit_pips:
            return float(result.take_profit_pips) * self.cfg.risk.pip_size
        return None

    def _check_drift(self) -> None:
        stats = self.journal.closed_trade_stats(self._session_id)
        n = stats["n_trades"]
        self.status.n_trades_closed = n
        self.status.win_rate = stats["win_rate"]
        self.status.net_pnl = stats["net_pnl"]
        if self.cfg.baseline_win_rate is None or n < self.cfg.min_drift_sample or stats["win_rate"] is None:
            return
        deviation = stats["win_rate"] - self.cfg.baseline_win_rate
        if abs(deviation) >= self.cfg.drift_tolerance_pts:
            direction = "below" if deviation < 0 else "above"
            msg = (f"Forward-test win rate ({stats['win_rate']:.1f}%, n={n}) is {abs(deviation):.1f} "
                   f"points {direction} the backtest's ({self.cfg.baseline_win_rate:.1f}%) -- "
                   "worth a closer look before trusting this strategy further.")
            self.status.drift_flag = msg
            self._log("warn", msg)

    def _run_loop(self) -> None:
        try:
            while not self._stop_flag.is_set():
                try:
                    self._poll_once()
                except Exception as exc:  # noqa: BLE001 -- one bad poll must not kill the session
                    self._log("error", f"Poll error: {exc}\n{traceback.format_exc(limit=3)}")
                self._on_status(self.status)
                self._stop_flag.wait(self.cfg.poll_seconds)
        finally:
            self.status.running = False
            self._on_status(self.status)

    def _poll_once(self) -> None:
        # Verify the MT5 connection is actually alive before touching it --
        # a terminal restart, Windows update, or network blip otherwise
        # leaves the session silently dead (every call below raising or
        # returning None) until the user notices and manually restarts.
        was_down = self._mt5_down
        reconnect = self.connector.ensure_connected()
        if not reconnect.ok:
            self._mt5_down = True
            self._log("error", f"MT5 connection lost, reconnect failed: {reconnect.message}")
            return
        self._mt5_down = False
        if was_down:
            self._log("info", "MT5 connection recovered -- resuming polling.")
        summary = self.connector.account_summary()
        if summary:
            self.status.balance = summary["balance"]
            self.status.equity = summary["equity"]

        df = self.connector.fetch_completed_bars(self.cfg.symbol, self.cfg.timeframe_minutes, self.cfg.history_bars)
        if df.empty:
            return
        latest_bar_time = df["timestamp"].iloc[-1]
        self._reset_daily_counter_if_needed(latest_bar_time)

        if self.status.last_bar_time is not None and latest_bar_time <= self.status.last_bar_time:
            return  # no new completed bar yet
        self.status.last_bar_time = latest_bar_time

        # Reconcile: did our open position close on the broker side since last poll?
        self._reconcile_closed_trade()

        if self._daily_loss_breached():
            reason = f"Daily loss limit reached ({self.cfg.risk.daily_loss_limit_pct}% of balance) -- no new entries today."
            if self.status.halted_reason != reason:
                self._log("warn", reason)
            self.status.halted_reason = reason
            self._check_drift()
            return

        result = self.strategy.generate(df)
        signal = int(result.signals.iloc[-1])
        price = float(df["close"].iloc[-1])
        self.status.last_signal = signal

        if self.status.open_position_ticket is None and signal != 0:
            self._open_new_trade(signal, result, price)
        elif self.status.open_position_ticket is not None and signal == 0:
            self._close_current_trade("signal flat")
        elif self.status.open_position_ticket is not None and signal != 0 and signal != self._last_trade_direction():
            self._close_current_trade("signal reversed")
            self._open_new_trade(signal, result, price)

        self._check_drift()

    def _last_trade_direction(self) -> int:
        open_trades = self.journal.open_trades(self._session_id)
        return open_trades[0].direction if open_trades else 0

    def _open_new_trade(self, signal: int, result, price: float) -> None:
        stop_distance = self._resolve_stop_distance(result, price)
        target_distance = self._resolve_target_distance(result, price)
        stop_pips = stop_distance / self.cfg.risk.pip_size if self.cfg.risk.pip_size else 0
        equity = self.status.equity or self.cfg.risk.initial_balance
        volume = self.cfg.risk.position_size(equity, stop_pips)
        if volume <= 0:
            self._log("warn", "Computed position size was zero -- skipping entry.")
            return

        sl_price = price - signal * stop_distance
        tp_price = price + signal * target_distance if target_distance else None

        order = self.connector.place_market_order(
            self.cfg.symbol, signal, volume, sl_price=sl_price, tp_price=tp_price,
        )
        if not order.ok:
            self._log("error", f"Order failed: {order.message}")
            return

        self.status.open_position_ticket = order.ticket
        self._open_trade_row_id = self.journal.record_open(
            self._session_id, order.ticket, signal, order.volume or volume,
            order.price or price, sl_price, tp_price,
        )
        self._log("info", f"Opened {'LONG' if signal == 1 else 'SHORT'} {order.volume or volume:.2f} "
                           f"{self.cfg.symbol} @ {order.price or price:.5f} (ticket {order.ticket}).")

    def _close_current_trade(self, reason: str) -> None:
        ticket = self.status.open_position_ticket
        if ticket is None:
            return
        result = self.connector.close_position(ticket)
        if not result.ok:
            self._log("error", f"Close failed for ticket {ticket}: {result.message}")
            return
        self._finalize_closed_trade(result.price)
        self._log("info", f"Closed position (ticket {ticket}) -- {reason}.")

    def _reconcile_closed_trade(self) -> None:
        """If we think a position is open but MT5 no longer shows it (hit
        its SL/TP on the broker side between polls), record the close from
        journal + account history instead of losing track of it."""
        ticket = self.status.open_position_ticket
        if ticket is None:
            return
        still_open = any(p.ticket == ticket for p in self.connector.get_open_positions(self.cfg.symbol))
        if still_open:
            return
        self._finalize_closed_trade(exit_price=None)
        self._log("info", f"Position (ticket {ticket}) closed on the broker side (SL/TP hit) since last check.")

    def _finalize_closed_trade(self, exit_price: Optional[float]) -> None:
        if self._open_trade_row_id is None:
            self.status.open_position_ticket = None
            return
        open_trades = self.journal.open_trades(self._session_id)
        row = next((t for t in open_trades if t.id == self._open_trade_row_id), None)
        pnl = 0.0
        if row is not None:
            exit_p = exit_price if exit_price is not None else row.entry_price
            pnl = (exit_p - row.entry_price) * row.direction * row.volume
            self.journal.record_close(row.id, exit_p, pnl)
        self._daily_realized_pnl += pnl
        self.status.open_position_ticket = None
        self._open_trade_row_id = None
