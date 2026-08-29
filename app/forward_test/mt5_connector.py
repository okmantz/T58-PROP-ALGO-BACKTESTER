"""
Thin wrapper around the official `MetaTrader5` Python package.

This is the ONLY file in the app that imports `MetaTrader5` directly, and
the import is guarded: the package only works against a running MT5
terminal on Windows, so everywhere else in the app (including this module's
own class definitions) must load cleanly even when it's absent -- the
Forward Test tab checks `is_available()` and shows a clear explanation
instead of a traceback if it's not.

Getting a free MT5 demo account: any MT5-supporting broker's own website
("open a demo account"), or the prop firm's own demo/trial account if they
offer one. No paid subscription anywhere in this path -- that's the entire
point of building this on MT5 instead of TradingView (see the
app.forward_test package docstring for why).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 -- absence is expected on non-Windows/dev machines
    mt5 = None  # type: ignore
    _MT5_IMPORT_ERROR = str(exc)


def is_available() -> bool:
    """True only if the MetaTrader5 package imported successfully. Does NOT
    mean a terminal is actually running/logged in -- call connect() to find
    that out."""
    return mt5 is not None


def unavailable_reason() -> str:
    if is_available():
        return ""
    if _MT5_IMPORT_ERROR and "No module named" in _MT5_IMPORT_ERROR:
        return (
            "The MetaTrader5 package isn't installed, or this isn't running on Windows. "
            "Forward Test requires: Windows, a running MT5 terminal, and `pip install "
            "MetaTrader5` (already in requirements.txt on Windows)."
        )
    return f"MetaTrader5 package failed to load: {_MT5_IMPORT_ERROR}"


# Human-readable explanations for the MT5 error codes this app's users
# actually hit in practice, keyed off mt5.last_error()'s numeric code.
# -10003 in particular ("IPC initialize failed") is NOT a login/credential
# problem -- it means the MetaTrader5 Python package could not find or
# start an actual MT5 terminal executable on this machine at all, so no
# login was ever attempted. The generic message the package itself
# returns for this doesn't say any of that, which is why it reads as
# mysterious.
_MT5_ERROR_EXPLANATIONS = {
    -10003: (
        "MT5 terminal not found/started (IPC initialize failed). This means the MetaTrader5 "
        "Python package couldn't locate or launch an actual MT5 terminal on this machine -- it "
        "never got as far as trying your login. Most common causes, in order: (1) the MT5 "
        "terminal isn't installed at all -- download it from your broker's site and log into "
        "your demo account in it at least once; (2) it IS installed, but not where this package "
        "auto-detects -- enter the exact path to terminal64.exe in 'Terminal path' below (use "
        "Browse/Auto-detect); (3) a 32-bit-only terminal build, which the MetaTrader5 Python "
        "package cannot drive -- reinstall the 64-bit terminal from your broker."
    ),
    -10004: "MT5 terminal found but not responding in time -- it may still be starting up. Try again in a few seconds.",
    -10005: "MT5 IPC timeout -- the terminal didn't respond in time. Make sure it isn't showing a blocking dialog (e.g. a Windows update prompt) and try again.",
    -2: "Invalid MT5 login/password/server combination -- double-check them against your broker's demo account details.",
    -6: "MT5 authorization failed -- the account number, password, or server name is wrong, or this account has been disabled.",
    -1: "MT5 returned a generic internal failure -- try restarting the MT5 terminal, then Save & Test Connection again.",
}


def describe_mt5_error(code: int, desc: str) -> str:
    """Turns an mt5.last_error() (code, desc) pair into an actionable
    explanation for whichever specific failure this app's users have
    actually hit, falling back to the package's own description
    unembellished for anything not in the table above."""
    explanation = _MT5_ERROR_EXPLANATIONS.get(code)
    if explanation:
        return f"MT5 connection failed ({code}): {desc}\n\n{explanation}"
    return f"MT5 connection failed ({code}): {desc}"


def find_terminal_candidates() -> list[str]:
    """Best-effort search of the handful of places a Windows MT5 install
    actually lands, for when auto-detection inside the MetaTrader5 package
    itself fails (the -10003 case). Returns existing terminal64.exe paths
    only, most-recently-modified first so a freshly (re)installed terminal
    sorts ahead of stale leftovers from an old broker. Silently returns an
    empty list on any non-Windows platform or permission error -- this is
    a convenience for the UI's 'Auto-detect' button, never something a
    caller should treat as authoritative."""
    import glob
    import os

    if os.name != "nt":
        return []

    patterns: list[str] = []
    program_files = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                      os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    for pf in program_files:
        if pf:
            patterns.append(os.path.join(pf, "*", "terminal64.exe"))
            patterns.append(os.path.join(pf, "*", "*", "terminal64.exe"))
    # The far more common location in practice: MetaQuotes/brokers install
    # per-user under AppData\Roaming\MetaQuotes\Terminal\<hash>\, not
    # Program Files at all.
    appdata = os.environ.get("APPDATA")
    if appdata:
        patterns.append(os.path.join(appdata, "MetaQuotes", "Terminal", "*", "terminal64.exe"))

    found: list[str] = []
    for pattern in patterns:
        try:
            found.extend(glob.glob(pattern))
        except Exception:
            continue

    seen = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    try:
        unique.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        pass
    return unique


# Maps this app's timeframe-in-minutes convention (matching the data/raw/
# CSV naming, e.g. XAUUSD15.csv = 15 minutes) to MT5's TIMEFRAME_* constants.
_TIMEFRAME_MAP = {
    1: "TIMEFRAME_M1", 5: "TIMEFRAME_M5", 15: "TIMEFRAME_M15", 30: "TIMEFRAME_M30",
    60: "TIMEFRAME_H1", 240: "TIMEFRAME_H4", 1440: "TIMEFRAME_D1",
}


def _mt5_timeframe(minutes: int):
    name = _TIMEFRAME_MAP.get(minutes)
    if name is None:
        raise ValueError(
            f"Unsupported timeframe: {minutes} minutes. Supported: {sorted(_TIMEFRAME_MAP)}."
        )
    return getattr(mt5, name)


@dataclass
class ConnectionResult:
    ok: bool
    message: str
    account_login: int | None = None
    account_server: str | None = None
    balance: float | None = None
    equity: float | None = None
    currency: str | None = None
    resolved_terminal_path: str | None = None  # set only when auto-detection found a working path the caller didn't supply


@dataclass
class OrderResult:
    ok: bool
    message: str
    ticket: int | None = None
    price: float | None = None
    volume: float | None = None


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    direction: int  # 1 long, -1 short
    volume: float
    open_price: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    profit: float


class MT5Connector:
    """One connector per forward-test session. Not thread-safe against
    other simultaneous connectors in the same process (the underlying
    MetaTrader5 package is a single global connection to one terminal) --
    the Forward Test tab only ever runs one session at a time for this
    reason."""

    def __init__(self, login: str, password: str, server: str, terminal_path: str = ""):
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path or None
        self._connected = False

    def connect(self) -> ConnectionResult:
        if not is_available():
            return ConnectionResult(ok=False, message=unavailable_reason())
        try:
            login_int = int(self.login)
        except ValueError:
            return ConnectionResult(ok=False, message=f"MT5 login must be numeric, got '{self.login}'.")

        kwargs = {}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        ok = mt5.initialize(login=login_int, password=self.password, server=self.server, **kwargs)

        if not ok:
            code, desc = mt5.last_error()
            # The one failure worth retrying automatically: "terminal not
            # found" with no explicit path given. If a real terminal64.exe
            # turns up in one of the common install locations, try it once
            # before giving up -- this is exactly the fix for the
            # -10003/"MetaTrader 5 x64 not found" case with an empty
            # Terminal path field.
            if code == -10003 and not self.terminal_path:
                for candidate in find_terminal_candidates():
                    retry_ok = mt5.initialize(
                        login=login_int, password=self.password, server=self.server, path=candidate,
                    )
                    if retry_ok:
                        ok = True
                        self.terminal_path = candidate  # remembered so the caller can persist it
                        break
            if not ok:
                code, desc = mt5.last_error()
                return ConnectionResult(ok=False, message=describe_mt5_error(code, desc))

        info = mt5.account_info()
        if info is None:
            code, desc = mt5.last_error()
            mt5.shutdown()
            return ConnectionResult(ok=False, message=f"Connected but could not read account info ({code}): {desc}")

        self._connected = True
        return ConnectionResult(
            ok=True, message="Connected." if not self.terminal_path else f"Connected (using terminal at {self.terminal_path}).",
            account_login=info.login, account_server=info.server,
            balance=info.balance, equity=info.equity, currency=info.currency,
            resolved_terminal_path=self.terminal_path or None,
        )

    def disconnect(self) -> None:
        if is_available() and self._connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def account_summary(self) -> Optional[dict]:
        if not self._connected:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login, "server": info.server, "balance": info.balance,
            "equity": info.equity, "margin_free": info.margin_free, "currency": info.currency,
        }

    def symbol_point(self, symbol: str) -> float:
        """The instrument's minimum price increment -- this app's `pip_size`
        for a given symbol, read straight from the broker instead of
        guessed from price magnitude."""
        si = mt5.symbol_info(symbol)
        if si is None:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Symbol '{symbol}' not found/enabled on this MT5 account.")
            si = mt5.symbol_info(symbol)
        return float(si.point)

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        """Returns the last `count` fully-CLOSED bars as a DataFrame with
        this app's standard timestamp/open/high/low/close/volume schema
        (see app.data.importer). The most recent still-forming bar is
        always excluded so signals never see a partial bar."""
        tf = _mt5_timeframe(timeframe_minutes)
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}' on this MT5 account.")
        # Pull one extra bar and drop the most recent -- MT5's rates array
        # includes the currently-forming bar at index -1.
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count + 1)
        if rates is None or len(rates) < 2:
            code, desc = mt5.last_error()
            raise RuntimeError(f"Could not fetch rates for {symbol} ({code}): {desc}")
        df = pd.DataFrame(rates)
        df = df.iloc[:-1]  # drop the still-forming bar
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["time"], unit="s", utc=True),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["tick_volume"].astype(float),
        }).reset_index(drop=True)
        return out

    def latest_completed_bar_time(self, symbol: str, timeframe_minutes: int) -> Optional[pd.Timestamp]:
        df = self.fetch_completed_bars(symbol, timeframe_minutes, 1)
        if df.empty:
            return None
        return df["timestamp"].iloc[-1]

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        out: list[OpenPosition] = []
        if not positions:
            return out
        for p in positions:
            out.append(OpenPosition(
                ticket=p.ticket, symbol=p.symbol,
                direction=1 if p.type == mt5.ORDER_TYPE_BUY else -1,
                volume=p.volume, open_price=p.price_open, sl=p.sl, tp=p.tp,
                open_time=pd.to_datetime(p.time, unit="s", utc=True), profit=p.profit,
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Forward Test", deviation: int = 20,
    ) -> OrderResult:
        if direction not in (1, -1):
            return OrderResult(ok=False, message=f"Invalid direction {direction}, must be 1 or -1.")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(ok=False, message=f"No tick data for {symbol}.")
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price:
            request["sl"] = float(sl_price)
        if tp_price:
            request["tp"] = float(tp_price)
        result = mt5.order_send(request)
        if result is None:
            code, desc = mt5.last_error()
            return OrderResult(ok=False, message=f"order_send returned None ({code}): {desc}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(ok=False, message=f"Order rejected (retcode {result.retcode}): {result.comment}")
        return OrderResult(ok=True, message="Filled.", ticket=result.order, price=result.price, volume=result.volume)

    def close_position(self, ticket: int, comment: str = "T58 Forward Test close") -> OrderResult:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(ok=False, message=f"No open position with ticket {ticket}.")
        p = positions[0]
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return OrderResult(ok=False, message=f"No tick data for {p.symbol}.")
        is_long = p.type == mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.bid if is_long else tick.ask,
            "deviation": 20,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            code, desc = mt5.last_error()
            return OrderResult(ok=False, message=f"order_send returned None ({code}): {desc}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(ok=False, message=f"Close rejected (retcode {result.retcode}): {result.comment}")
        return OrderResult(ok=True, message="Closed.", ticket=result.order, price=result.price, volume=result.volume)

    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]:
        return [self.close_position(p.ticket) for p in self.get_open_positions(symbol)]
