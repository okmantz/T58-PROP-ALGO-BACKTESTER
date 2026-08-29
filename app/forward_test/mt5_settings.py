"""
Persistent settings for one saved MT5 demo account connection.

Mirrors app.ai.ollama_settings's exact storage split: non-secret fields
(login number, server, default symbol/timeframe) go in a small plain JSON
file; the one real secret (password) goes to the OS keyring, falling back
to a lightly-obfuscated local file if no keyring backend is available.
This is demo-account credential storage -- there is intentionally no
"live account" variant of this file (see app.forward_test package
docstring).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"
KEYRING_USERNAME = "mt5_demo"

DEFAULT_SERVER = ""
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME_MINUTES = 15


@dataclass
class MT5Settings:
    login: str = ""              # MT5 account number, kept as text (never arithmetic)
    server: str = DEFAULT_SERVER  # broker/prop-firm server name, e.g. "ICMarketsSC-Demo"
    password: str = ""            # secret -- keyring-backed
    symbol: str = DEFAULT_SYMBOL
    timeframe_minutes: int = DEFAULT_TIMEFRAME_MINUTES
    terminal_path: str = ""       # optional: path to terminal64.exe if MT5 can't auto-locate it

    @property
    def is_usable(self) -> bool:
        return bool(self.login.strip()) and bool(self.server.strip()) and bool(self.password)


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return _config_dir() / "mt5_settings.json"


def _password_fallback_path() -> Path:
    return _config_dir() / "mt5_password.txt"


def _try_keyring():
    try:
        import keyring  # type: ignore
        from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

        backend = keyring.get_keyring()
        if isinstance(backend, FailKeyring):
            return None
        return keyring
    except Exception:
        return None


def _obfuscate(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _deobfuscate(value: str) -> str:
    try:
        return base64.b64decode(value.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


def save_settings(settings: MT5Settings) -> None:
    payload = {
        "login": (settings.login or "").strip(),
        "server": (settings.server or DEFAULT_SERVER).strip(),
        "symbol": (settings.symbol or DEFAULT_SYMBOL).strip(),
        "timeframe_minutes": int(settings.timeframe_minutes or DEFAULT_TIMEFRAME_MINUTES),
        "terminal_path": (settings.terminal_path or "").strip(),
    }
    _settings_path().write_text(json.dumps(payload), encoding="utf-8")

    password = settings.password or ""
    kr = _try_keyring()
    if kr is not None:
        try:
            if password:
                kr.set_password(SERVICE_NAME, KEYRING_USERNAME, password)
            else:
                kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
            _password_fallback_path().unlink(missing_ok=True)
            return
        except Exception:
            pass  # fall through to the file-based fallback below

    if password:
        _password_fallback_path().write_text(_obfuscate(password), encoding="utf-8")
    else:
        _password_fallback_path().unlink(missing_ok=True)


def load_settings() -> MT5Settings:
    """Never raises -- returns blank/default settings if nothing is saved yet."""
    path = _settings_path()
    login, server, symbol, tf, terminal_path = "", DEFAULT_SERVER, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME_MINUTES, ""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            login = data.get("login") or ""
            server = data.get("server") or DEFAULT_SERVER
            symbol = data.get("symbol") or DEFAULT_SYMBOL
            tf = int(data.get("timeframe_minutes") or DEFAULT_TIMEFRAME_MINUTES)
            terminal_path = data.get("terminal_path") or ""
        except Exception:
            pass

    password = ""
    kr = _try_keyring()
    if kr is not None:
        try:
            password = kr.get_password(SERVICE_NAME, KEYRING_USERNAME) or ""
        except Exception:
            password = ""
    if not password:
        fallback = _password_fallback_path()
        if fallback.exists():
            try:
                password = _deobfuscate(fallback.read_text(encoding="utf-8"))
            except Exception:
                password = ""

    return MT5Settings(
        login=login, server=server, password=password,
        symbol=symbol, timeframe_minutes=tf, terminal_path=terminal_path,
    )
