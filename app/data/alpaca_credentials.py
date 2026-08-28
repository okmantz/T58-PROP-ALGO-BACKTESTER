"""Persistent storage for the user's Alpaca API credentials.

The falsification-kit-tester scripts (fetch_5m.py, fetch_cache.py) read
ALPACA_API_KEY / ALPACA_SECRET_KEY from environment variables every run --
fine for a one-off CLI script, but this app needs the user to type keys
into a form once and have them come back on the next launch.

This module prefers the OS credential store (via the optional `keyring`
package) when one is available, since that's actually encrypted at rest by
the operating system (Windows Credential Manager / macOS Keychain / Linux
Secret Service). If `keyring` isn't installed, or no real backend is
available (some minimal/headless environments only ship keyring's no-op
"fail" backend), it falls back to a local JSON file under this app's own
data/config/ directory. That fallback file is only lightly obfuscated
(base64) -- NOT encrypted -- which is roughly the security level of a
saved browser autofill on a machine only the user has access to. Nothing
here is ever transmitted anywhere; every function in this module only
touches the local OS vault or the local filesystem.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.data.storage import get_app_base_dir

SERVICE_NAME = "T58PropAlgoBacktester"
KEYRING_USERNAME = "alpaca"


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str

    @property
    def is_usable(self) -> bool:
        return bool(self.api_key and self.secret_key)


def _config_dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fallback_path() -> Path:
    return _config_dir() / "alpaca_credentials.json"


def _try_keyring():
    """Returns the keyring module if it's installed AND a real (non no-op)
    backend is usable, else None. Never raises."""
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


def save_credentials(api_key: str, secret_key: str) -> None:
    """Persists the given keys, preferring the OS keyring. Overwrites
    whatever was previously saved (by either storage path)."""
    api_key = (api_key or "").strip()
    secret_key = (secret_key or "").strip()

    kr = _try_keyring()
    if kr is not None:
        try:
            kr.set_password(
                SERVICE_NAME,
                KEYRING_USERNAME,
                json.dumps({"api_key": api_key, "secret_key": secret_key}),
            )
            # Don't leave a stale copy sitting in the fallback file too.
            _fallback_path().unlink(missing_ok=True)
            return
        except Exception:
            pass  # fall through to the file-based fallback below

    payload = {"api_key": _obfuscate(api_key), "secret_key": _obfuscate(secret_key)}
    _fallback_path().write_text(json.dumps(payload), encoding="utf-8")


def load_credentials() -> Optional[AlpacaCredentials]:
    """Returns saved credentials, or None if nothing has been saved yet."""
    kr = _try_keyring()
    if kr is not None:
        try:
            raw = kr.get_password(SERVICE_NAME, KEYRING_USERNAME)
            if raw:
                data = json.loads(raw)
                return AlpacaCredentials(data.get("api_key", ""), data.get("secret_key", ""))
        except Exception:
            pass

    path = _fallback_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AlpacaCredentials(
            _deobfuscate(data.get("api_key", "")),
            _deobfuscate(data.get("secret_key", "")),
        )
    except Exception:
        return None


def has_saved_credentials() -> bool:
    creds = load_credentials()
    return bool(creds and creds.is_usable)


def clear_credentials() -> None:
    """Removes saved credentials from both storage paths, if present."""
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE_NAME, KEYRING_USERNAME)
        except Exception:
            pass
    _fallback_path().unlink(missing_ok=True)
