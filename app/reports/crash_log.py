"""
Process-wide crash logging for T58's long-running background jobs.

Evolution Lab, Full Pipeline, and Speed Run all run on background
threads, sometimes for hours, unattended (overnight). Before this
module existed, an unhandled exception in one of those threads -- or
the whole process dying (OOM kill, a worker crash taking a
ProcessPoolExecutor's parent down with it, etc.) -- left NOTHING on
disk to explain what happened: the only record was whatever had
scrolled past in the Tkinter log widget or the Flask job's in-memory
log, which is gone the moment the process exits. This writes every
crash straight to a plain text file the moment it happens,
independent of the GUI event loop or any in-memory log, so "the app
crashed overnight and I have no idea why" has an actual answer next
time: check data/logs/crash_log.txt (next to the .exe, or under your
user AppData folder -- see app.data.storage.get_app_base_dir).
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.data.storage import get_app_base_dir


def crash_log_path() -> Path:
    p = get_app_base_dir() / "data" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p / "crash_log.txt"


_WRITE_LOCK = threading.Lock()


def log_crash(component: str, exc: Optional[BaseException] = None, extra: Optional[str] = None) -> Path:
    """Appends a timestamped crash record to the crash log and returns
    its path. Safe to call from any thread; never raises -- a failure
    to log a crash must never itself crash the caller.

    exc: pass the caught exception directly when you have it (keeps
        its own traceback rather than whatever is on the stack at the
        point log_crash is called). If omitted, falls back to
        traceback.format_exc() -- only meaningful when called from
        inside an `except:` block.
    """
    path = crash_log_path()
    try:
        ts = datetime.now(timezone.utc).isoformat()
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            tb = traceback.format_exc()
        lines = [f"\n{'=' * 70}\n[{ts}] {component}\n{'=' * 70}\n"]
        if extra:
            lines.append(extra.rstrip() + "\n")
        lines.append(tb)
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(lines)
    except Exception:
        pass  # logging a crash must never itself raise
    return path


_HOOK_INSTALLED = False
_HOOK_LOCK = threading.Lock()


def install_thread_excepthook() -> None:
    """Installs a process-wide threading.excepthook that logs any
    exception which kills a background thread (daemon or not) to the
    crash log before Python's default hook runs (which, for a frozen
    .exe with no console attached, otherwise silently swallows it).
    Call this once, as early as possible, from each entry point
    (run_app.py / run_web.py) -- idempotent, safe to call more than
    once."""
    global _HOOK_INSTALLED
    with _HOOK_LOCK:
        if _HOOK_INSTALLED:
            return
        default_hook = threading.excepthook

        def _hook(args) -> None:
            try:
                log_crash(f"Unhandled exception in thread {args.thread.name!r}", exc=args.exc_value)
            except Exception:
                pass
            try:
                default_hook(args)
            except Exception:
                pass

        threading.excepthook = _hook
        _HOOK_INSTALLED = True
