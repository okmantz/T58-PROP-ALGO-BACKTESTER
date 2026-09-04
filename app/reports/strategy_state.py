"""
Persisted "current strategy" pointer + its validation checklist.

This is a thin, dependency-free layer alongside run_history.py, answering
two questions the UI needs and nothing tracked before:

  1. "Which strategy is the person currently focused on?"
  2. "Which of the deeper validation tools (Walk-Forward Opt, CPCV,
     Sensitivity, Walk-Forward GA, Regime Survival Matrix) has actually
     been run against THAT strategy, and what did each one find?"

Nothing here is inferred or guessed:
  - The current strategy is only ever set by an explicit action (the
    "Set as current" button on the Dashboard scorecard, or the query-string
    action /run redirects into after a backtest -- see server.py).
  - A validation result is only ever recorded against the current strategy
    when the name+instrument a validation job just ran against matches the
    current strategy's own name+instrument exactly. Running CPCV against
    some other, unrelated strategy never silently overwrites the
    checklist for whatever you've marked current.
  - A tool with no strict pass/fail verdict (Sensitivity, today) records
    passed=None -- "this has been run", not "and it passed". Nothing here
    invents a threshold the underlying tool doesn't itself compute.

Storage is two small JSON files under data/config/, next to run_history.json
and ui_theme.json. Both web and desktop already share get_app_base_dir(),
so this file is usable from either build.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_CURRENT_FILENAME = "current_strategy.json"
_CHECKLIST_FILENAME = "validation_checklist.json"

VALIDATION_KINDS = ("wfo", "cpcv", "sensitivity", "wfga", "regime_matrix")

VALIDATION_LABELS = {
    "wfo": "Walk-Forward Opt",
    "cpcv": "CPCV / PBO",
    "sensitivity": "Sensitivity",
    "wfga": "Walk-Forward GA",
    "regime_matrix": "Regime Survival Matrix",
}

# Where each check lives in the guided nav, for the Validate hub's "Run" links.
VALIDATION_HREFS = {
    "wfo": "/walk-forward-opt",
    "cpcv": "/cpcv",
    "sensitivity": "/sensitivity",
    "wfga": "/walk-forward-ga",
    "regime_matrix": "/regime-matrix",
}


def _config_dir() -> Path:
    # Imported locally (not at module load time) so tests can monkeypatch
    # app.data.storage.get_app_base_dir and have it actually take effect --
    # matches the pattern app.ui.main_window's theme persistence already uses.
    from app.data.storage import get_app_base_dir

    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def strategy_key(strategy_name: str, instrument: str) -> str:
    """Case-insensitive identity for a (strategy, instrument) pair -- the
    same strategy re-run against the same instrument is "the same
    strategy" for checklist purposes even if capitalization drifts."""
    return f"{(strategy_name or '').strip().lower()}::{(instrument or '').strip().lower()}"


def get_current_strategy() -> Optional[dict]:
    """Returns {"strategy_name", "instrument", "timeframe", "set_at"} or
    None if nothing has been set yet (or the file is missing/corrupt --
    never raises)."""
    path = _config_dir() / _CURRENT_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("strategy_name"):
            return None
        return data
    except Exception:
        return None


def set_current_strategy(strategy_name: str, instrument: str, timeframe: str = "") -> dict:
    data = {
        "strategy_name": strategy_name,
        "instrument": instrument,
        "timeframe": timeframe,
        "set_at": time.time(),
    }
    (_config_dir() / _CURRENT_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    return data


def clear_current_strategy() -> None:
    path = _config_dir() / _CURRENT_FILENAME
    if path.exists():
        path.unlink()


def _load_checklist_store() -> dict:
    path = _config_dir() / _CHECKLIST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checklist_store(store: dict) -> None:
    (_config_dir() / _CHECKLIST_FILENAME).write_text(json.dumps(store), encoding="utf-8")


def record_validation(
    strategy_name: str,
    instrument: str,
    kind: str,
    *,
    passed: Optional[bool],
    summary: str = "",
    report_html: str = "",
) -> None:
    """Records one validation tool's result against (strategy_name,
    instrument). Best-effort: never raises -- a checklist-recording
    failure must not break the validation job that produced it."""
    if kind not in VALIDATION_KINDS:
        return
    try:
        key = strategy_key(strategy_name, instrument)
        store = _load_checklist_store()
        entry = store.setdefault(key, {"strategy_name": strategy_name, "instrument": instrument, "checks": {}})
        entry["strategy_name"] = strategy_name
        entry["instrument"] = instrument
        entry.setdefault("checks", {})[kind] = {
            "ran": True,
            "passed": passed,
            "summary": summary,
            "report_html": report_html,
            "recorded_at": time.time(),
        }
        _save_checklist_store(store)
    except Exception:
        pass


def get_checklist(strategy_name: str, instrument: str) -> dict:
    """Always returns every kind in VALIDATION_KINDS, defaulting an unset
    one to {"ran": False, "passed": None, "summary": "", "report_html": ""}."""
    key = strategy_key(strategy_name, instrument)
    store = _load_checklist_store()
    checks = store.get(key, {}).get("checks", {})
    return {
        kind: checks.get(kind, {"ran": False, "passed": None, "summary": "", "report_html": ""})
        for kind in VALIDATION_KINDS
    }


def robustness_score(strategy_name: str, instrument: str) -> dict:
    """A transparent count, not a fabricated single number: how many of
    the 5 checks have been run at all, and of those, how many passed
    (only where the tool itself computes a verdict -- Sensitivity's
    passed=None entries are excluded from decided_count/passed_count,
    not counted as failures)."""
    checklist = get_checklist(strategy_name, instrument)
    ran = [c for c in checklist.values() if c["ran"]]
    decided = [c for c in ran if c["passed"] is not None]
    passed = [c for c in decided if c["passed"]]
    return {
        "ran_count": len(ran),
        "total_count": len(VALIDATION_KINDS),
        "decided_count": len(decided),
        "passed_count": len(passed),
        "pct_run": round(100.0 * len(ran) / len(VALIDATION_KINDS), 1),
        "pct_passed_of_decided": round(100.0 * len(passed) / len(decided), 1) if decided else None,
    }
