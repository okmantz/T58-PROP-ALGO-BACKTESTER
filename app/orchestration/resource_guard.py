"""
Shared process-wide guardrails for T58's heavy, multi-worker-process
jobs (Search Lab, Evolution Lab, Full Pipeline batch, Speed Run).

Why this exists
----------------
Every one of those jobs independently spins up its own
ProcessPoolExecutor sized to (roughly) os.cpu_count() workers, and
each worker process loads its OWN full copy of the active market
data DataFrame (see app.search.batch_runner._init_worker and
app.evolution.engine._evo_init_worker). That's fine in isolation.

It is NOT fine if two or three of these jobs are started at the same
time in the same running app -- e.g. Evolution Lab left running
overnight, Full Pipeline batch started on top of it, then Speed Run
clicked on top of THAT. Each one independently claims the whole
machine's CPU count, so N simultaneous jobs try to hold N x
os.cpu_count() worker processes at once, each with its own copy of a
potentially large (months/years of 1-minute bars) DataFrame in
memory. On a real machine that is exactly the failure mode that looks
like "the whole desktop froze and then shut down" -- not a bug in any
one job, but the complete absence of any coordination between jobs
that each assume they own the whole machine.

This module provides two independent guardrails against that:

  HEAVY_JOB_GUARD    -- a simple named-slot lock. Only one heavy job
                        (by name) may be "active" at a time in this
                        process. Both the desktop GUI
                        (app.ui.main_window) and the web server
                        (app.web.server) import the same singleton,
                        so this protects whichever of the two entry
                        points is running -- it does NOT protect
                        against running the .exe AND the web server
                        at the same time from two separate OS
                        processes; there's no cheap way to guard
                        across process boundaries, so the UI-level
                        warning this enables is a strong hint, not an
                        absolute guarantee. If you run both, still
                        avoid stacking heavy jobs across them
                        yourself.

  safe_worker_count  -- caps a ProcessPoolExecutor's worker count so
                        that (workers x one worker's DataFrame memory
                        footprint) stays within a conservative budget
                        of AVAILABLE system memory, instead of
                        blindly using os.cpu_count() regardless of
                        how big the loaded dataset is. Falls back to
                        a conservative fixed assumption if psutil
                        isn't installed (it's an optional dependency
                        here, same as pywebview / MetaTrader5
                        elsewhere in this repo's requirements).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# Conservative assumption used ONLY when psutil isn't installed and we
# therefore have no way to ask the OS how much RAM is actually free
# right now. Deliberately low -- better to under-parallelize on a
# beefy machine than to repeat the freeze this module exists to
# prevent.
_FALLBACK_AVAILABLE_BYTES = 3 * 1024 ** 3  # 3 GB

# Never plan to use more than this fraction of *currently available*
# memory for one job's worker pool -- leaves headroom for the GUI /
# Flask process itself, the OS, and (since this can't see other heavy
# jobs running in a different OS process) some slack for those too.
_MAX_MEMORY_FRACTION = 0.5

# Per-worker overhead multiplier on top of the raw DataFrame size --
# indicator columns, intermediate arrays, and the backtest engine's
# own bookkeeping cost roughly this much again per worker in practice.
_WORKER_OVERHEAD_MULTIPLIER = 1.6


def _available_memory_bytes() -> int:
    if _HAS_PSUTIL:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return _FALLBACK_AVAILABLE_BYTES


def safe_worker_count(
    df: Optional[pd.DataFrame],
    requested: Optional[int] = None,
    max_candidates_in_flight: Optional[int] = None,
) -> int:
    """Returns a worker-process count that is safe to hand to a
    ProcessPoolExecutor given the size of `df` (the dataset every
    worker will load its own full copy of) and the memory actually
    available right now.

    requested: the caller's own preferred worker count (e.g.
        stage_cfg.workers, or None to mean "os.cpu_count()"). This is
        the UPPER bound -- safe_worker_count only ever returns this
        value or something SMALLER, never more workers than the
        caller wanted.
    max_candidates_in_flight: if the caller also can't usefully use
        more workers than it has work items for (e.g. don't spin up
        16 workers to process 3 candidates), pass that count here to
        additionally cap on it.

    Always returns at least 1.
    """
    cpu_cap = requested if requested is not None else (os.cpu_count() or 2)
    cpu_cap = max(1, int(cpu_cap))
    if max_candidates_in_flight is not None:
        cpu_cap = max(1, min(cpu_cap, int(max_candidates_in_flight)))

    if df is None or len(df) == 0:
        return cpu_cap

    try:
        one_copy_bytes = float(df.memory_usage(deep=True).sum()) * _WORKER_OVERHEAD_MULTIPLIER
    except Exception:
        return cpu_cap
    if one_copy_bytes <= 0:
        return cpu_cap

    budget = _available_memory_bytes() * _MAX_MEMORY_FRACTION
    memory_cap = max(1, int(budget // one_copy_bytes))

    return max(1, min(cpu_cap, memory_cap))


class HeavyJobGuard:
    """Named-slot mutex: only one heavy multi-process job may be
    'active' at a time per process. Not a queue -- a second job that
    tries to acquire while one is active is simply refused (the
    caller decides what to tell the user)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_name: Optional[str] = None

    def try_acquire(self, name: str) -> bool:
        with self._lock:
            if self._active_name is not None:
                return False
            self._active_name = name
            return True

    def release(self, name: str) -> None:
        with self._lock:
            if self._active_name == name:
                self._active_name = None

    @property
    def active_name(self) -> Optional[str]:
        with self._lock:
            return self._active_name


# Shared singleton -- imported by both app.ui.main_window and
# app.web.server so the two entry points respect the same guard
# whenever both happen to be running inside the same Python process.
# (The web server does not normally run inside the same OS process as
# the desktop .exe -- see module docstring for that limit.)
HEAVY_JOB_GUARD = HeavyJobGuard()

# Human-readable names for the guard's slot, shared by both UIs so the
# "X is already running" message always names the job consistently.
JOB_SEARCH_LAB = "Search Lab"
JOB_EVOLUTION_LAB = "Evolution Lab"
JOB_FULL_PIPELINE = "Full Pipeline"
JOB_SPEED_RUN = "Speed Run"

# UPGRADE (Sep 2026 UI pass, round 2): these five don't spin up their own
# ProcessPoolExecutor the way the four above do, but Walk-Forward Opt/GA in
# particular re-run a full parameter search per fold (their own internal
# multi-generation population search, repeated n_folds times) -- easily as
# expensive as one of the four "big" jobs, just single-process. None of
# them were guarded before, so nothing stopped e.g. WFO and CPCV and
# Sensitivity all running at once on top of each other (or on top of a
# Search Lab run) and adding up to the exact same "the whole machine froze"
# failure mode this module was written to prevent -- they just didn't
# multiply by cpu_count() while doing it. They share the SAME
# HEAVY_JOB_GUARD slot as the four above (not a separate guard), so any one
# of these nine job types blocks any other from starting concurrently in
# this process.
JOB_WFO = "Walk-Forward Optimization"
JOB_WFGA = "Walk-Forward GA"
JOB_CPCV = "CPCV"
JOB_SENSITIVITY = "Sensitivity"
JOB_MULTI_OBJECTIVE = "Multi-Objective Optimization"
JOB_REGIME_MATRIX = "Regime Survival Matrix"
