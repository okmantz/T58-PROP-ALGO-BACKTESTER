"""
T58 Trading — Prop Algo Backtester
PyInstaller entry point.

This file must stay at the repository root, *outside* the ``app`` package.

Why this file exists:
    The previous build pointed PyInstaller directly at ``app/main.py``.
    Because that script lives *inside* the ``app`` package (a directory
    containing ``app/__init__.py``), PyInstaller's import analysis treats
    the script's own directory as the search root instead of the repo
    root. That makes ``from app.ui.main_window import launch`` inside
    ``app/main.py`` unresolvable at freeze time, which is exactly why the
    packaged .exe crashed on launch with:

        ModuleNotFoundError: No module named 'app.ui.main_window'

    Pointing PyInstaller at this file instead fixes that: this script's
    directory *is* the repo root, so the ``app`` package is found
    normally, both when run with plain ``python run_app.py`` and when
    frozen into an .exe.
"""
from __future__ import annotations

import multiprocessing

from app.main import main
from app.reports.crash_log import install_thread_excepthook

if __name__ == "__main__":
    # Catches any exception that kills a background thread (Evolution Lab,
    # Full Pipeline, Speed Run, Search Lab all run on one) and writes it to
    # data/logs/crash_log.txt immediately -- independent of the GUI, so a
    # crash that also takes the whole process down (e.g. out-of-memory)
    # still leaves a record. See app.reports.crash_log's module docstring.
    install_thread_excepthook()
    # REQUIRED for the packaged .exe: Search Lab (app/search/batch_runner.py)
    # spawns worker processes with concurrent.futures.ProcessPoolExecutor.
    # On Windows, a frozen/PyInstaller build has no real `fork()` -- every
    # worker process is started by re-launching this SAME .exe from
    # scratch. Without freeze_support() called first, each of those
    # re-launches falls through to `main()` again instead of running as a
    # plain worker, so every worker process tries to boot a second copy of
    # the whole Tkinter GUI, immediately conflicts with the already-running
    # instance, and dies -- which is exactly what surfaces to the user as
    # "concurrent.futures.process.BrokenProcessPool: A process in the
    # process pool was terminated abruptly while the future was running or
    # pending" as soon as Search Lab starts Stage 1. freeze_support() makes
    # multiprocessing detect it's being called as a worker re-launch and
    # skip straight to running the worker, instead of falling through to
    # main(). This is a no-op (returns immediately) on macOS/Linux and when
    # NOT frozen, so it's always safe to call first.
    multiprocessing.freeze_support()
    main()
