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

from app.main import main

if __name__ == "__main__":
    main()
