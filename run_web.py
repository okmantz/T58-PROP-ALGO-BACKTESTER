"""
T58 Trading — Prop Algo Backtester (Web/Phone edition).
PyInstaller entry point.

This file must stay at the repository root, *outside* the ``app``
package -- for the same reason ``run_app.py`` does (see that file's
docstring): pointing PyInstaller at a script that lives inside the
``app`` package breaks its import analysis and produces a
"ModuleNotFoundError: No module named 'app...'" crash at runtime.

Running this (or the built T58-Web-App.exe) starts the exact same
backtester as the desktop app, but as a small local website: it prints
this computer's address, opens it in this computer's browser, and pops
up a QR code so a phone on the same Wi-Fi can open it too.
"""
from __future__ import annotations

import multiprocessing

from app.web.launcher import run

if __name__ == "__main__":
    # See run_app.py for why this is required in a packaged .exe: Search
    # Lab's ProcessPoolExecutor workers re-launch this frozen executable,
    # and without freeze_support() each re-launch falls through to main()
    # again instead of running as a worker.
    multiprocessing.freeze_support()
    run()
