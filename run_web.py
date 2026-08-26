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

from app.web.launcher import main

if __name__ == "__main__":
    main()
