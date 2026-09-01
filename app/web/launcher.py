"""
T58 Prop Algo Backtester — phone-friendly launcher.

This is the entry point behind the "T58-Web-App.exe" build. It exists so
Owen (or anyone else) can:

    1. Download a zip from GitHub Releases.
    2. Extract it.
    3. Double-click T58-Web-App.exe.
    4. Get a QR code on screen -- scan it with a phone camera on the same
       Wi-Fi -- and the full backtester opens in the phone's browser,
       installable as a home-screen app (PWA) via "Add to Home Screen".

No hosting account, no app store, no Termux/third-party runtime on the
phone. The only requirement is that the PC running this .exe stays on
and both devices share the same Wi-Fi network while the phone is used.

This module does NOT duplicate the Flask app in app.web.server -- it
only adds the "find my LAN IP / show a QR code / open a browser" glue
around the exact same server.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 5000


def get_lan_ip() -> str:
    """Best-effort guess at this machine's LAN IP (not the loopback one).

    Opens a UDP socket to a public address without sending any data --
    this is just a trick to ask the OS which local interface/IP it would
    use, so it works with no internet connection too.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _qr_image_path(url: str) -> Path | None:
    """Render a QR code PNG for `url` and return its path, or None if the
    qrcode/Pillow libraries aren't available or QR generation fails for any
    other reason (server still works fine without this -- the printed URL
    is always shown too). Deliberately broad: this is a nice-to-have, so a
    missing optional dependency, a locked/unwritable home directory, or any
    other environment quirk should never crash the launcher."""
    try:
        import qrcode

        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image()
        out_dir = Path.home() / ".t58-backtester"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "phone-qr-code.png"
        img.save(out_path)
        return out_path
    except Exception:
        return None


def _open_qr_image(path: Path) -> None:
    """Open the QR code image with whatever the OS uses for PNGs."""
    try:
        if sys.platform.startswith("win"):
            import os

            os.startfile(path)  # noqa: S606 -- intentional, user-facing helper
        elif sys.platform == "darwin":
            import subprocess

            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess

            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        # Non-fatal -- the console message with the raw URL is enough.
        pass


def _banner(url: str, qr_path: Path | None) -> None:
    line = "=" * 64
    print(line)
    print("  T58 PROP ALGO BACKTESTER -- now running as a website")
    print(line)
    print(f"  On THIS computer, it just opened at:\n      {url}")
    print()
    print("  On your PHONE (same Wi-Fi as this computer), open:")
    print(f"      {url}")
    if qr_path is not None:
        print()
        print(f"  A QR code also opened in a picture viewer ({qr_path}).")
        print("  Scan it with your phone's camera to jump straight there.")
    else:
        print()
        print("  (Could not generate a QR code image -- just type the")
        print("   address above into your phone's browser instead.)")
    print()
    print("  Once it's open on your phone, use the browser menu ->")
    print('  "Add to Home Screen" to get a real app icon.')
    print()
    print("  If your phone says the site can't be reached, this almost")
    print("  always means something on the network is blocking the")
    print("  connection, not a problem with the app itself (it's already")
    print("  listening on every network interface on this machine).")
    print("  Most likely causes, in order:")
    print("   1. Windows Defender Firewall -- the FIRST time this runs,")
    print("      Windows should prompt \"Allow python.exe to communicate")
    print("      on Private networks?\". If you clicked Cancel/No (or the")
    print("      prompt never appeared), open Windows Security -> Firewall")
    print("      & network protection -> Allow an app through firewall,")
    print("      and enable Python for Private networks.")
    print("   2. Your PC's network is set to \"Public\" instead of")
    print("      \"Private\" in Windows -- Public profiles block incoming")
    print("      connections like this one by default.")
    print("   3. Phone and PC aren't actually on the same network (phone")
    print("      on cellular data, or on a guest Wi-Fi network that")
    print("      isolates devices from each other -- common on coffee")
    print("      shop / office / hotel Wi-Fi).")
    print("   4. A VPN active on either device can also hide LAN traffic")
    print("      from the other one -- try disabling it temporarily.")
    print()
    print("  Keep this window open while you use the app on your phone.")
    print("  Closing this window stops the backtester.")
    print(line)


def main() -> None:
    # Printed and flushed immediately, before importing the (much heavier)
    # Flask app or generating a QR code -- on a slow machine, antivirus-
    # scanned first run, or a large PyInstaller onefile .exe that has to
    # self-extract before Python even starts running user code, several
    # seconds can pass with the console window sitting there before this
    # point is reached. Without an immediate flushed print, that gap looks
    # indistinguishable from "nothing is happening" (the exact complaint
    # this fixes) -- a person has no way to tell "it's loading" apart from
    # "it silently died."
    print("Starting T58 Prop Algo Backtester (web/phone edition)...", flush=True)
    print("(First launch can take a little while to unpack -- please wait.)", flush=True)

    # Import here (not at module top) so this launcher's own startup
    # banner logic has zero dependency on the heavier app import graph
    # succeeding before we've at least tried to explain what's happening.
    from app.web.server import app  # noqa: WPS433

    ip = get_lan_ip()
    url = f"http://{ip}:{PORT}"
    qr_path = _qr_image_path(url)

    def _open_things_once_server_is_up() -> None:
        time.sleep(1.0)
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        if qr_path is not None:
            _open_qr_image(qr_path)

    threading.Thread(target=_open_things_once_server_is_up, daemon=True).start()

    _banner(url, qr_path)
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False)
    except OSError as exc:
        # Most common real-world case: port 5000 already in use, either by
        # a previous copy of this same app still running in the
        # background, or (on macOS) the AirPlay Receiver service. Flask/
        # Werkzeug's own message for this is a plain OSError with an errno,
        # not obviously about a port conflict at a glance -- spell it out.
        print(flush=True)
        print("=" * 64, flush=True)
        print(f"  Could not start the server: {exc}", flush=True)
        print("  This almost always means port 5000 is already in use --", flush=True)
        print("  either another copy of this app is already running (check", flush=True)
        print("  for another console window with this same banner already", flush=True)
        print("  open), or something else on this PC is using that port.", flush=True)
        print("=" * 64, flush=True)
        raise


def run() -> None:
    """Entry point used by both `python run_web.py` and this module's own
    `if __name__ == "__main__"` guard -- wraps main() so an abnormal exit
    is printed AND the console waits for a keypress before closing.

    Why this exists: Windows closes a console window the instant the
    process that opened it exits. If main() failed here with nothing
    catching it, the very message that would explain what went wrong
    flashes past in a fraction of a second and the window vanishes --
    which is exactly what "the terminal appeared, but nothing was
    printed" looks like from the outside: the failure happened too fast
    to read, not that nothing happened at all. This catches BOTH an
    uncaught exception AND Werkzeug's own habit of calling sys.exit()
    directly on a startup failure (e.g. "port 5000 already in use") --
    the latter raises SystemExit, which a plain `except Exception` does
    NOT catch, so it would otherwise still slip through and close the
    window instantly even with error handling in main() itself. A clean
    Ctrl+C (KeyboardInterrupt) or an explicit sys.exit(0) are left alone
    since those are the normal, intentional ways to stop the server --
    pausing on those would just be annoying.
    """
    try:
        main()
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        _pause_on_abnormal_exit(f"exited with an error (code {exc.code}) -- see the message above, if any.")
    except Exception:
        import traceback

        print(flush=True)
        traceback.print_exc()
        _pause_on_abnormal_exit("crashed on startup -- see the traceback above.")


def _pause_on_abnormal_exit(reason: str) -> None:
    print(flush=True)
    print("=" * 64, flush=True)
    print(f"  T58 Prop Algo Backtester {reason}", flush=True)
    print("=" * 64, flush=True)
    try:
        input("Press Enter to close this window...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    run()
