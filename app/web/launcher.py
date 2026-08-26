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
    qrcode/Pillow libraries aren't available (server still works fine
    without this -- the printed URL is always shown too)."""
    try:
        import qrcode
    except ImportError:
        return None

    img = qrcode.make(url, box_size=10, border=2)
    out_dir = Path.home() / ".t58-backtester"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phone-qr-code.png"
    img.save(out_path)
    return out_path


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
    print("  Keep this window open while you use the app on your phone.")
    print("  Closing this window stops the backtester.")
    print(line)


def main() -> None:
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
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
