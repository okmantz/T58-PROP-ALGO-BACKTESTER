"""
Shared LAN-IP + QR-code helpers for the web app.

Before this module existed, there were TWO different ways to start the
web app, and only ONE of them ever showed a QR code or a LAN address at
all:

  - `python run_web.py` (or the built T58-Web-App.exe) -- via
    app.web.launcher -- printed a banner with the LAN address and popped
    open a QR code image.
  - `python -m app.web.server` (the "plainer alternative" the README
    also documented) -- called `app.run(host="0.0.0.0", port=5000)`
    directly, with NO banner, NO LAN-IP printout beyond whatever
    Werkzeug's own dev-server startup log happens to include, and NO QR
    code at all.

Reported symptom: "the qr code still doesn't generate" (there was
nothing to generate one *from* on this path) plus typing
`https://127.0.0.1:5000` into a phone -- which fails for two independent
reasons even on the launcher path: 127.0.0.1 is the loopback address
(meaningless from a second device), and this server is plain HTTP, not
HTTPS, so the `https://` itself never even completes a handshake.

Fix: one shared module, used by BOTH entry points, so both print the
identical banner and both can render the identical QR code -- and the
web app itself now also exposes this at `/mobile-access` (see
app.web.server), so the QR code is visible from a browser tab on the PC
itself, not only from a separate launcher script's popped-open image.
"""
from __future__ import annotations

import base64
import io
import socket
from pathlib import Path

PORT = 5000


def get_lan_ip() -> str:
    """Best-effort guess at this machine's LAN IP (not the loopback one).

    Opens a UDP socket to a public address without sending any data --
    this is just a trick to ask the OS which local interface/IP it would
    use, so it works even with no real internet connection.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def lan_url(port: int = PORT) -> str:
    return f"http://{get_lan_ip()}:{port}"


def qr_code_data_uri(url: str) -> str | None:
    """Renders a QR code for `url` and returns it as a `data:` URI PNG a
    browser can display directly with no temp file, no filesystem write,
    and no OS-specific "open this image" step -- each of those was a
    silent single point of failure in the old file-based flow (a locked-
    down home directory, no default PNG viewer registered, antivirus
    flagging os.startfile). Returns None if the optional qrcode/Pillow
    dependency isn't installed or generation fails for any other reason
    -- the caller always has the plain URL as a fallback."""
    try:
        import qrcode

        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def qr_code_file(url: str) -> Path | None:
    """Same QR code, saved to a real PNG file -- kept for the standalone
    launcher's "pop open a picture viewer" behavior (some people prefer
    a real image window they can leave open over having to keep a
    browser tab around)."""
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


def startup_banner_lines(url: str, qr_path: Path | None) -> list[str]:
    """The exact banner text both entry points print -- pulled out as
    plain lines (rather than each entry point half-duplicating its own
    print() calls) so `python run_web.py` and `python -m app.web.server`
    can never again drift into showing different information."""
    lines = [
        "=" * 64,
        "  T58 QUANT ALGO BACKTESTER -- now running as a website",
        "=" * 64,
        f"  On THIS computer, it just opened at:\n      {url}",
        "",
        "  On your PHONE (same Wi-Fi as this computer), open EXACTLY this",
        "  address -- note it starts with http, NOT https, and it is NOT",
        "  127.0.0.1 (that address only ever means \"this computer itself\"):",
        f"      {url}",
    ]
    if qr_path is not None:
        lines += [
            "",
            f"  A QR code also opened in a picture viewer ({qr_path}).",
            "  Scan it with your phone's camera to jump straight there.",
        ]
    else:
        lines += [
            "",
            "  (Could not generate a QR code image -- just type the",
            "   address above into your phone's browser instead.)",
        ]
    lines += [
        "",
        "  This same QR code and address are also available any time from",
        "  the running app itself, at the /mobile-access page (there's a",
        "  \U0001F4F1 link for it in the sidebar).",
        "",
        "  Once it's open on your phone, use the browser menu ->",
        "  \"Add to Home Screen\" to get a real app icon.",
        "",
        "  If your phone says the site can't be reached, this almost",
        "  always means something on the network is blocking the",
        "  connection, not a problem with the app itself (it's already",
        "  listening on every network interface on this machine).",
        "  Most likely causes, in order:",
        "   1. Windows Defender Firewall -- the FIRST time this runs,",
        "      Windows should prompt \"Allow python.exe to communicate",
        "      on Private networks?\". If you clicked Cancel/No (or the",
        "      prompt never appeared), open Windows Security -> Firewall",
        "      & network protection -> Allow an app through firewall,",
        "      and enable Python for Private networks.",
        "   2. Your PC's network is set to \"Public\" instead of",
        "      \"Private\" in Windows -- Public profiles block incoming",
        "      connections like this one by default.",
        "   3. Phone and PC aren't actually on the same network (phone",
        "      on cellular data, or on a guest Wi-Fi network that",
        "      isolates devices from each other -- common on coffee",
        "      shop / office / hotel Wi-Fi).",
        "   4. A VPN active on either device can also hide LAN traffic",
        "      from the other one -- try disabling it temporarily.",
        "",
        "  Keep this window open while you use the app on your phone.",
        "  Closing this window stops the backtester.",
        "=" * 64,
    ]
    return lines


def print_startup_banner(url: str, qr_path: Path | None) -> None:
    for line in startup_banner_lines(url, qr_path):
        print(line)
