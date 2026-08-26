"""Tests for app.web.launcher (the T58-Web-App.exe entry point glue)."""
from __future__ import annotations

from pathlib import Path

from app.web.launcher import _qr_image_path, get_lan_ip


def test_get_lan_ip_returns_a_plausible_ipv4_address():
    ip = get_lan_ip()
    parts = ip.split(".")
    assert len(parts) == 4
    assert all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def test_qr_image_path_creates_a_real_png(tmp_path, monkeypatch):
    # Redirect the QR code's output location into a temp dir so the test
    # doesn't touch the real user's home directory.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    out_path = _qr_image_path("http://192.168.1.23:5000")

    assert out_path is not None
    assert out_path.exists()
    assert out_path.suffix == ".png"
    # PNG file signature.
    with open(out_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
