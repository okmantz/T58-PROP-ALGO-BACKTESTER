"""Tests for app.ui.main_window's theme system -- covers the pure-logic
pieces (apply_theme, persistence) without needing a real Tk display."""
from __future__ import annotations

import json

import pytest

from app.ui import main_window as mw


@pytest.fixture(autouse=True)
def _restore_dark_theme_after_each_test():
    """Every test in this module mutates module-global color constants --
    always leave them back on 'dark' afterward so other test modules that
    import main_window aren't affected by test order."""
    yield
    mw.apply_theme("dark")


def test_both_themes_define_every_color_key():
    dark_keys = set(mw.THEMES["dark"].keys())
    light_keys = set(mw.THEMES["light"].keys())
    assert dark_keys == light_keys


def test_apply_theme_updates_module_globals():
    mw.apply_theme("dark")
    assert mw.BG == mw.THEMES["dark"]["BG"]
    mw.apply_theme("light")
    assert mw.BG == mw.THEMES["light"]["BG"]
    assert mw.CURRENT_THEME == "light"


def test_apply_theme_ignores_unknown_name():
    mw.apply_theme("dark")
    before = mw.BG
    mw.apply_theme("not_a_real_theme")
    assert mw.BG == before
    assert mw.CURRENT_THEME == "dark"


def test_apply_theme_persists_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    mw.apply_theme("light")
    saved = json.loads((tmp_path / "data" / "config" / "ui_theme.json").read_text())
    assert saved["theme"] == "light"
    mw.apply_theme("dark")
    saved = json.loads((tmp_path / "data" / "config" / "ui_theme.json").read_text())
    assert saved["theme"] == "dark"


def test_load_theme_name_defaults_to_dark_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    assert mw._load_theme_name() == "dark"


def test_load_theme_name_reads_persisted_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ui_theme.json").write_text(json.dumps({"theme": "light"}))
    assert mw._load_theme_name() == "light"


def test_load_theme_name_ignores_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ui_theme.json").write_text("{not valid json")
    assert mw._load_theme_name() == "dark"


def test_light_theme_status_colors_differ_from_dark():
    """The semantic pass/fail/info/warning colors must actually change
    between themes, not just the backgrounds -- a light theme reusing the
    neon-bright dark-theme green/red would be unreadable on a white panel."""
    assert mw.THEMES["dark"]["GREEN"] != mw.THEMES["light"]["GREEN"]
    assert mw.THEMES["dark"]["RED"] != mw.THEMES["light"]["RED"]
