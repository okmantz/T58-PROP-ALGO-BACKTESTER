"""Tests for app.live_deploy -- the curated prop-firm reference data and
the multi-account live-settings storage (mirrors app.forward_test.mt5_settings'
own test conventions, but for multiple named accounts)."""
from __future__ import annotations

import pytest

from app.live_deploy import live_settings, prop_firms


def test_every_prop_firm_has_at_least_one_platform():
    for firm in prop_firms.PROP_FIRMS:
        assert firm.platforms


def test_connectable_today_is_consistent_with_platforms():
    """connectable_today must be True if and only if at least one listed
    platform is MT4/MT5 (the only integration this app actually has)."""
    for firm in prop_firms.PROP_FIRMS:
        has_mt = any(p in ("MT4", "MT5") for p in firm.platforms)
        assert firm.connectable_today == has_mt, firm.name


def test_futures_only_firms_are_flagged_not_connectable():
    for name in ("Apex Trader Funding", "Topstep", "MyFundedFutures"):
        firm = prop_firms.find(name)
        assert firm is not None
        assert firm.connectable_today is False


def test_find_returns_none_for_unknown_firm():
    assert prop_firms.find("Not A Real Firm") is None


def _account(**overrides) -> live_settings.LiveAccount:
    defaults = dict(
        id=None, nickname="Test Account", firm_name="FTMO", platform="MT5",
        login="12345", server="FTMO-Server", password="secret123", terminal_path="",
    )
    defaults.update(overrides)
    return live_settings.LiveAccount(**defaults)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.live_deploy.live_settings.get_app_base_dir", lambda: tmp_path)
    yield


def test_load_accounts_empty_when_nothing_saved():
    assert live_settings.load_accounts() == []


def test_save_and_load_roundtrip_populates_password():
    account_id = live_settings.save_account(_account())
    loaded = live_settings.load_accounts()
    assert len(loaded) == 1
    assert loaded[0].id == account_id
    assert loaded[0].nickname == "Test Account"
    assert loaded[0].password == "secret123"


def test_password_never_written_to_plain_json(tmp_path):
    live_settings.save_account(_account(password="supersecret"))
    raw = live_settings._accounts_path().read_text(encoding="utf-8")
    assert "supersecret" not in raw


def test_saving_again_with_same_id_updates_rather_than_duplicates():
    account_id = live_settings.save_account(_account(nickname="First"))
    live_settings.save_account(_account(id=account_id, nickname="Renamed", password=""))
    accounts = live_settings.load_accounts()
    assert len(accounts) == 1
    assert accounts[0].nickname == "Renamed"


def test_editing_without_new_password_keeps_the_old_one():
    account_id = live_settings.save_account(_account(password="original-pw"))
    live_settings.save_account(_account(id=account_id, nickname="Renamed", password=""))
    accounts = live_settings.load_accounts()
    assert accounts[0].password == "original-pw"


def test_editing_with_new_password_replaces_it():
    account_id = live_settings.save_account(_account(password="original-pw"))
    live_settings.save_account(_account(id=account_id, password="new-pw"))
    accounts = live_settings.load_accounts()
    assert accounts[0].password == "new-pw"


def test_multiple_accounts_coexist():
    live_settings.save_account(_account(nickname="Acct A", login="111"))
    live_settings.save_account(_account(nickname="Acct B", login="222"))
    accounts = live_settings.load_accounts()
    assert {a.nickname for a in accounts} == {"Acct A", "Acct B"}


def test_delete_account_removes_it_and_its_password():
    account_id = live_settings.save_account(_account())
    live_settings.delete_account(account_id)
    assert live_settings.load_accounts() == []
    assert live_settings._load_password(account_id) == ""


def test_delete_one_account_leaves_others_intact():
    id_a = live_settings.save_account(_account(nickname="Keep", login="1"))
    id_b = live_settings.save_account(_account(nickname="Remove", login="2"))
    live_settings.delete_account(id_b)
    remaining = live_settings.load_accounts()
    assert len(remaining) == 1
    assert remaining[0].id == id_a


def test_is_usable_requires_login_server_and_password():
    assert _account().is_usable is True
    assert _account(login="").is_usable is False
    assert _account(server="").is_usable is False
    assert _account(password="").is_usable is False
