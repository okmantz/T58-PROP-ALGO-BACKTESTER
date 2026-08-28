import pytest

from app.data import alpaca_credentials


@pytest.fixture(autouse=True)
def isolate_credentials(tmp_path, monkeypatch):
    """Redirects both storage paths (OS keyring and the file fallback) away
    from anything real for the duration of each test: the fallback file
    goes under a throwaway tmp_path instead of the real app data dir, and
    the OS keyring is forced "unavailable" so every test exercises the
    file-based fallback deterministically regardless of what's installed
    on the machine running the suite."""
    monkeypatch.setattr(alpaca_credentials, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(alpaca_credentials, "_try_keyring", lambda: None)
    yield


def test_no_credentials_saved_initially():
    assert alpaca_credentials.load_credentials() is None
    assert alpaca_credentials.has_saved_credentials() is False


def test_save_and_load_roundtrip():
    alpaca_credentials.save_credentials("AKMYKEY123", "myS3cret/withslash+chars=")
    creds = alpaca_credentials.load_credentials()
    assert creds is not None
    assert creds.api_key == "AKMYKEY123"
    assert creds.secret_key == "myS3cret/withslash+chars="
    assert creds.is_usable
    assert alpaca_credentials.has_saved_credentials() is True


def test_save_overwrites_previous_value():
    alpaca_credentials.save_credentials("first_key", "first_secret")
    alpaca_credentials.save_credentials("second_key", "second_secret")
    creds = alpaca_credentials.load_credentials()
    assert creds.api_key == "second_key"
    assert creds.secret_key == "second_secret"


def test_clear_removes_saved_credentials():
    alpaca_credentials.save_credentials("k", "s")
    assert alpaca_credentials.has_saved_credentials() is True
    alpaca_credentials.clear_credentials()
    assert alpaca_credentials.load_credentials() is None
    assert alpaca_credentials.has_saved_credentials() is False


def test_fallback_file_is_not_plaintext():
    """The obfuscation isn't real encryption, but the raw key/secret
    strings should not appear verbatim in the fallback file -- a basic
    'don't leave it in cleartext in an obvious spot' bar."""
    alpaca_credentials.save_credentials("AKPLAINTEXTCHECK", "SECRETPLAINTEXTCHECK")
    raw = alpaca_credentials._fallback_path().read_text(encoding="utf-8")
    assert "AKPLAINTEXTCHECK" not in raw
    assert "SECRETPLAINTEXTCHECK" not in raw


def test_clear_is_a_no_op_when_nothing_saved():
    # Should not raise even though nothing has been saved yet.
    alpaca_credentials.clear_credentials()
    assert alpaca_credentials.load_credentials() is None
