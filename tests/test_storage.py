import shutil

import pytest

from app.data import storage


@pytest.fixture(autouse=True)
def clean_raw_dir(tmp_path, monkeypatch):
    """
    Isolate every test in this file from the real data/raw/ directory.

    CRITICAL: get_raw_data_dir() resolves to <repo_root>/data/raw in normal
    (non-frozen) runs -- the SAME directory a real user's stored market-data
    CSVs live in. This fixture used to shutil.rmtree() that real directory
    directly, before AND after every test, which means simply running this
    test file with pytest permanently deleted a user's actual uploaded
    datasets. Redirecting get_app_base_dir() to a pytest tmp_path for the
    duration of each test gives every test its own throwaway data/raw/ to
    freely create and destroy, with zero risk to the real one.
    """
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw_dir = storage.get_raw_data_dir()
    yield
    shutil.rmtree(raw_dir, ignore_errors=True)


def test_get_raw_data_dir_is_created():
    d = storage.get_raw_data_dir()
    assert d.exists()
    assert d.name == "raw"
    assert d.parent.name == "data"


def test_store_and_list_csv_bytes():
    storage.store_csv_bytes(b"a,b\n1,2\n", "one.csv")
    storage.store_csv_bytes(b"a,b\n3,4\n", "two.csv")
    names = sorted(ds.name for ds in storage.list_stored_datasets())
    assert names == ["one.csv", "two.csv"]


def test_duplicate_filename_does_not_clobber(tmp_path):
    storage.store_csv_bytes(b"first content\n", "dup.csv")
    storage.store_csv_bytes(b"second content\n", "dup.csv")
    names = sorted(ds.name for ds in storage.list_stored_datasets())
    assert "dup.csv" in names
    assert any(n != "dup.csv" and "dup" in n for n in names)
    assert len(names) == 2


def test_store_csv_path_copies_external_file(tmp_path):
    src = tmp_path / "external.csv"
    src.write_text("a,b\n1,2\n")
    stored = storage.store_csv_path(src)
    assert stored.exists()
    assert stored.parent == storage.get_raw_data_dir()
    assert stored.read_text() == src.read_text()


def test_store_csv_path_is_idempotent_for_already_stored_file():
    dest = storage.store_csv_bytes(b"a,b\n1,2\n", "already.csv")
    result = storage.store_csv_path(dest)
    assert result == dest
    assert len(storage.list_stored_datasets()) == 1


def test_list_stored_datasets_empty_when_no_files():
    assert storage.list_stored_datasets() == []


def test_list_stored_datasets_finds_files_in_instrument_subfolders():
    raw_dir = storage.get_raw_data_dir()
    (raw_dir / "EURUSD").mkdir()
    (raw_dir / "EURUSD" / "EURUSD5.csv").write_text("a,b\n1,2\n")
    (raw_dir / "GBPUSD").mkdir()
    (raw_dir / "GBPUSD" / "GBPUSD5.csv").write_text("a,b\n3,4\n")
    storage.store_csv_bytes(b"a,b\n5,6\n", "flat_at_top_level.csv")  # still found alongside subfolders

    names = sorted(ds.name for ds in storage.list_stored_datasets())
    assert names == ["EURUSD/EURUSD5.csv", "GBPUSD/GBPUSD5.csv", "flat_at_top_level.csv"]


def test_stored_dataset_subfolder_name_resolves_back_to_its_real_path():
    raw_dir = storage.get_raw_data_dir()
    (raw_dir / "XAUUSD").mkdir()
    nested = raw_dir / "XAUUSD" / "XAUUSD15.csv"
    nested.write_text("a,b\n1,2\n")

    datasets = storage.list_stored_datasets()
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "XAUUSD/XAUUSD15.csv"
    # This is exactly how app/web/server.py's _resolve_dataset() and the
    # desktop app's stored-dataset picker turn the selected name back into
    # a real path -- must round-trip correctly on every OS, including
    # Windows where Path() still accepts "/" separators.
    assert raw_dir / ds.name == nested
    assert (raw_dir / ds.name).exists()


def test_list_stored_datasets_sorts_newest_first_across_subfolders(tmp_path):
    import time

    raw_dir = storage.get_raw_data_dir()
    (raw_dir / "A").mkdir()
    older = raw_dir / "A" / "older.csv"
    older.write_text("a,b\n1,2\n")
    time.sleep(0.02)
    newer = storage.store_csv_bytes(b"a,b\n3,4\n", "newer_at_top.csv")

    datasets = storage.list_stored_datasets()
    assert datasets[0].path == newer
    assert datasets[-1].path == older
