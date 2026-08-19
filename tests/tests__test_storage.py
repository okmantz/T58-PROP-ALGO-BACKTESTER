import shutil

import pytest

from app.data import storage


@pytest.fixture(autouse=True)
def clean_raw_dir():
    """Each test gets an empty data/raw/ and cleans up after itself."""
    raw_dir = storage.get_raw_data_dir()
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
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
