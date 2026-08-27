import shutil

import pytest

from app.strategy import library


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    """Isolate every test in this file from the real strategies/ directory,
    mirroring tests/test_storage.py's clean_raw_dir fixture."""
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def test_get_strategy_library_dir_creates_all_three_subfolders():
    base = library.get_strategy_library_dir()
    assert base.name == "strategies"
    for t in library.STRATEGY_TYPES:
        assert (base / t).is_dir()


def test_get_strategy_library_dir_for_one_type():
    d = library.get_strategy_library_dir("python")
    assert d.name == "python"
    assert d.parent.name == "strategies"


def test_unknown_strategy_type_rejected():
    with pytest.raises(ValueError):
        library.get_strategy_library_dir("cobol")
    with pytest.raises(ValueError):
        library.list_saved_strategies("cobol")


def test_save_and_list_strategy_bytes():
    library.save_strategy_bytes(b"print('hi')\n", "one.py", "python")
    library.save_strategy_bytes(b"//pine\n", "two.pine", "pinescript")
    names = sorted(s.name for s in library.list_saved_strategies())
    assert names == ["one.py", "two.pine"]

    python_only = library.list_saved_strategies("python")
    assert [s.name for s in python_only] == ["one.py"]
    assert python_only[0].strategy_type == "python"


def test_save_strategy_text_appends_extension_if_missing():
    dest = library.save_strategy_text("//mql5 code", "my_ea", "mql5")
    assert dest.name == "my_ea.mq5"
    assert dest.read_text() == "//mql5 code"


def test_save_strategy_text_does_not_double_extension():
    dest = library.save_strategy_text("print(1)", "already.py", "python")
    assert dest.name == "already.py"


def test_duplicate_filename_does_not_clobber():
    library.save_strategy_bytes(b"version 1", "dup.py", "python")
    library.save_strategy_bytes(b"version 2", "dup.py", "python")
    names = sorted(s.name for s in library.list_saved_strategies("python"))
    assert "dup.py" in names
    assert any(n != "dup.py" and "dup" in n for n in names)
    assert len(names) == 2


def test_save_strategy_path_copies_external_file(tmp_path):
    src = tmp_path / "external_strategy.py"
    src.write_text("print('external')")
    stored = library.save_strategy_path(src, "python")
    assert stored.exists()
    assert stored.parent == library.get_strategy_library_dir("python")
    assert stored.read_text() == src.read_text()


def test_save_strategy_path_is_idempotent_for_already_stored_file():
    dest = library.save_strategy_bytes(b"print(1)", "already.py", "python")
    result = library.save_strategy_path(dest, "python")
    assert result == dest
    assert len(library.list_saved_strategies("python")) == 1


def test_load_strategy_text_round_trips():
    library.save_strategy_bytes(b"print('round trip')", "rt.py", "python")
    assert library.load_strategy_text("python", "rt.py") == "print('round trip')"


def test_load_missing_strategy_raises():
    with pytest.raises(FileNotFoundError):
        library.load_strategy_text("python", "does_not_exist.py")


def test_resolve_saved_strategy_path_strips_path_traversal(tmp_path):
    # A malicious/careless filename with directory components must resolve
    # to a bare file inside the library dir, never escape it.
    outside = tmp_path / "secret.py"
    outside.write_text("should not be reachable")
    with pytest.raises(FileNotFoundError):
        library.resolve_saved_strategy_path("python", "../../secret.py")


def test_delete_saved_strategy():
    library.save_strategy_bytes(b"print(1)", "to_delete.py", "python")
    assert len(library.list_saved_strategies("python")) == 1
    library.delete_saved_strategy("python", "to_delete.py")
    assert library.list_saved_strategies("python") == []


def test_delete_missing_strategy_raises():
    with pytest.raises(FileNotFoundError):
        library.delete_saved_strategy("python", "ghost.py")


def test_list_saved_strategies_sorts_newest_first():
    import time

    library.save_strategy_bytes(b"old", "older.py", "python")
    time.sleep(0.02)
    newer_path = library.save_strategy_bytes(b"new", "newer.py", "python")

    strategies = library.list_saved_strategies("python")
    assert strategies[0].path == newer_path
    assert strategies[-1].name == "older.py"


def test_only_matching_extension_is_listed_per_type():
    library.save_strategy_bytes(b"x", "not_python.pine", "pinescript")
    assert library.list_saved_strategies("python") == []
    assert len(library.list_saved_strategies("pinescript")) == 1
