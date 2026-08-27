import shutil
import zipfile

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


# ---------------------------------------------------------------------------
# Overwrite vs. duplicate on save
# ---------------------------------------------------------------------------

def test_saving_same_name_by_default_raises_instead_of_duplicating():
    library.save_strategy_bytes(b"version 1", "fvg_v1.py", "python")
    with pytest.raises(library.StrategyAlreadyExists):
        library.save_strategy_bytes(b"version 2", "fvg_v1.py", "python")
    # No " (2)" duplicate should have been created.
    names = [s.name for s in library.list_saved_strategies("python")]
    assert names == ["fvg_v1.py"]
    assert library.load_strategy_text("python", "fvg_v1.py") == "version 1"


def test_saving_same_name_with_overwrite_replaces_content():
    library.save_strategy_bytes(b"version 1", "fvg_v1.py", "python")
    library.save_strategy_bytes(b"version 2", "fvg_v1.py", "python", overwrite=True)
    names = [s.name for s in library.list_saved_strategies("python")]
    assert names == ["fvg_v1.py"]
    assert library.load_strategy_text("python", "fvg_v1.py") == "version 2"


def test_save_strategy_text_overwrite_flag_respected():
    library.save_strategy_text("old", "strat.py", "python")
    with pytest.raises(library.StrategyAlreadyExists):
        library.save_strategy_text("new", "strat.py", "python")
    library.save_strategy_text("new", "strat.py", "python", overwrite=True)
    assert library.load_strategy_text("python", "strat.py") == "new"


def test_strategy_exists_checks_before_saving():
    assert not library.strategy_exists("python", "fvg_v1.py")
    library.save_strategy_bytes(b"x", "fvg_v1.py", "python")
    assert library.strategy_exists("python", "fvg_v1.py")
    # Works with or without the extension already included.
    assert library.strategy_exists("python", "fvg_v1")


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


def test_save_strategy_path_raises_on_collision_with_different_file(tmp_path):
    library.save_strategy_bytes(b"original", "external_strategy.py", "python")
    src = tmp_path / "external_strategy.py"
    src.write_text("a different file, same name")
    with pytest.raises(library.StrategyAlreadyExists):
        library.save_strategy_path(src, "python")


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


def test_delete_saved_strategy_also_removes_metadata_sidecar():
    library.save_strategy_bytes(b"print(1)", "with_meta.py", "python")
    library.save_strategy_metadata("python", "with_meta.py", {"description": "test"})
    meta_path = library.get_strategy_library_dir("python") / "with_meta.py.meta.json"
    assert meta_path.exists()
    library.delete_saved_strategy("python", "with_meta.py")
    assert not meta_path.exists()


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


# ---------------------------------------------------------------------------
# Metadata sidecars
# ---------------------------------------------------------------------------

def test_metadata_defaults_to_empty_dict():
    library.save_strategy_bytes(b"x", "no_meta.py", "python")
    assert library.load_strategy_metadata("python", "no_meta.py") == {}


def test_save_and_load_metadata():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.save_strategy_metadata("python", "meta.py", {
        "description": "NY liquidity sweep + FVG entry",
        "market": "XAUUSD",
        "timeframe": "15m",
        "tags": ["fvg", "liquidity"],
    })
    meta = library.load_strategy_metadata("python", "meta.py")
    assert meta["description"] == "NY liquidity sweep + FVG entry"
    assert meta["market"] == "XAUUSD"
    assert meta["tags"] == ["fvg", "liquidity"]


def test_metadata_merge_preserves_other_keys():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.save_strategy_metadata("python", "meta.py", {"description": "desc"})
    library.save_strategy_metadata("python", "meta.py", {"market": "EURUSD"})
    meta = library.load_strategy_metadata("python", "meta.py")
    assert meta["description"] == "desc"
    assert meta["market"] == "EURUSD"


def test_metadata_merge_false_replaces_wholesale():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.save_strategy_metadata("python", "meta.py", {"description": "desc", "market": "EURUSD"})
    library.save_strategy_metadata("python", "meta.py", {"market": "XAUUSD"}, merge=False)
    meta = library.load_strategy_metadata("python", "meta.py")
    assert meta == {"market": "XAUUSD"}


def test_record_backtest_result_stores_last_run_block():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.record_backtest_result("python", "meta.py", {
        "trades": 173, "net_profit": 34973.31, "win_rate": 51.7,
    })
    meta = library.load_strategy_metadata("python", "meta.py")
    assert meta["last_run"]["trades"] == 173
    assert meta["last_run"]["net_profit"] == 34973.31


def test_record_backtest_result_does_not_clobber_description():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.save_strategy_metadata("python", "meta.py", {"description": "my strategy"})
    library.record_backtest_result("python", "meta.py", {"trades": 10})
    meta = library.load_strategy_metadata("python", "meta.py")
    assert meta["description"] == "my strategy"
    assert meta["last_run"]["trades"] == 10


def test_list_saved_strategies_includes_metadata():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    library.save_strategy_metadata("python", "meta.py", {"market": "XAUUSD"})
    items = library.list_saved_strategies("python")
    assert items[0].metadata["market"] == "XAUUSD"


def test_corrupt_metadata_file_does_not_crash_listing():
    library.save_strategy_bytes(b"x", "meta.py", "python")
    meta_path = library.get_strategy_library_dir("python") / "meta.py.meta.json"
    meta_path.write_text("{not valid json")
    items = library.list_saved_strategies("python")
    assert items[0].metadata == {}


# ---------------------------------------------------------------------------
# Search / filter
# ---------------------------------------------------------------------------

def test_search_matches_filename():
    library.save_strategy_bytes(b"x", "fvg_v1.py", "python")
    library.save_strategy_bytes(b"x", "orb_v1.py", "python")
    results = library.list_saved_strategies("python", query="fvg")
    assert [s.name for s in results] == ["fvg_v1.py"]


def test_search_matches_metadata_description_and_market():
    library.save_strategy_bytes(b"x", "a.py", "python")
    library.save_strategy_metadata("python", "a.py", {"description": "gold scalper", "market": "XAUUSD"})
    library.save_strategy_bytes(b"x", "b.py", "python")
    library.save_strategy_metadata("python", "b.py", {"description": "fx swing", "market": "EURUSD"})

    assert [s.name for s in library.list_saved_strategies("python", query="gold")] == ["a.py"]
    assert [s.name for s in library.list_saved_strategies("python", query="xauusd")] == ["a.py"]
    assert [s.name for s in library.list_saved_strategies("python", query="swing")] == ["b.py"]


def test_search_is_case_insensitive_and_empty_query_returns_all():
    library.save_strategy_bytes(b"x", "FVG_v1.py", "python")
    assert [s.name for s in library.list_saved_strategies("python", query="fvg")] == ["FVG_v1.py"]
    assert len(library.list_saved_strategies("python", query="")) == 1


def test_search_across_all_types():
    library.save_strategy_bytes(b"x", "gold_strategy.py", "python")
    library.save_strategy_bytes(b"x", "gold_strategy.pine", "pinescript")
    results = library.list_saved_strategies(query="gold")
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

def test_rename_saved_strategy():
    library.save_strategy_bytes(b"content", "old_name.py", "python")
    new_path = library.rename_saved_strategy("python", "old_name.py", "new_name.py")
    assert new_path.name == "new_name.py"
    names = [s.name for s in library.list_saved_strategies("python")]
    assert names == ["new_name.py"]
    assert library.load_strategy_text("python", "new_name.py") == "content"


def test_rename_appends_extension_if_missing():
    library.save_strategy_bytes(b"content", "old_name.py", "python")
    new_path = library.rename_saved_strategy("python", "old_name.py", "new_name")
    assert new_path.name == "new_name.py"


def test_rename_moves_metadata_sidecar():
    library.save_strategy_bytes(b"content", "old_name.py", "python")
    library.save_strategy_metadata("python", "old_name.py", {"description": "keep me"})
    library.rename_saved_strategy("python", "old_name.py", "new_name.py")
    meta = library.load_strategy_metadata("python", "new_name.py")
    assert meta["description"] == "keep me"
    old_meta_path = library.get_strategy_library_dir("python") / "old_name.py.meta.json"
    assert not old_meta_path.exists()


def test_rename_to_existing_name_raises_without_overwrite():
    library.save_strategy_bytes(b"a", "a.py", "python")
    library.save_strategy_bytes(b"b", "b.py", "python")
    with pytest.raises(library.StrategyAlreadyExists):
        library.rename_saved_strategy("python", "a.py", "b.py")
    # Neither file should have been touched.
    assert library.load_strategy_text("python", "a.py") == "a"
    assert library.load_strategy_text("python", "b.py") == "b"


def test_rename_to_existing_name_with_overwrite_replaces_it():
    library.save_strategy_bytes(b"a", "a.py", "python")
    library.save_strategy_bytes(b"b", "b.py", "python")
    library.rename_saved_strategy("python", "a.py", "b.py", overwrite=True)
    names = sorted(s.name for s in library.list_saved_strategies("python"))
    assert names == ["b.py"]
    assert library.load_strategy_text("python", "b.py") == "a"


def test_rename_to_same_name_is_a_no_op():
    library.save_strategy_bytes(b"content", "same.py", "python")
    result = library.rename_saved_strategy("python", "same.py", "same.py")
    assert result.name == "same.py"
    assert len(library.list_saved_strategies("python")) == 1


def test_rename_missing_strategy_raises():
    with pytest.raises(FileNotFoundError):
        library.rename_saved_strategy("python", "ghost.py", "new.py")


# ---------------------------------------------------------------------------
# Export / backup
# ---------------------------------------------------------------------------

def test_export_library_zip_bytes_contains_all_saved_strategies():
    library.save_strategy_bytes(b"py content", "a.py", "python")
    library.save_strategy_bytes(b"pine content", "b.pine", "pinescript")
    library.save_strategy_metadata("python", "a.py", {"description": "test"})

    data = library.export_library_zip_bytes()
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert "strategies/python/a.py" in names
        assert "strategies/python/a.py.meta.json" in names
        assert "strategies/pinescript/b.pine" in names
        assert zf.read("strategies/python/a.py") == b"py content"


def test_export_library_zip_writes_to_disk(tmp_path):
    library.save_strategy_bytes(b"content", "a.py", "python")
    dest = tmp_path / "backup" / "strategies_backup.zip"
    result = library.export_library_zip(dest)
    assert result == dest
    assert dest.exists()
    with zipfile.ZipFile(dest) as zf:
        assert "strategies/python/a.py" in zf.namelist()


def test_export_library_zip_empty_library_still_produces_valid_zip():
    data = library.export_library_zip_bytes()
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.namelist() == []
