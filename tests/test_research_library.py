"""Tests for app.ai.research_library -- covers chunking, mtime-based
re-indexing, and the keyword-overlap relevance scoring, using plain .txt
files so no pypdf dependency is needed for the test suite itself."""
from __future__ import annotations

import time

import pytest

from app.ai import research_library


@pytest.fixture(autouse=True)
def _isolated_research_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.research_library._project_root", lambda: tmp_path)
    research_library._cache.clear()
    yield
    research_library._cache.clear()


def _write(name: str, text: str):
    path = research_library.research_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_research_dir_returns_no_excerpts():
    assert research_library.find_relevant_excerpts("mean reversion") == []


def test_list_research_files_finds_txt_and_md_only_supported_types():
    _write("paper1.txt", "some content")
    _write("paper2.md", "some content")
    _write("ignore_me.docx", "some content")
    names = {p.name for p in research_library.list_research_files()}
    assert names == {"paper1.txt", "paper2.md"}


def test_build_index_chunks_long_paragraphs_separately():
    long_para_a = "Momentum strategies exploit trend persistence. " * 40
    long_para_b = "Mean reversion strategies exploit price overreaction. " * 40
    _write("paper.txt", long_para_a + "\n\n" + long_para_b)
    chunks, stats = research_library.build_index()
    assert stats.files_indexed == 1
    assert stats.chunks_indexed >= 1
    assert all(len(c.text) <= research_library.CHUNK_TARGET_CHARS * 2 for c in chunks)


def test_short_fragments_are_dropped():
    _write("paper.txt", "Page 3\n\n" + "Real content. " * 30)
    chunks, _ = research_library.build_index()
    assert not any(c.text.strip() == "Page 3" for c in chunks)


def test_relevant_excerpts_favor_matching_content():
    _write("momentum.txt", "Momentum strategies exploit trend persistence in trending markets. " * 15)
    _write("unrelated.txt", "Bread baking requires careful attention to yeast fermentation times. " * 15)
    results = research_library.find_relevant_excerpts("momentum trend persistence strategy")
    assert results
    assert results[0]["source"] == "momentum.txt"


def test_query_with_no_usable_terms_returns_empty():
    _write("paper.txt", "Momentum strategies exploit trend persistence. " * 15)
    assert research_library.find_relevant_excerpts("the a an of") == []


def test_reindex_picks_up_file_changes_via_mtime(tmp_path):
    path = _write("paper.txt", "Momentum strategies exploit trend persistence. " * 15)
    research_library.build_index()
    # Force a distinct mtime (filesystem mtime resolution can be coarse).
    new_time = time.time() + 5
    path.write_text("Completely different content about seasonality effects. " * 15, encoding="utf-8")
    import os
    os.utime(path, (new_time, new_time))
    chunks, _ = research_library.build_index()
    assert any("seasonality" in c.text.lower() for c in chunks)
    assert not any("momentum" in c.text.lower() for c in chunks)


def test_removed_file_drops_out_of_the_index():
    path = _write("paper.txt", "Momentum strategies exploit trend persistence. " * 15)
    research_library.build_index()
    path.unlink()
    chunks, stats = research_library.build_index()
    assert stats.chunks_indexed == 0
    assert path.as_posix() not in [str(k) for k in research_library._cache.keys()]


def test_unreadable_pdf_is_skipped_not_raised(tmp_path):
    # A .pdf that isn't actually valid PDF bytes -- must not raise.
    path = research_library.research_dir() / "broken.pdf"
    path.write_bytes(b"not a real pdf")
    chunks, stats = research_library.build_index()
    assert "broken.pdf" in stats.files_skipped
    assert chunks == []


def test_max_excerpts_is_respected():
    for i in range(5):
        _write(f"paper{i}.txt", f"Momentum strategy variant {i} exploits trend persistence. " * 15)
    results = research_library.find_relevant_excerpts("momentum trend persistence", max_excerpts=2)
    assert len(results) <= 2
