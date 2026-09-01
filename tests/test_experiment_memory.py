"""Tests for app.ai.experiment_memory -- SQLite recording/querying and
the keyword-search fallback path (semantic search itself is exercised
indirectly via a stubbed OllamaEmbedder so no real Ollama is needed)."""
from __future__ import annotations

import pytest

from app.ai import experiment_memory
from app.ai.ollama_settings import OllamaSettings


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_memory, "_db_path", lambda: tmp_path / "experiments.db")
    # Keep semantic indexing off by default in these tests -- settings.enabled=False
    # means record_experiment/search_similar_experiments both take the plain
    # SQL/keyword path, which is the behavior every caller must be able to rely on
    # with zero Ollama setup.
    monkeypatch.setattr(experiment_memory, "load_settings", lambda: OllamaSettings(enabled=False))
    yield


def _record(name="Test Strategy", verdict="READY", **kwargs):
    defaults = dict(
        origin="full_pipeline", strategy_name=name, source_type="python", instrument="XAUUSD15",
        verdict=verdict, trades=100, net_profit=500.0, win_rate=55.0, profit_factor=1.3,
        max_drawdown_pct=8.0, eval_pass_probability=40.0, first_payout_probability=20.0,
        risk_of_ruin_pct=5.0, lesson="",
    )
    defaults.update(kwargs)
    return experiment_memory.record_experiment(**defaults)


def test_record_experiment_returns_an_id():
    exp_id = _record()
    assert exp_id is not None
    assert isinstance(exp_id, str)


def test_get_recent_experiments_returns_newest_first():
    _record(name="First")
    _record(name="Second")
    recent = experiment_memory.get_recent_experiments(limit=10)
    assert [e.strategy_name for e in recent] == ["Second", "First"]


def test_get_recent_experiments_filters_by_origin():
    _record(name="A", origin="full_pipeline")
    experiment_memory.record_experiment(
        origin="batch_test", strategy_name="B", source_type="python", instrument="XAUUSD15",
        verdict="TESTED", trades=10, net_profit=1.0, win_rate=50.0, profit_factor=1.0,
        max_drawdown_pct=1.0, eval_pass_probability=1.0, first_payout_probability=1.0,
        risk_of_ruin_pct=1.0,
    )
    only_batch = experiment_memory.get_recent_experiments(origin="batch_test")
    assert len(only_batch) == 1
    assert only_batch[0].strategy_name == "B"


def test_get_summary_counts_totals_and_breaks_down_by_verdict():
    _record(name="A", verdict="READY")
    _record(name="B", verdict="READY")
    _record(name="C", verdict="NOT READY")
    counts = experiment_memory.get_summary_counts()
    assert counts["total"] == 3
    assert counts["by_verdict"]["READY"] == 2
    assert counts["by_verdict"]["NOT READY"] == 1


def test_get_summary_counts_empty_db_is_all_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_memory, "_db_path", lambda: tmp_path / "fresh.db")
    counts = experiment_memory.get_summary_counts()
    assert counts == {"total": 0, "by_verdict": {}, "top_strategies": {}}


def test_search_similar_experiments_keyword_fallback_matches_strategy_name():
    _record(name="NY Liquidity Sweep FVG")
    _record(name="Unrelated EMA Crossover")
    hits = experiment_memory.search_similar_experiments("liquidity sweep")
    names = [h.strategy_name for h in hits]
    assert "NY Liquidity Sweep FVG" in names


def test_search_similar_experiments_with_no_data_returns_empty():
    hits = experiment_memory.search_similar_experiments("anything")
    assert hits == []


def test_record_experiment_stores_lesson_and_config():
    exp_id = _record(name="Lesson Test", lesson="Overfit to high-volatility regime", config={"period": 14})
    recent = experiment_memory.get_recent_experiments(limit=1)
    assert recent[0].lesson == "Overfit to high-volatility regime"
    assert recent[0].config == {"period": 14}


def test_record_experiment_never_raises_on_bad_db_path(monkeypatch):
    monkeypatch.setattr(experiment_memory, "_db_path", lambda: "/nonexistent/deeply/nested/path.db")
    result = _record()
    assert result is None


def test_search_similar_experiments_uses_semantic_search_when_available(monkeypatch, tmp_path):
    from app.ai import vector_store as vector_store_module

    monkeypatch.setattr(vector_store_module, "_store_dir", lambda: tmp_path)
    monkeypatch.setattr(experiment_memory, "load_settings", lambda: OllamaSettings(enabled=True, host="http://x"))

    class _FakeEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_one(self, text):
            # Deterministic fake embedding: encodes whether "sweep" is present.
            return ([1.0, 0.0] if "sweep" in text.lower() else [0.0, 1.0]), None

    monkeypatch.setattr(experiment_memory, "OllamaEmbedder", _FakeEmbedder)

    id1 = _record(name="Liquidity Sweep Strategy")
    id2 = _record(name="Totally Different Approach")
    assert id1 and id2

    hits = experiment_memory.search_similar_experiments("looking for a sweep-based strategy")
    assert hits
    assert hits[0].strategy_name == "Liquidity Sweep Strategy"
    assert hits[0].similarity == pytest.approx(1.0)
