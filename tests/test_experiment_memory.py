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


# ---------------------------------------------------------------------------
# Strategy DNA dedup helpers (get_dna_tagsets_by_verdict / is_dna_tagset_previously_discarded)
# ---------------------------------------------------------------------------

def test_get_dna_tagsets_by_verdict_returns_only_matching_verdict():
    _record(name="Discarded1", verdict="DISCARD", config={"dna": ["entry.liquidity", "exit.atr"]})
    _record(name="Kept1", verdict="KEEP", config={"dna": ["entry.momentum"]})
    tagsets = experiment_memory.get_dna_tagsets_by_verdict(verdict="DISCARD")
    assert ["entry.liquidity", "exit.atr"] in tagsets
    assert ["entry.momentum"] not in tagsets


def test_get_dna_tagsets_by_verdict_skips_experiments_without_dna():
    _record(name="NoDNA", verdict="DISCARD")  # no config passed -> config={}
    _record(name="WithDNA", verdict="DISCARD", config={"dna": ["risk.adaptive"]})
    tagsets = experiment_memory.get_dna_tagsets_by_verdict(verdict="DISCARD")
    assert tagsets == [["risk.adaptive"]]


def test_get_dna_tagsets_by_verdict_filters_by_origin():
    _record(name="A", verdict="DISCARD", origin="research_loop", config={"dna": ["entry.liquidity"]})
    _record(name="B", verdict="DISCARD", origin="full_pipeline", config={"dna": ["entry.momentum"]})
    tagsets = experiment_memory.get_dna_tagsets_by_verdict(origin="research_loop", verdict="DISCARD")
    assert tagsets == [["entry.liquidity"]]


def test_is_dna_tagset_previously_discarded_detects_exact_match():
    _record(name="Discarded1", verdict="DISCARD", config={"dna": ["entry.liquidity", "exit.atr", "risk.adaptive"]})
    is_repeat, similarity = experiment_memory.is_dna_tagset_previously_discarded(
        ["entry.liquidity", "exit.atr", "risk.adaptive"]
    )
    assert is_repeat is True
    assert similarity == pytest.approx(1.0)


def test_is_dna_tagset_previously_discarded_detects_near_match_via_jaccard():
    _record(name="Discarded1", verdict="DISCARD", config={"dna": ["entry.liquidity", "exit.atr", "risk.adaptive", "entry.volatility"]})
    # 3 tags shared, 1 extra in the recorded set (a 4th tag not present in
    # the new candidate) -> intersection=3, union=4 -> jaccard = 0.75,
    # below the default 0.85 threshold but well above 0.5.
    is_repeat, similarity = experiment_memory.is_dna_tagset_previously_discarded(
        ["entry.liquidity", "exit.atr", "risk.adaptive"], min_jaccard=0.85,
    )
    assert is_repeat is False
    assert similarity == pytest.approx(0.75)
    # lowering the threshold should flip it to a repeat
    is_repeat_loose, _ = experiment_memory.is_dna_tagset_previously_discarded(
        ["entry.liquidity", "exit.atr", "risk.adaptive"], min_jaccard=0.5,
    )
    assert is_repeat_loose is True


def test_is_dna_tagset_previously_discarded_false_when_nothing_recorded():
    is_repeat, similarity = experiment_memory.is_dna_tagset_previously_discarded(["entry.liquidity"])
    assert is_repeat is False
    assert similarity == 0.0


def test_is_dna_tagset_previously_discarded_false_for_empty_tags():
    _record(name="Discarded1", verdict="DISCARD", config={"dna": ["entry.liquidity"]})
    is_repeat, similarity = experiment_memory.is_dna_tagset_previously_discarded([])
    assert is_repeat is False
    assert similarity == 0.0


def test_is_dna_tagset_previously_discarded_ignores_kept_experiments():
    _record(name="Kept1", verdict="KEEP", config={"dna": ["entry.liquidity", "exit.atr"]})
    is_repeat, similarity = experiment_memory.is_dna_tagset_previously_discarded(["entry.liquidity", "exit.atr"])
    assert is_repeat is False
    assert similarity == 0.0


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def test_get_experiment_by_id_round_trips():
    exp_id = _record(name="Findable", eval_pass_probability=55.0)
    exp = experiment_memory.get_experiment_by_id(exp_id)
    assert exp is not None
    assert exp.strategy_name == "Findable"
    assert exp.eval_pass_probability == 55.0


def test_get_experiment_by_id_returns_none_for_unknown_id():
    assert experiment_memory.get_experiment_by_id("does-not-exist") is None


def test_get_leaderboard_sorts_by_eval_pass_probability_descending():
    _record(name="Low", eval_pass_probability=20.0)
    _record(name="High", eval_pass_probability=80.0)
    _record(name="Mid", eval_pass_probability=50.0)
    board = experiment_memory.get_leaderboard(limit=10)
    assert [e.strategy_name for e in board] == ["High", "Mid", "Low"]


def test_get_leaderboard_filters_by_verdict_and_origin():
    _record(name="A", origin="research_loop", verdict="KEEP", eval_pass_probability=60.0)
    _record(name="B", origin="research_loop", verdict="DISCARD", eval_pass_probability=90.0)
    _record(name="C", origin="full_pipeline", verdict="KEEP", eval_pass_probability=70.0)
    board = experiment_memory.get_leaderboard(origin="research_loop", verdict="KEEP")
    assert [e.strategy_name for e in board] == ["A"]


def test_get_leaderboard_falls_back_to_default_sort_for_unknown_column():
    _record(name="Low", eval_pass_probability=20.0)
    _record(name="High", eval_pass_probability=80.0)
    board = experiment_memory.get_leaderboard(sort_by="not_a_real_column; DROP TABLE experiments;")
    assert [e.strategy_name for e in board] == ["High", "Low"]


def test_render_leaderboard_card_without_parent():
    exp_id = _record(
        name="Liquidity Reclaim V7", verdict="KEEP", eval_pass_probability=84.0,
        first_payout_probability=71.0, max_drawdown_pct=4.7,
    )
    exp = experiment_memory.get_experiment_by_id(exp_id)
    card = experiment_memory.render_leaderboard_card(exp)
    assert "Liquidity Reclaim V7" in card
    assert "Pass: 84.0%" in card
    assert "Payout: 71.0%" in card
    assert "DD: 4.7%" in card
    assert "Verdict:" in card
    assert "KEEP" in card
    assert "Compared with parent" not in card


def test_render_leaderboard_card_with_parent_shows_diff_and_deltas():
    parent_id = _record(
        name="Liquidity Reclaim V6", verdict="DISCARD",
        eval_pass_probability=75.0, first_payout_probability=57.0, max_drawdown_pct=6.8,
        config={"dna": ["entry.liquidity", "exit.fixed_rr", "risk.fixed_pct"]},
    )
    child_id = _record(
        name="Liquidity Reclaim V7", verdict="KEEP",
        eval_pass_probability=84.0, first_payout_probability=71.0, max_drawdown_pct=4.7,
        config={"dna": ["entry.liquidity", "exit.atr", "risk.adaptive"], "parent_experiment": parent_id},
    )
    child = experiment_memory.get_experiment_by_id(child_id)
    parent = experiment_memory.get_experiment_by_id(parent_id)
    card = experiment_memory.render_leaderboard_card(child, parent=parent)

    assert "Changes:" in card
    assert "+ atr" in card
    assert "+ adaptive" in card
    assert "- fixed rr" in card
    assert "- fixed pct" in card
    assert "Compared with parent:" in card
    assert "Pass: +9.0%" in card
    assert "Payout: +14.0%" in card
    assert "DD: -2.1%" in card


def test_render_leaderboard_card_includes_lesson_when_present():
    exp_id = _record(name="X", lesson="73% of losses happened in low-volatility regimes.")
    exp = experiment_memory.get_experiment_by_id(exp_id)
    card = experiment_memory.render_leaderboard_card(exp)
    assert "Lesson:" in card
    assert "low-volatility regimes" in card


def test_find_similar_discarded_dna_wraps_jaccard_check():
    _record(name="Discarded1", verdict="DISCARD", config={"dna": ["entry.liquidity", "exit.atr", "risk.adaptive"]})
    assert experiment_memory.find_similar_discarded_dna(
        ["entry.liquidity", "exit.atr", "risk.adaptive"],
    ) is True
    assert experiment_memory.find_similar_discarded_dna(["entry.momentum"]) is False
