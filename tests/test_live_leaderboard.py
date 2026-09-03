"""Tests for the live-leaderboard polling additions to app.search.results_db."""
from __future__ import annotations

from app.search.results_db import ResultsDB, poll_live_leaderboard


def _record(**overrides):
    record = {
        "family": "trend_breakout",
        "source_type": "manual",
        "config": {"name": "test config"},
        "quick_score": 1.0,
        "fitness": None,
        "composite_score": None,
        "passed_stage1": 1,
        "passed_stage2": 0,
        "passed_stage3_gate": 0,
        "error": None,
    }
    record.update(overrides)
    return record


def test_live_leaderboard_shows_furthest_stage_reached_per_candidate(tmp_path):
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "XAUUSD", "15m", 10, {})
        # cand-1 only reached Stage 1.
        db.insert_candidate("run1", "cand-1", "stage1", _record(quick_score=1.0))
        # cand-2 reached Stage 2 -- should show its stage2 row, not a stage1 one.
        db.insert_candidate("run1", "cand-2", "stage1", _record(quick_score=5.0))
        db.insert_candidate("run1", "cand-2", "stage2", _record(quick_score=5.0, fitness=42.0))
        board = db.live_leaderboard("run1", top_n=10)

    by_id = {r["candidate_id"]: r for r in board}
    assert by_id["cand-1"]["stage"] == "stage1"
    assert by_id["cand-2"]["stage"] == "stage2"
    assert by_id["cand-2"]["fitness"] == 42.0


def test_live_leaderboard_ranks_by_best_available_score(tmp_path):
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "XAUUSD", "15m", 10, {})
        db.insert_candidate("run1", "low", "stage3", _record(composite_score=1.0))
        db.insert_candidate("run1", "high", "stage3", _record(composite_score=99.0))
        board = db.live_leaderboard("run1", top_n=10)
    assert [r["candidate_id"] for r in board][0] == "high"


def test_poll_live_leaderboard_matches_direct_query(tmp_path):
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "XAUUSD", "15m", 10, {})
        db.insert_candidate("run1", "cand-1", "stage1", _record(quick_score=3.0))

    polled = poll_live_leaderboard(db_path, "run1", top_n=10)
    assert len(polled) == 1
    assert polled[0]["candidate_id"] == "cand-1"


def test_poll_live_leaderboard_returns_empty_for_missing_db(tmp_path):
    assert poll_live_leaderboard(tmp_path / "does_not_exist.db", "run1") == []
