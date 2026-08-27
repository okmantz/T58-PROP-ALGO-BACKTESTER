"""Tests for app.search.results_db."""
from __future__ import annotations

import pytest

from app.search.results_db import ResultsDB


def test_create_and_read_run(tmp_path):
    db_path = tmp_path / "results.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "EURUSD", "5m", 100, {"seed": 42})
        summary = db.run_summary("run1")
    assert summary["run_id"] == "run1"
    assert summary["mode"] == "family"
    assert summary["family"] == "trend_breakout"
    assert summary["total_candidates"] == 100
    assert summary["config"]["seed"] == 42
    assert summary["status"] == "running"
    assert summary["stage_counts"] == {"stage1": 0, "stage2": 0, "stage3": 0}


def test_run_summary_returns_none_for_unknown_run(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        assert db.run_summary("does-not-exist") is None


def test_finish_run_updates_status(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "single", None, "x", "y", 1, {})
        db.finish_run("run1", status="completed")
        summary = db.run_summary("run1")
    assert summary["status"] == "completed"
    assert summary["finished_at"] is not None


def _candidate_record(**overrides):
    record = {
        "family": "trend_breakout",
        "source_type": "manual",
        "config": {"name": "test config"},
        "params": {"lookback": 20},
        "statistics": {"net_profit": 123.4, "total_trades": 50},
        "prop_summary": {"evaluation_pass_pct": 100.0},
        "mc_summary": {"evaluation_pass_probability": 61.2},
        "quick_score": 3.5,
        "fitness": 42.0,
        "composite_score": 17.8,
        "passed_stage1": 1,
        "passed_stage2": 1,
        "passed_stage3_gate": 1,
        "gate_notes": "",
        "error": None,
    }
    record.update(overrides)
    return record


def test_insert_and_get_candidate_round_trips_json_fields(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-1", "stage1", _candidate_record())
        rec = db.get_candidate("cand-1", run_id="run1", stage="stage1")
    assert rec["config"] == {"name": "test config"}
    assert rec["params"] == {"lookback": 20}
    assert rec["statistics"]["net_profit"] == 123.4
    assert rec["composite_score"] == 17.8
    assert bool(rec["passed_stage3_gate"]) is True


def test_reinserting_same_candidate_same_stage_overwrites_not_duplicates(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-1", "stage1", _candidate_record(fitness=1.0))
        db.insert_candidate("run1", "cand-1", "stage1", _candidate_record(fitness=2.0))
        rec = db.get_candidate("cand-1", run_id="run1", stage="stage1")
        assert rec["fitness"] == 2.0
        count = db.count_stage("run1", "stage1")
    assert count == 1  # overwritten, not duplicated


def test_same_candidate_id_can_exist_at_different_stages(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-1", "stage1", _candidate_record(fitness=1.0))
        db.insert_candidate("run1", "cand-1", "stage3", _candidate_record(fitness=99.0))
        stage1_rec = db.get_candidate("cand-1", run_id="run1", stage="stage1")
        stage3_rec = db.get_candidate("cand-1", run_id="run1", stage="stage3")
    assert stage1_rec["fitness"] == 1.0
    assert stage3_rec["fitness"] == 99.0


def test_leaderboard_orders_by_composite_score_descending(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        for i, score in enumerate([5.0, 90.0, 30.0]):
            db.insert_candidate("run1", f"cand-{i}", "stage3", _candidate_record(composite_score=score))
        board = db.leaderboard("run1", stage="stage3", top_n=10)
    assert [r["composite_score"] for r in board] == [90.0, 30.0, 5.0]


def test_leaderboard_only_passed_filters_correctly(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-pass", "stage3", _candidate_record(composite_score=50.0, passed_stage3_gate=1))
        db.insert_candidate("run1", "cand-fail", "stage3", _candidate_record(composite_score=90.0, passed_stage3_gate=0))
        board = db.leaderboard("run1", stage="stage3", only_passed=True)
    assert len(board) == 1
    assert board[0]["candidate_id"] == "cand-pass"


def test_leaderboard_scoped_to_run_id(tmp_path):
    with ResultsDB(tmp_path / "r.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.create_run("run2", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-a", "stage3", _candidate_record(composite_score=10.0))
        db.insert_candidate("run2", "cand-b", "stage3", _candidate_record(composite_score=20.0))
        board1 = db.leaderboard("run1", stage="stage3")
        board2 = db.leaderboard("run2", stage="stage3")
    assert [r["candidate_id"] for r in board1] == ["cand-a"]
    assert [r["candidate_id"] for r in board2] == ["cand-b"]


def test_reopening_same_db_path_preserves_data(tmp_path):
    db_path = tmp_path / "persist.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 10, {})
        db.insert_candidate("run1", "cand-1", "stage1", _candidate_record())
    # Simulate a resumed process re-opening the same file.
    with ResultsDB(db_path) as db2:
        rec = db2.get_candidate("cand-1", run_id="run1", stage="stage1")
        summary = db2.run_summary("run1")
    assert rec is not None
    assert summary is not None
