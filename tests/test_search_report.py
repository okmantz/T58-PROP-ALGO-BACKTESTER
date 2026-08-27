"""Tests for app.search.search_report."""
from __future__ import annotations

import json

from app.search.batch_runner import SearchSummary
from app.search.search_report import generate_search_report
from app.search.strategy_space import generate_search_space


def _fake_leaderboard_row(candidate_id, composite_score, passed=True):
    return {
        "candidate_id": candidate_id,
        "family": "trend_breakout",
        "composite_score": composite_score,
        "deflated_sharpe": {"probabilistic_sharpe": 0.62, "n_trials": 40},
        "statistics": {"net_profit": 250.0, "profit_factor": 1.3, "win_rate": 55.0, "total_trades": 80},
        "mc_summary": {"evaluation_pass_probability": 61.0, "first_payout_probability": 40.0, "risk_of_ruin_pct": 12.0},
        "passed_stage3_gate": passed,
        "gate_notes": "" if passed else "Walk-forward efficiency below threshold.",
    }


def _summary(leaderboard, champion_id=None):
    return SearchSummary(
        run_id="abc123def456", mode="family", family="trend_breakout",
        total_candidates=40, stage1_survivors=12, stage2_survivors=5, stage3_survivors=5,
        champion_candidate_id=champion_id, elapsed_seconds=42.5, db_path="/tmp/x.db",
        leaderboard=leaderboard,
    )


def test_report_written_with_champion(tmp_path):
    space = generate_search_space("family", family="trend_breakout", max_candidates=5, seed=1)
    board = [_fake_leaderboard_row("trend_breakout-aaa", 55.0, passed=True),
             _fake_leaderboard_row("trend_breakout-bbb", 30.0, passed=False)]
    summary = _summary(board, champion_id="trend_breakout-aaa")

    paths = generate_search_report(str(tmp_path), summary, space, instrument="EURUSD", timeframe="5m")
    assert paths["html"].exists()
    assert paths["json"].exists()

    html = paths["html"].read_text(encoding="utf-8")
    assert "trend_breakout-aaa" in html
    assert "Champion" in html
    assert "PASSED" in html and "FAILED" in html

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["run_id"] == "abc123def456"
    assert data["champion_candidate_id"] == "trend_breakout-aaa"
    assert len(data["leaderboard"]) == 2


def test_report_handles_no_champion(tmp_path):
    space = generate_search_space("family", family="trend_breakout", max_candidates=3, seed=1)
    board = [_fake_leaderboard_row("trend_breakout-ccc", 10.0, passed=False)]
    summary = _summary(board, champion_id=None)

    paths = generate_search_report(str(tmp_path), summary, space)
    html = paths["html"].read_text(encoding="utf-8")
    assert "No candidate passed every Stage 3 gate" in html


def test_report_handles_empty_leaderboard(tmp_path):
    space = generate_search_space("family", family="trend_breakout", max_candidates=3, seed=1)
    summary = _summary([], champion_id=None)
    paths = generate_search_report(str(tmp_path), summary, space)
    html = paths["html"].read_text(encoding="utf-8")
    assert "No candidates reached Stage 3" in html


def test_report_filenames_are_scoped_to_run_id(tmp_path):
    space = generate_search_space("family", family="trend_breakout", max_candidates=3, seed=1)
    summary1 = _summary([], champion_id=None)
    summary2 = SearchSummary(**{**summary1.__dict__, "run_id": "different999"})
    p1 = generate_search_report(str(tmp_path), summary1, space)
    p2 = generate_search_report(str(tmp_path), summary2, space)
    assert p1["html"] != p2["html"]
    assert p1["html"].exists() and p2["html"].exists()
