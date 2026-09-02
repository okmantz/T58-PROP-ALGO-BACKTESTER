"""Tests for app.search.family_diversity."""
from __future__ import annotations

from app.search.family_diversity import (
    best_families,
    enforce_family_diversity,
    render_family_report,
    summarize_family_performance,
)


def _rec(candidate_id, family, score, source_type="manual", **extra):
    return {
        "candidate_id": candidate_id, "family": family, "source_type": source_type,
        "config": {}, "composite_score": score, **extra,
    }


def test_enforce_family_diversity_caps_each_family_independently():
    records = [
        _rec("a1", "trend_breakout", 90), _rec("a2", "trend_breakout", 80),
        _rec("a3", "trend_breakout", 70), _rec("a4", "trend_breakout", 60),
        _rec("b1", "mean_reversion_band", 50),
    ]
    kept, dropped = enforce_family_diversity(records, max_per_family=2)
    kept_ids = {r["candidate_id"] for r in kept}
    assert kept_ids == {"a1", "a2", "b1"}
    assert {r["candidate_id"] for r in dropped} == {"a3", "a4"}


def test_enforce_family_diversity_keeps_all_when_under_the_cap():
    records = [_rec("a1", "trend_breakout", 90), _rec("b1", "mean_reversion_band", 50)]
    kept, dropped = enforce_family_diversity(records, max_per_family=5)
    assert len(kept) == 2
    assert dropped == []


def test_enforce_family_diversity_rejects_non_positive_cap():
    import pytest
    with pytest.raises(ValueError):
        enforce_family_diversity([], max_per_family=0)


def test_enforce_family_diversity_treats_missing_score_as_worst_not_dropped():
    records = [
        _rec("a1", "trend_breakout", 10),
        {"candidate_id": "a2", "family": "trend_breakout", "source_type": "manual", "config": {}},
    ]
    kept, dropped = enforce_family_diversity(records, max_per_family=1)
    assert [r["candidate_id"] for r in kept] == ["a1"]
    assert [r["candidate_id"] for r in dropped] == ["a2"]


def test_summarize_family_performance_ranks_best_family_first():
    records = [
        _rec("a1", "trend_breakout", 40), _rec("a2", "trend_breakout", 60),
        _rec("b1", "liquidity_sweep_reversal", 90),
        _rec("c1", "vwap_reversion", None),
    ]
    summaries = summarize_family_performance(records)
    assert summaries[0].group == "liquidity_sweep"
    assert summaries[0].best_score == 90
    assert summaries[0].best_candidate_id == "b1"
    breakout = next(s for s in summaries if s.group == "breakout")
    assert breakout.n_candidates == 2
    assert breakout.median_score == 40
    vwap = next(s for s in summaries if s.group == "vwap")
    assert vwap.best_score is None
    # An unscored family must sort AFTER every scored family, never dropped.
    assert summaries[-1].group == "vwap"


def test_best_families_only_returns_scored_entries():
    records = [_rec("a1", "trend_breakout", 40), _rec("c1", "vwap_reversion", None)]
    summaries = summarize_family_performance(records)
    top = best_families(summaries, top_n=2)
    assert len(top) == 1
    assert top[0].group == "breakout"


def test_render_family_report_includes_headline_sentence():
    records = [_rec("a1", "trend_breakout", 40), _rec("b1", "liquidity_sweep_reversal", 90)]
    summaries = summarize_family_performance(records)
    report = render_family_report(summaries)
    assert "Best strategy family for this market/data:" in report
    assert "Liquidity Sweep" in report
