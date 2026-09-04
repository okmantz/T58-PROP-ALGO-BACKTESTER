"""Tests for app.reports.strategy_state -- the persisted 'current
strategy' pointer and its validation checklist."""
from __future__ import annotations

import json

import pytest

from app.reports import strategy_state as ss


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    yield tmp_path


def test_get_current_strategy_defaults_to_none():
    assert ss.get_current_strategy() is None


def test_set_and_get_current_strategy_round_trips():
    ss.set_current_strategy("Regime-Gated Liquidity Reclaim", "ES", "5m")
    current = ss.get_current_strategy()
    assert current["strategy_name"] == "Regime-Gated Liquidity Reclaim"
    assert current["instrument"] == "ES"
    assert current["timeframe"] == "5m"
    assert "set_at" in current


def test_set_current_strategy_overwrites_previous():
    ss.set_current_strategy("Strategy A", "ES")
    ss.set_current_strategy("Strategy B", "NQ")
    current = ss.get_current_strategy()
    assert current["strategy_name"] == "Strategy B"
    assert current["instrument"] == "NQ"


def test_clear_current_strategy():
    ss.set_current_strategy("Strategy A", "ES")
    ss.clear_current_strategy()
    assert ss.get_current_strategy() is None


def test_clear_current_strategy_is_safe_when_nothing_set():
    ss.clear_current_strategy()  # must not raise
    assert ss.get_current_strategy() is None


def test_get_current_strategy_ignores_corrupt_file(tmp_path):
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "current_strategy.json").write_text("{not valid json")
    assert ss.get_current_strategy() is None


def test_checklist_defaults_every_kind_to_not_ran():
    checklist = ss.get_checklist("Strategy A", "ES")
    assert set(checklist.keys()) == set(ss.VALIDATION_KINDS)
    for check in checklist.values():
        assert check == {"ran": False, "passed": None, "summary": "", "report_html": ""}


def test_record_validation_updates_only_that_kind():
    ss.record_validation("Strategy A", "ES", "cpcv", passed=True, summary="robust", report_html="/cpcv_reports/x.html")
    checklist = ss.get_checklist("Strategy A", "ES")
    assert checklist["cpcv"]["ran"] is True
    assert checklist["cpcv"]["passed"] is True
    assert checklist["cpcv"]["summary"] == "robust"
    assert checklist["wfo"]["ran"] is False


def test_record_validation_with_no_verdict_records_ran_not_passed():
    ss.record_validation("Strategy A", "ES", "sensitivity", passed=None, summary="3 params swept")
    checklist = ss.get_checklist("Strategy A", "ES")
    assert checklist["sensitivity"]["ran"] is True
    assert checklist["sensitivity"]["passed"] is None


def test_record_validation_is_keyed_by_strategy_and_instrument():
    ss.record_validation("Strategy A", "ES", "cpcv", passed=True)
    other = ss.get_checklist("Strategy A", "NQ")
    assert other["cpcv"]["ran"] is False


def test_record_validation_key_matching_is_case_insensitive():
    ss.record_validation("Strategy A", "es", "cpcv", passed=True)
    checklist = ss.get_checklist("strategy a", "ES")
    assert checklist["cpcv"]["ran"] is True


def test_record_validation_ignores_unknown_kind():
    ss.record_validation("Strategy A", "ES", "not_a_real_kind", passed=True)  # must not raise
    checklist = ss.get_checklist("Strategy A", "ES")
    assert all(not c["ran"] for c in checklist.values())


def test_robustness_score_with_nothing_run():
    score = ss.robustness_score("Strategy A", "ES")
    assert score == {
        "ran_count": 0, "total_count": 5, "decided_count": 0,
        "passed_count": 0, "pct_run": 0.0, "pct_passed_of_decided": None,
    }


def test_robustness_score_counts_ran_and_passed_separately():
    ss.record_validation("Strategy A", "ES", "cpcv", passed=True)
    ss.record_validation("Strategy A", "ES", "wfo", passed=False)
    ss.record_validation("Strategy A", "ES", "sensitivity", passed=None)  # ran, no verdict
    score = ss.robustness_score("Strategy A", "ES")
    assert score["ran_count"] == 3
    assert score["decided_count"] == 2  # sensitivity excluded -- no verdict
    assert score["passed_count"] == 1
    assert score["pct_run"] == 60.0
    assert score["pct_passed_of_decided"] == 50.0


def test_record_validation_failure_never_raises(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(ss, "_save_checklist_store", _boom)
    ss.record_validation("Strategy A", "ES", "cpcv", passed=True)  # must not raise
