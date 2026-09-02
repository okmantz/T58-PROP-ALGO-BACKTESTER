"""Tests for app.reports.survival_report (the Payout Probability report)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, run_prop_survival_analysis
from app.reports.survival_report import (
    build_survival_report, export_survival_html, export_survival_json, generate_survival_report,
)


def _mock_trades(n=250, seed=1, mean=25.0, std=180.0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    trades = []
    for i in range(n):
        t = base + pd.Timedelta(days=i // 3)
        pnl = float(rng.normal(mean, std))
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _rules():
    return PropRules(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=3, consistency_rule_pct=30, payout_frequency_days=14,
    )


@pytest.fixture()
def survival_result():
    trades = _mock_trades(300)
    cfg = PropSurvivalConfig(n_simulations=300, life_simulations=100, random_seed=1, max_payouts_tracked=4)
    return run_prop_survival_analysis(trades, _rules(), cfg), cfg


def test_build_survival_report_is_json_serializable(survival_result):
    result, cfg = survival_result
    report = build_survival_report(result, "Test Strategy", "XAUUSD15", _rules(), cfg)
    # Must round-trip through json.dumps/loads without a custom encoder
    # error -- this is what export_survival_json relies on.
    json.loads(json.dumps(report, default=str))
    assert report["strategy_name"] == "Test Strategy"
    assert report["result"]["funnel"]["n_accounts"] == cfg.n_simulations


def test_export_survival_json_writes_file(tmp_path, survival_result):
    result, cfg = survival_result
    report = build_survival_report(result, "Test Strategy", "XAUUSD15", _rules(), cfg)
    path = export_survival_json(report, tmp_path / "survival.json")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["result"]["prop_survival_score"] == pytest.approx(result.prop_survival_score)


def test_export_survival_html_contains_funnel_stages(tmp_path, survival_result):
    result, cfg = survival_result
    report = build_survival_report(result, "Test Strategy", "XAUUSD15", _rules(), cfg)
    path = export_survival_html(report, tmp_path / "survival.html")
    html = path.read_text()
    assert "Payout Probability Report" in html
    assert "Passed Evaluation" in html
    assert "Reached Funded" in html
    assert "Payout #1" in html
    assert "Payout #4" in html
    assert "<svg" in html


def test_generate_survival_report_returns_both_paths(tmp_path, survival_result):
    result, cfg = survival_result
    paths = generate_survival_report(tmp_path, result, "Test Strategy", "XAUUSD15", _rules(), cfg)
    assert paths["html"].exists()
    assert paths["json"].exists()


def test_report_notes_are_rendered_when_present(tmp_path):
    # A deliberately bad (losing) strategy should trigger at least one
    # of survival_engine's warning notes, and the report must surface it.
    trades = _mock_trades(150, mean=-100.0, std=50.0)
    cfg = PropSurvivalConfig(n_simulations=150, life_simulations=80, random_seed=2)
    result = run_prop_survival_analysis(trades, _rules(), cfg)
    assert result.notes  # sanity check on the fixture itself
    report = build_survival_report(result, "Losing Strategy", "", _rules(), cfg)
    path = export_survival_html(report, tmp_path / "survival.html")
    html = path.read_text()
    assert "Read before trusting these numbers" in html
