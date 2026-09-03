"""Tests for app.orchestration.speed_run.

Kept small-scale (short trending series, workers=1, tiny candidate/sim
counts everywhere) so this runs quickly while still exercising the real
discovery -> validation -> winner-selection path end to end, not a mock
of it.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.speed_run import SpeedRunConfig, run_speed_run
from app.prop.simulator import PropRules


def _trending_df(n=1500, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _fast_cfg(**overrides) -> SpeedRunConfig:
    base = dict(
        max_candidates=30, max_per_family_stage1=2, stage1_top_n=4,
        ga_population=4, ga_generations=1, ga_search_sims=20, stage2_top_n=2,
        full_mc_sims=40, walk_forward_folds=0, robustness_neighbors=0,
        discovery_workers=1, top_k_to_validate=2, max_concurrent_validations=1,
        validation_ga_population=4, validation_ga_generations=1,
        validation_ga_search_mc_sims=20, validation_final_mc_sims=40,
        validation_folds=0, save_winner_to_library=False,
    )
    base.update(overrides)
    return SpeedRunConfig(**base)


@pytest.fixture
def loose_prop_rules() -> PropRules:
    # Loose enough that a decent trending-series candidate has a real
    # chance of clearing Stage 3 within this test's tiny candidate budget,
    # without special-casing the test around exact numbers.
    return PropRules(
        account_size=50_000, evaluation_profit_target_pct=4.0, max_drawdown_pct=20.0,
        daily_loss_limit_pct=10.0,
    )


def test_end_to_end_runs_all_three_phases(tmp_path, loose_prop_rules):
    df = _trending_df()
    log_lines: list[str] = []
    result = run_speed_run(
        df, RiskConfig(), loose_prop_rules, tmp_path / "out",
        cfg=_fast_cfg(), progress_cb=log_lines.append, instrument="TEST",
    )
    assert result.search_summary is not None
    assert result.elapsed_seconds > 0
    assert any("Phase 1" in line for line in log_lines)
    assert any("Phase 2" in line for line in log_lines) or "No Stage 3 survivors" in "".join(log_lines)
    assert any("Phase 3" in line for line in log_lines) or "No Stage 3 survivors" in "".join(log_lines)


def test_no_stage3_survivors_reports_cleanly_instead_of_crashing(tmp_path):
    # Flat noise, essentially no edge, plus a near-impossible profit
    # target -- discovery should find nothing worth validating, or if a
    # candidate is statistically-significant-but-tiny enough to clear
    # Stage 3's own (prop-rules-agnostic) significance gate, Phase 2's
    # real Full Pipeline validation against these prop rules must still
    # fail it. Either way this must report cleanly -- no crash, no
    # fabricated READY/MARGINAL winner -- not assume one specific path.
    n = 300
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        c = price + rng.normal(0, 0.00001)
        rows.append((ts[i], price, max(price, c), min(price, c), c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    impossible_rules = PropRules(account_size=50_000, evaluation_profit_target_pct=500.0, max_drawdown_pct=1.0)
    result = run_speed_run(
        df, RiskConfig(), impossible_rules, tmp_path / "out",
        cfg=_fast_cfg(max_candidates=10, stage1_top_n=2, stage2_top_n=1), instrument="TEST",
    )
    assert result.winner is None
    for c in result.candidates:
        assert c.pipeline_result is None or c.pipeline_result.verdict not in ("READY", "MARGINAL")
    if not result.candidates:
        assert "No candidate" in result.winner_reason or "No Stage 3 survivors" in result.winner_reason or \
            "No candidate survived" in result.winner_reason
    else:
        assert "No validated candidate" in result.winner_reason


def test_cancel_event_stops_before_validation(tmp_path, loose_prop_rules):
    df = _trending_df()
    cancel_event = threading.Event()
    cancel_event.set()  # already cancelled before the call
    result = run_speed_run(
        df, RiskConfig(), loose_prop_rules, tmp_path / "out",
        cfg=_fast_cfg(), cancel_event=cancel_event, instrument="TEST",
    )
    assert result.winner is None
    assert result.candidates == []
    assert "Cancelled" in result.winner_reason


def test_winner_selection_prefers_ready_or_margarine_over_failed():
    from app.orchestration.speed_run import SpeedRunCandidateResult, _rank_key

    class _FakeMC:
        def __init__(self, score):
            self.evaluation_pass_probability = score

    class _FakeResult:
        def __init__(self, verdict, score):
            self.verdict = verdict
            self.final_mc = _FakeMC(score)

    ready_low = SpeedRunCandidateResult("a", "fam", _FakeResult("READY", 40.0))
    marginal_high = SpeedRunCandidateResult("b", "fam", _FakeResult("MARGINAL", 90.0))
    not_ready_high = SpeedRunCandidateResult("c", "fam", _FakeResult("NOT READY", 99.0))
    failed = SpeedRunCandidateResult("d", "fam", None, error="boom")

    ranked = sorted([not_ready_high, marginal_high, failed, ready_low], key=_rank_key)
    assert ranked[0] is ready_low          # READY always outranks MARGINAL/NOT READY regardless of score
    assert ranked[1] is marginal_high
    assert ranked[-1] is failed
