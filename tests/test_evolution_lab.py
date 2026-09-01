"""Tests for app.evolution -- PROP FITNESS scoring, the knowledge graph,
and a short smoke run of the full Evolution Lab generation loop against
synthetic data. These intentionally use tiny, cheap configs -- the point
is to catch wiring/regressions, not to evaluate real trading edges."""
import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution.engine import EvolutionConfig, EvolutionRunner
from app.evolution.knowledge_graph import KnowledgeGraph, feature_vector_for_spec
from app.evolution.prop_fitness import compute_prop_fitness
from app.prop.simulator import PropRules


def _trending_df(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })


def test_compute_prop_fitness_penalizes_thin_sample_and_concentration():
    stats = {"total_trades": 3, "max_drawdown_pct": 5.0, "max_losing_streak": 1}
    mc_summary = {"evaluation_pass_probability": 80.0, "first_payout_probability": 60.0}
    trade_pnls = [500.0, 10.0, 10.0]  # one trade dominates gross profit
    breakdown = compute_prop_fitness(stats, mc_summary, None, None, trade_pnls, min_trades_target=30)
    assert breakdown.penalty_too_few_trades > 0
    assert breakdown.penalty_concentration > 0
    assert breakdown.final_score < breakdown.base_score


def test_compute_prop_fitness_high_pbo_penalized():
    stats = {"total_trades": 50, "max_drawdown_pct": 5.0, "max_losing_streak": 2}
    mc_summary = {"evaluation_pass_probability": 70.0, "first_payout_probability": 50.0}
    trade_pnls = [10.0] * 50
    low_pbo = compute_prop_fitness(stats, mc_summary, None, None, trade_pnls, pbo=0.2)
    high_pbo = compute_prop_fitness(stats, mc_summary, None, None, trade_pnls, pbo=0.9)
    assert high_pbo.final_score < low_pbo.final_score


def test_knowledge_graph_round_trip_and_similarity(tmp_path):
    kg = KnowledgeGraph(tmp_path / "kg.jsonl")
    fv_a = {"family": "trend_breakout", "source_type": "manual", "uses_ema": True, "uses_rsi": False}
    fv_b = {"family": "trend_breakout", "source_type": "manual", "uses_ema": True, "uses_rsi": True}
    kg.record(fv_a, {"passed": True, "final_score": 12.0})
    kg.record(fv_b, {"passed": False, "final_score": -3.0})

    results = kg.query_similar(fv_a, top_k=5)
    assert results, "expected at least one similarity match"
    assert results[0][0] == pytest.approx(1.0)  # exact match to itself first

    desc = kg.describe(fv_a)
    assert "Similarity" in desc
    assert "Historical success rate" in desc


def test_knowledge_graph_empty_describe_is_honest(tmp_path):
    kg = KnowledgeGraph(tmp_path / "empty.jsonl")
    desc = kg.describe({"family": "trend_breakout"})
    assert "No prior strategies" in desc


def test_feature_vector_for_spec_detects_mechanism_keywords():
    spec = {"source_type": "pinescript", "code_text": "rsiVal = ta.rsi(close, 14)\n// buy the oversold extreme\nlongCondition = rsiVal < 25"}
    fv = feature_vector_for_spec(spec, {"family": "mean_reversion_band"})
    assert fv["uses_rsi"] is True
    assert fv["mean_reversion"] is True
    assert fv["family"] == "mean_reversion_band"


def test_evolution_runner_one_generation_smoke(tmp_path):
    """End-to-end smoke test of the full funnel (generate -> pre-filter ->
    robustness/OOS/MC/prop-sim -> CPCV/PBO -> stress -> cluster -> keep
    top N -> knowledge graph) on synthetic trending data, with every
    threshold loosened and every expensive stage's sample size minimized
    so this runs in seconds, not minutes."""
    df = _trending_df()
    cfg = EvolutionConfig(
        population_size=10, elite_keep=2, max_generations=1,
        min_trades=3, min_profit_factor=0.0, max_drawdown_buffer_mult=20.0,
        mc_sims=50, robustness_neighbors=1, walk_forward_folds=2,
        cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3,
        save_to_library=False, knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        tested_log_path=str(tmp_path / "tested_candidates.jsonl"),
    )
    runner = EvolutionRunner(df, RiskConfig(), PropRules(), cfg, progress_cb=None)
    runner._run_loop()  # calling the loop body directly (no thread) keeps this test deterministic and fast
    # generation is the index of the LAST generation started (0 for a
    # single max_generations=1 run), not a count -- the loop breaks before
    # incrementing past the cap.
    assert runner.generation == 0
    assert not runner.is_running
    assert not runner.resumed
    # Not every synthetic run is guaranteed survivors -- the real assertion
    # is that the whole funnel ran without raising, and produced a journal
    # entry either way.
    assert len(runner.journal) == 1
    assert (tmp_path / "kg.jsonl").exists() or len(runner.leaderboard) == 0
    # A checkpoint should always be written after a completed generation
    # (whether or not anything survived far enough to reach the
    # leaderboard), and the tested-candidates log should have at least
    # one row per population member from PRE-FILTER.
    assert (tmp_path / "checkpoint.json").exists()
    tested_rows = runner.tested_candidates()
    assert len(tested_rows) >= 10


def test_evolution_runner_resumes_from_checkpoint(tmp_path):
    """STOP then START again should continue from the saved generation/
    elites/leaderboard/journal instead of starting a fresh run -- this is
    the fix for the reported "generation 11, leaderboard size 0" bug,
    where every restart silently discarded all prior progress."""
    df = _trending_df()
    checkpoint_path = str(tmp_path / "checkpoint.json")
    tested_log_path = str(tmp_path / "tested_candidates.jsonl")
    kg_path = str(tmp_path / "kg.jsonl")
    base_kwargs = dict(
        population_size=10, elite_keep=2, min_trades=3, min_profit_factor=0.0,
        max_drawdown_buffer_mult=20.0, mc_sims=30, robustness_neighbors=1,
        walk_forward_folds=2, cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3,
        save_to_library=False, knowledge_graph_path=kg_path,
        checkpoint_path=checkpoint_path, tested_log_path=tested_log_path,
        random_seed=7,
    )

    cfg1 = EvolutionConfig(max_generations=1, **base_kwargs)
    runner1 = EvolutionRunner(df, RiskConfig(), PropRules(), cfg1, progress_cb=None)
    assert not runner1.resumed
    runner1._run_loop()
    journal_len_after_run1 = len(runner1.journal)

    cfg2 = EvolutionConfig(max_generations=2, **base_kwargs)
    runner2 = EvolutionRunner(df, RiskConfig(), PropRules(), cfg2, progress_cb=None)
    assert runner2.resumed
    assert runner2.generation == 1  # picks up right after generation 0
    assert len(runner2.journal) == journal_len_after_run1
    runner2._run_loop()
    assert runner2.generation == 1  # ran exactly one more generation (index 1) before hitting max_generations=2
    assert len(runner2.journal) == journal_len_after_run1 + 1


def test_evolution_runner_refuses_to_resume_against_different_data(tmp_path):
    """Resuming a checkpoint built from one dataset against a DIFFERENTLY
    SHAPED dataset must start fresh rather than silently mixing
    incompatible runs."""
    checkpoint_path = str(tmp_path / "checkpoint.json")
    cfg1 = EvolutionConfig(
        population_size=5, elite_keep=2, max_generations=1, min_trades=3,
        min_profit_factor=0.0, mc_sims=20, cpcv_top_n=2, cpcv_max_paths=2,
        save_to_library=False, knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=checkpoint_path, tested_log_path=str(tmp_path / "tested.jsonl"),
    )
    runner1 = EvolutionRunner(_trending_df(n=4000, seed=1), RiskConfig(), PropRules(), cfg1, progress_cb=None)
    runner1._run_loop()

    different_df = _trending_df(n=500, seed=99)
    cfg2 = EvolutionConfig(
        population_size=5, elite_keep=2, max_generations=1,
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=checkpoint_path, tested_log_path=str(tmp_path / "tested.jsonl"),
    )
    runner2 = EvolutionRunner(different_df, RiskConfig(), PropRules(), cfg2, progress_cb=None)
    assert not runner2.resumed
    assert runner2.generation == 0
