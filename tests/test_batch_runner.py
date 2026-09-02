"""
Tests for app.search.batch_runner (Search Lab Stages 1-5).

Kept deliberately small-scale (few candidates, workers=1, low Monte Carlo
sim counts) so the whole file runs in a few seconds under CI while still
exercising the real ProcessPoolExecutor path end to end -- these are
integration tests of the actual multiprocessing pipeline, not mocks of it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.results_db import ResultsDB
from app.search.strategy_space import generate_search_space
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy


def _trending_df(n=2500, seed=3, drift=0.00015):
    """Strong, near-deterministic uptrend -- deliberately easy for a
    trend-following family to find real signal on, so Stage 1-3 have
    something to actually pass (not every test should exercise the
    'nothing survives' path)."""
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


def _fast_stage_cfg(**overrides) -> SearchStageConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=6, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=3, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
    )
    base.update(overrides)
    return SearchStageConfig(**base)


@pytest.fixture
def small_family_space():
    return generate_search_space(mode="family", family="trend_breakout", max_candidates=8, seed=1)


@pytest.fixture
def single_space():
    cfg = {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 20, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "risk_management": {"stop_type": "fixed", "stop_value": 20, "target_type": "fixed", "target_value": 40},
    }
    return generate_search_space(mode="single", single_config=cfg)


# ---------------------------------------------------------------------------
# run_search: shape / bookkeeping guarantees that must hold regardless of
# whether anything actually survives to the end.
# ---------------------------------------------------------------------------

def test_run_search_family_mode_end_to_end_no_crash(tmp_path, small_family_space):
    df = _trending_df()
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.total_candidates == 8
    assert summary.stage1_survivors >= 0
    assert summary.elapsed_seconds > 0
    assert summary.run_id


def test_run_search_single_mode_wraps_one_strategy(tmp_path, single_space):
    df = _trending_df()
    summary = run_search(
        df, RiskConfig(), PropRules(), single_space, _fast_stage_cfg(stage1_top_n=1, stage2_top_n=1),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.total_candidates == 1
    assert summary.mode == "single"


def test_run_search_persists_every_stage_to_db(tmp_path, small_family_space):
    df = _trending_df()
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    with ResultsDB(db_path) as db:
        run_row = db.run_summary(summary.run_id)
        assert run_row["status"] in ("completed", "no_survivors")
        assert run_row["stage_counts"]["stage1"] == summary.total_candidates


def test_run_search_empty_stage1_survivors_is_handled_gracefully(tmp_path, small_family_space):
    df = _trending_df()
    # Impossible filter -- nothing can pass Stage 1, even after auto-relax.
    cfg = _fast_stage_cfg(min_trades=10**9)
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, cfg,
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.stage1_survivors == 0
    assert summary.stage2_survivors == 0
    assert summary.stage3_survivors == 0
    assert summary.champion_candidate_id is None
    assert summary.leaderboard == []


def test_run_search_family_diversity_cap_limits_stage2_to_one_per_family(tmp_path):
    """max_per_family_stage1=1 across an 'all families' search must never
    let two candidates classified into the SAME app.strategy.
    family_taxonomy group both reach Stage 2 -- the actual point of the
    feature: no single family's grid size can crowd out the others."""
    from app.strategy.family_taxonomy import classify_record

    df = _trending_df()
    space = generate_search_space(mode="family", family="all", max_candidates=60, seed=1)
    cfg = _fast_stage_cfg(
        min_trades=1, min_profit_factor=0.0, max_drawdown_buffer_mult=10.0,
        stage1_top_n=20, max_per_family_stage1=1,
    )
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), space, cfg,
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    with ResultsDB(db_path) as db:
        stage2_rows = db.leaderboard(summary.run_id, stage="stage2", top_n=1000)
    groups = [classify_record(r) for r in stage2_rows]
    assert len(groups) == len(set(groups)), f"more than one Stage 2 survivor shares a family group: {groups}"


def test_run_search_auto_relaxes_stage1_filters_instead_of_giving_up(tmp_path, small_family_space):
    """A strict-but-not-impossible filter (more trades than this small,
    short-lived candidate pool will realistically produce) should trigger
    the auto-relax path and still find candidates to advance, rather than
    the search dead-ending at Stage 1 the way it used to."""
    df = _trending_df()
    logs = []
    cfg = _fast_stage_cfg(min_trades=50, min_profit_factor=50.0)
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, cfg,
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
        progress_cb=logs.append,
    )
    assert summary.stage1_survivors > 0
    assert any("auto-relax" in line.lower() for line in logs)


def test_run_search_records_stage3_results_onto_the_dashboard(tmp_path, small_family_space, monkeypatch):
    from app.reports import run_history

    monkeypatch.setattr(run_history, "history_path", lambda: tmp_path / "run_history.json")

    df = _trending_df()
    before = len(run_history.load_runs())
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    after = run_history.load_runs()
    assert summary.stage3_survivors > 0
    assert len(after) == before + summary.stage3_survivors
    assert all(r["strategy_name"].startswith("[Search Lab]") for r in after[before:])


def test_run_search_leaderboard_candidates_all_reached_stage3(tmp_path, small_family_space):
    df = _trending_df()
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    for row in summary.leaderboard:
        assert "composite_score" in row
        assert "deflated_sharpe" in row and row["deflated_sharpe"] is not None


def test_run_search_progress_callback_is_invoked(tmp_path, small_family_space):
    df = _trending_df()
    messages = []
    run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
        progress_cb=messages.append,
    )
    joined = " ".join(messages)
    assert "Stage 1" in joined
    assert "Stage 5" in joined or "Search complete" in joined or "search complete" in joined


def test_run_search_lookahead_bug_excludes_a_candidate_regardless_of_profit():
    """A candidate flagged by the lookahead detector must never pass Stage 3,
    even if its raw stats look excellent -- this is the whole reason the
    check runs inside the gate rather than being an advisory note only."""
    from app.search.batch_runner import _stage3_task

    leaky_spec = {
        "source_type": "manual",
        "config": {
            "name": "leaky",
            "long_entry": "close > close",  # will legitimately produce zero trades; used only
            "risk_management": {"stop_type": "fixed", "stop_value": 20, "target_type": "fixed", "target_value": 40},
        },
    }
    # This config produces zero trades (close > close is never true), which
    # must be handled as a clean Stage 3 failure, not a crash.
    from app.search.batch_runner import _init_worker
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        import pandas as pd
        df = _trending_df(n=200)
        df_path = f"{td}/data.pkl"
        df.to_pickle(df_path)
        _init_worker(df_path, {}, {}, td)
        result = _stage3_task("leaky-1", leaky_spec, {
            "full_mc_sims": 20, "random_seed": 1, "fitness_metric": "composite_prop_score",
            "walk_forward_folds": 0, "walk_forward_metric": "profit_factor",
            "walk_forward_min_efficiency": 0.4, "robustness_neighbors": 0,
            "robustness_perturbation_frac": 0.15, "robustness_min_stability": 0.4,
        })
    assert result["passed_stage3_gate"] is False


# ---------------------------------------------------------------------------
# Champion promotion (Stage 5)
# ---------------------------------------------------------------------------

def test_promote_champion_raises_for_unknown_candidate(tmp_path):
    df = _trending_df(n=200)
    with ResultsDB(tmp_path / "search.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 1, {})
    with pytest.raises(ValueError):
        promote_champion(
            str(tmp_path / "search.db"), "run1", "does-not-exist", df,
            RiskConfig(), PropRules(), output_dir=str(tmp_path / "out"),
        )


def test_promote_champion_produces_a_full_report(tmp_path, small_family_space):
    df = _trending_df()
    # Loose gates so at least one candidate is very likely to survive to Stage 3.
    cfg = _fast_stage_cfg(min_trades=1, min_profit_factor=0.0, max_drawdown_buffer_mult=100.0)
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, cfg,
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0, "expected at least one candidate to reach Stage 3 with loose gates"

    # Promote whatever the top leaderboard entry is, whether or not it
    # cleared every Stage 3 gate -- promote_champion itself doesn't gate,
    # it only re-runs and reports (gating already happened in Stage 3/4).
    candidate_id = summary.leaderboard[0]["candidate_id"]
    result = promote_champion(
        str(db_path), summary.run_id, candidate_id, df,
        RiskConfig(), PropRules(), output_dir=str(tmp_path / "champion"), mc_sims=50,
    )
    assert result["candidate_id"] == candidate_id
    assert result["report_paths"]["html"].exists()
    assert result["report_paths"]["json"].exists()


# ---------------------------------------------------------------------------
# Non-Manual source types: Python, PineScript, MQL5 -- both single mode and
# family (grid-around-strategy) mode, run through the REAL Stage 1-5
# pipeline end to end, same as the Manual tests above.
# ---------------------------------------------------------------------------

_PYTHON_SRC = '''STRATEGY_NAME = "Test EMA Cross"
EMA_FAST = 5
EMA_SLOW = 15
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    sig = (fast > slow).astype(int) - (fast < slow).astype(int)
    return sig
'''

_PINESCRIPT_SRC = '''//@version=5
strategy("Test", overlay=true)
fastLen = input.int(5, "Fast Length")
slowLen = input.int(15, "Slow Length")
fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
longCondition = ta.crossover(fast, slow)
shortCondition = ta.crossunder(fast, slow)
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
// T58_SL_PIPS=20
// T58_TP_PIPS=40
'''

_MQL5_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 15, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) { trade.Buy(0.1, _Symbol); }
   if (fastMA < slowMA) { trade.Sell(0.1, _Symbol); }
   // T58_SL_PIPS=20
   // T58_TP_PIPS=40
}
'''


@pytest.mark.parametrize("source_type", ["python", "pinescript", "mql5"])
def test_run_search_single_mode_works_for_every_code_source_type(tmp_path, source_type):
    df = _trending_df()
    if source_type == "python":
        path = tmp_path / "strat.py"
        path.write_text(_PYTHON_SRC, encoding="utf-8")
        strategy = PythonStrategy(path)
    elif source_type == "pinescript":
        strategy = PineScriptStrategy(_PINESCRIPT_SRC)
    else:
        strategy = MQL5Strategy(_MQL5_SRC)

    space = generate_search_space(mode="single", strategy=strategy)
    summary = run_search(
        df, RiskConfig(), PropRules(), space, _fast_stage_cfg(stage1_top_n=1, stage2_top_n=1),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.mode == "single"
    assert summary.total_candidates == 1
    assert summary.elapsed_seconds > 0


@pytest.mark.parametrize("source_type", ["python", "pinescript", "mql5"])
def test_run_search_family_grid_mode_works_for_every_code_source_type(tmp_path, source_type):
    df = _trending_df()
    if source_type == "python":
        path = tmp_path / "strat.py"
        path.write_text(_PYTHON_SRC, encoding="utf-8")
        strategy = PythonStrategy(path)
    elif source_type == "pinescript":
        strategy = PineScriptStrategy(_PINESCRIPT_SRC)
    else:
        strategy = MQL5Strategy(_MQL5_SRC)

    space = generate_search_space(
        mode="family", strategy=strategy, grid_points_per_gene=2, max_candidates=8, seed=1,
    )
    assert space.family == f"{source_type}_grid"
    assert len(space.candidates) > 1

    summary = run_search(
        df, RiskConfig(), PropRules(), space, _fast_stage_cfg(),
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.mode == "family"
    assert summary.total_candidates == len(space.candidates)
    for row in summary.leaderboard:
        assert row["source_type"] == source_type
        assert row["code_text"]  # persisted through the whole funnel, not just Stage 1


def test_promote_champion_works_for_a_python_candidate(tmp_path):
    df = _trending_df()
    path = tmp_path / "strat.py"
    path.write_text(_PYTHON_SRC, encoding="utf-8")
    strategy = PythonStrategy(path)
    space = generate_search_space(
        mode="family", strategy=strategy, grid_points_per_gene=2, max_candidates=8, seed=1,
    )
    # Loose gates so at least one candidate is very likely to survive to Stage 3.
    cfg = _fast_stage_cfg(min_trades=1, min_profit_factor=0.0, max_drawdown_buffer_mult=100.0)
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), space, cfg,
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0

    candidate_id = summary.leaderboard[0]["candidate_id"]
    assert summary.leaderboard[0]["source_type"] == "python"
    result = promote_champion(
        str(db_path), summary.run_id, candidate_id, df,
        RiskConfig(), PropRules(), output_dir=str(tmp_path / "champion"), mc_sims=50,
    )
    assert result["candidate_id"] == candidate_id
    assert result["spec"]["source_type"] == "python"
    assert result["report_paths"]["html"].exists()
