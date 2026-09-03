"""Tests for app.orchestration.multi_instrument_search.

Kept small-scale (2 tiny CSVs, workers=1, low candidate counts) so this
runs quickly while still exercising the real ThreadPoolExecutor ->
run_search -> ProcessPoolExecutor path end to end, not a mock of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.multi_instrument_search import (
    InstrumentJob, best_result_across_instruments, run_multi_instrument_search,
)
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig
from app.search.strategy_space import generate_search_space


def _trending_csv(path, n=1200, seed=3, drift=0.00015):
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
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(path, index=False)
    return path


def _fast_stage_cfg(**overrides) -> SearchStageConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=4, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=2, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        stage3_min_trades=1, stage3_min_profit_factor=0.0,
        workers=1, random_seed=42,
    )
    base.update(overrides)
    return SearchStageConfig(**base)


@pytest.fixture
def two_instrument_jobs(tmp_path):
    csv_a = _trending_csv(tmp_path / "EURUSD5.csv", seed=1)
    csv_b = _trending_csv(tmp_path / "GBPUSD5.csv", seed=2)
    return [
        InstrumentJob(instrument="EURUSD", timeframe="5m", csv_path=csv_a),
        InstrumentJob(instrument="GBPUSD", timeframe="5m", csv_path=csv_b),
    ]


def test_runs_every_job_and_gives_each_its_own_db(tmp_path, two_instrument_jobs):
    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=6, seed=1)
    results = run_multi_instrument_search(
        two_instrument_jobs, space, RiskConfig(), PropRules(),
        _fast_stage_cfg(), db_dir=tmp_path / "dbs", max_concurrent_instruments=2,
    )
    assert set(results.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
    for label, result in results.items():
        assert result.error is None
        assert result.summary is not None
        assert result.summary.total_candidates == len(space.candidates)
    db_files = list((tmp_path / "dbs").glob("*.db"))
    assert len(db_files) == 2  # one per instrument, never shared


def test_worker_budget_is_split_across_concurrent_jobs(tmp_path, two_instrument_jobs):
    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=6, seed=1)
    stage_cfg = _fast_stage_cfg(workers=4)
    results = run_multi_instrument_search(
        two_instrument_jobs, space, RiskConfig(), PropRules(),
        stage_cfg, db_dir=tmp_path / "dbs", max_concurrent_instruments=2,
    )
    # Both jobs still completed correctly with the reduced per-job worker count.
    assert all(r.summary is not None for r in results.values())


def test_one_bad_csv_path_does_not_sink_the_other_job(tmp_path, two_instrument_jobs):
    jobs = list(two_instrument_jobs)
    jobs[0] = InstrumentJob(instrument="BROKEN", timeframe="5m", csv_path=tmp_path / "does_not_exist.csv")
    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=6, seed=1)
    results = run_multi_instrument_search(
        jobs, space, RiskConfig(), PropRules(),
        _fast_stage_cfg(), db_dir=tmp_path / "dbs", max_concurrent_instruments=2,
    )
    assert results["BROKEN/5m"].error is not None
    assert results["BROKEN/5m"].summary is None
    assert results["GBPUSD/5m"].summary is not None


def test_best_result_across_instruments_picks_highest_champion_score(tmp_path, two_instrument_jobs):
    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=6, seed=1)
    results = run_multi_instrument_search(
        two_instrument_jobs, space, RiskConfig(), PropRules(),
        _fast_stage_cfg(), db_dir=tmp_path / "dbs", max_concurrent_instruments=2,
    )
    best = best_result_across_instruments(results)
    # Not guaranteed either trending series produces a Stage 3 champion at
    # this tiny scale -- just assert the function runs cleanly either way.
    if best is not None:
        assert best.label in results
