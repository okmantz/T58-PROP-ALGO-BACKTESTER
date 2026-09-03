"""
Multi-instrument / multi-timeframe Search Lab orchestration.

Search Lab and Evolution Lab both validate against ONE market data file at
a time (Owen's primary XAUUSD15 feed). Since a real edge is instrument-
and timeframe-dependent -- a hypothesis that's dead on XAUUSD15 might be
alive on EURUSD5 or GC1!60m -- running the exact same family/search space
across several instruments and timeframes CONCURRENTLY multiplies the
chance of finding a validated edge per unit wall-clock time, instead of
only ever finding out "gold at 15 minutes" one search at a time.

This module is a thin orchestration layer on top of the EXISTING, already-
tested app.search.batch_runner.run_search -- it does not reimplement any
search logic. It only:

  1. Loads each instrument/timeframe's CSV once.
  2. Divides `stage_cfg.workers` across the number of CONCURRENT instrument
     runs (each run_search call still gets its own ProcessPoolExecutor
     internally), so running 3 instruments at once doesn't oversubscribe
     the machine's CPU count 3x over.
  3. Gives each instrument/timeframe its OWN results database file rather
     than sharing one -- app.search.results_db.ResultsDB is explicitly not
     built for concurrent writers from separate processes/connections (see
     its own docstring), and the simplest way to honor that is to never
     ask it to be one, rather than adding locking/retry logic to a module
     that was deliberately kept simple. Each returned SearchSummary
     carries its own db_path so every instrument's leaderboard is still
     independently queryable.
  4. Runs jobs on a ThreadPoolExecutor (not another process pool) capped
     at `max_concurrent_instruments` -- each thread's run_search call
     still spawns its OWN process pool for its own Stage 1-3 work, so the
     real parallelism ends up as (concurrent instruments) x (workers per
     instrument), bounded to stay within one shared CPU budget.

A single `cancel_event` is honored across every concurrent job, so
stopping a multi-instrument run stops all of them together, not one at a
time.
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, SearchSummary, run_search
from app.search.strategy_space import SearchSpace

ProgressCallback = Callable[[str, str], None]   # (job_label, message)


@dataclass(frozen=True)
class InstrumentJob:
    """One (instrument, timeframe) target: a market-data file to search
    the SAME strategy space against. `instrument`/`timeframe` are just
    labels used for logging, the results DB, and the returned summary
    dict's keys -- they don't have to match the CSV's own folder/file
    naming convention."""
    instrument: str
    timeframe: str
    csv_path: str | Path


@dataclass
class MultiInstrumentResult:
    job: InstrumentJob
    summary: SearchSummary | None
    error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.job.instrument}/{self.job.timeframe}"


def _resolved_workers_per_job(stage_cfg: SearchStageConfig, n_concurrent: int) -> int | None:
    """Splits the configured worker budget across however many instrument
    jobs will actually run at once, so N concurrent instruments don't each
    independently claim os.cpu_count() workers. Never goes below 1."""
    total = stage_cfg.workers or (os.cpu_count() or 2)
    return max(1, total // max(1, n_concurrent))


def run_multi_instrument_search(
    jobs: list[InstrumentJob],
    space: SearchSpace,
    risk: RiskConfig,
    prop_rules: PropRules,
    stage_cfg: SearchStageConfig,
    db_dir: str | Path,
    max_concurrent_instruments: int = 2,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, MultiInstrumentResult]:
    """Runs `space` (the SAME strategy family / grid / single-strategy
    space -- it's just candidate specs, not tied to any one dataset)
    against every job's own market data, up to `max_concurrent_instruments`
    at once. Returns {\"INSTRUMENT/TIMEFRAME\": MultiInstrumentResult}.

    `db_dir`: a directory (created if missing) -- each job gets its own
    `<db_dir>/<instrument>_<timeframe>.db` results database (see this
    module's own docstring for why they're never shared).
    """
    if not jobs:
        raise ValueError("run_multi_instrument_search requires at least one InstrumentJob.")

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    n_concurrent = max(1, min(int(max_concurrent_instruments), len(jobs)))
    per_job_workers = _resolved_workers_per_job(stage_cfg, n_concurrent)
    per_job_stage_cfg = replace(stage_cfg, workers=per_job_workers)

    def log(job: InstrumentJob, msg: str) -> None:
        if progress_cb:
            progress_cb(f"{job.instrument}/{job.timeframe}", msg)

    def run_one(job: InstrumentJob) -> MultiInstrumentResult:
        try:
            log(job, f"Loading {job.csv_path}...")
            import_result = import_csv(job.csv_path)
            if not import_result.is_valid:
                raise ValueError(
                    f"Could not import {job.csv_path}: " + "; ".join(import_result.errors)
                )
            df = import_result.dataframe
            db_path = db_dir / f"{job.instrument}_{job.timeframe}.db"
            log(job, f"Starting search ({per_job_workers} worker(s))...")
            summary = run_search(
                df, risk, prop_rules, space, per_job_stage_cfg, str(db_path),
                instrument=job.instrument, timeframe=job.timeframe,
                progress_cb=lambda m: log(job, m),
                cancel_event=cancel_event,
            )
            log(job, f"Complete: {summary.stage3_survivors} candidate(s) passed Stage 3.")
            return MultiInstrumentResult(job=job, summary=summary)
        except Exception as exc:  # noqa: BLE001 -- one instrument's failure must not sink the others
            log(job, f"FAILED: {exc}")
            return MultiInstrumentResult(
                job=job, summary=None,
                error=f"{exc}\n{traceback.format_exc()}",
            )

    results: dict[str, MultiInstrumentResult] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results[res.label] = res

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r.summary is not None)
    if progress_cb:
        progress_cb(
            "multi-instrument",
            f"All {len(jobs)} instrument/timeframe job(s) complete in {elapsed:.1f}s "
            f"({n_ok} succeeded, {len(jobs) - n_ok} failed).",
        )
    return results


def best_result_across_instruments(results: dict[str, MultiInstrumentResult]) -> MultiInstrumentResult | None:
    """Convenience: which instrument/timeframe actually produced a Stage 3
    champion, ranked by that champion's own composite_score. Returns None
    if nothing anywhere passed Stage 3."""
    candidates = [
        r for r in results.values()
        if r.summary is not None and r.summary.champion_candidate_id and r.summary.leaderboard
    ]
    if not candidates:
        return None

    def champion_score(r: MultiInstrumentResult) -> float:
        for rec in r.summary.leaderboard:
            if rec.get("candidate_id") == r.summary.champion_candidate_id:
                score = rec.get("composite_score")
                return score if isinstance(score, (int, float)) and math.isfinite(score) else float("-inf")
        return float("-inf")

    return max(candidates, key=champion_score)
