"""
Speed Run -- one button: discover a strategy AND validate it, tuned for
minimum wall-clock time to a READY / MARGINAL / NOT READY verdict.

Context: every existing "run everything" path (Full Pipeline, Step 15)
starts from a strategy you already have. Finding a strategy worth
feeding it still means running Search Lab or Evolution Lab separately,
reading its leaderboard, and hand-carrying the winner into Full
Pipeline yourself. When the actual goal is "find *any* strategy that
clears a prop eval, as fast as possible" -- e.g. a hard multi-day
deadline -- that hand-off is exactly the kind of manual step this
module exists to remove.

Speed Run chains, automatically, in one call:

  Phase 1  Wide discovery  -- app.search.batch_runner.run_search across
                              EVERY registered strategy family at once
                              (mode="family", family="all"), with every
                              stage's own settings turned toward speed
                              (smaller GA population/generations, a
                              tighter Stage 1 top-N, fewer Monte Carlo
                              paths at every stage, a per-family Stage 1
                              cap so one family's combinatorial size
                              can't crowd out the others) rather than
                              thoroughness -- this phase's whole job is
                              to throw out everything that clearly
                              doesn't work, as fast as possible, not to
                              fully validate what's left.
  Phase 2  Validate the leaders -- the top `top_k_to_validate` Stage 3
                              survivors (not just the single champion --
                              Stage 3's own gates already caught the
                              obvious dead ends, but only Full
                              Pipeline's walk-forward-aware GA + fresh
                              Monte Carlo + out-of-sample/holdout checks
                              are strict enough to trust for "ready to
                              run live") each get run through
                              app.orchestration.full_pipeline.run_full_pipeline,
                              with ITS OWN settings also turned toward
                              speed (fewer folds/generations/sims than
                              Full Pipeline's own defaults). Validations
                              run concurrently, capped at
                              `max_concurrent_validations` (same
                              worker-budget-splitting approach as
                              app.orchestration.multi_instrument_search,
                              so N concurrent validations don't each
                              independently claim the whole machine's
                              CPU count).
  Phase 3  Pick a winner   -- the best READY verdict wins; if none is
                              READY, the best MARGINAL; only if nothing
                              cleared even MARGINAL does this report
                              "no winner found" rather than picking a
                              NOT READY strategy and calling it done.
                              Ranked by final eval_pass_probability --
                              the same objective every fitness metric
                              default in this app already targets.

Every knob here is a SPEED tradeoff, not a correctness one: nothing in
this module skips a check Full Pipeline or Search Lab would otherwise
run, it only asks each of them to spend less time per candidate. A
strategy this produces is exactly as trustworthy as one built by
running Search Lab then Full Pipeline by hand on the same candidate --
it just does both, across several candidates at once, without a
manual hand-off in between. It does NOT shortcut risk: Full Pipeline's
own with_prop_safety_defaults(), the account-blown circuit breaker,
and every existing lookahead/pip-scale/stop-honesty safeguard still
run exactly as they do everywhere else in this app.

This module's only caller is the Speed Run tab in app.ui.main_window,
which calls run_speed_run() directly on a background thread and blocks
until it returns -- there is no separate progress-polling API to wire
up, and no other entry point (CLI, web) exists for it yet.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.risk import RiskConfig
from app.orchestration.full_pipeline import (
    FullPipelineConfig, FullPipelineResult, run_full_pipeline,
)
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchCancelled, SearchStageConfig, SearchSummary, _spec_from_record, run_search
from app.search.strategy_space import build_strategy_from_spec, generate_search_space

ProgressCallback = Callable[[str], None]


@dataclass
class SpeedRunConfig:
    # -- Phase 1: wide discovery across every family ---------------------
    # Every default below is deliberately smaller/cheaper than
    # SearchStageConfig's own defaults -- see this module's docstring.
    max_candidates: int = 1200
    # Keeps every family in contention rather than letting whichever
    # family happens to have the largest combinatorial grid crowd out
    # the rest of Stage 1 before the expensive stages even see them.
    max_per_family_stage1: int | None = 6
    stage1_top_n: int = 24
    ga_population: int = 8
    ga_generations: int = 3
    ga_search_sims: int = 150
    stage2_top_n: int = 8
    full_mc_sims: int = 1500
    walk_forward_folds: int = 3
    robustness_neighbors: int = 4
    discovery_workers: int | None = None       # None = os.cpu_count()
    discovery_random_seed: int = 42

    # -- Phase 2: validate the discovery phase's own leaders through
    # Full Pipeline, with Full Pipeline's own settings also turned
    # toward speed. --------------------------------------------------
    top_k_to_validate: int = 3
    max_concurrent_validations: int = 2
    validation_ga_population: int = 8
    validation_ga_generations: int = 3
    validation_ga_search_mc_sims: int = 150
    validation_final_mc_sims: int = 3000
    validation_folds: int = 3
    validation_holdout_frac: float = 0.2
    fitness_metric: str = "eval_pass_probability"
    save_winner_to_library: bool = True
    random_seed: int = 42

    def __post_init__(self):
        self.max_candidates = max(int(self.max_candidates), 1)
        self.stage1_top_n = max(int(self.stage1_top_n), 1)
        self.ga_population = max(int(self.ga_population), 4)
        self.ga_generations = max(int(self.ga_generations), 1)
        self.stage2_top_n = max(int(self.stage2_top_n), 1)
        self.full_mc_sims = max(int(self.full_mc_sims), 100)
        self.top_k_to_validate = max(int(self.top_k_to_validate), 1)
        self.max_concurrent_validations = max(int(self.max_concurrent_validations), 1)
        self.validation_ga_population = max(int(self.validation_ga_population), 4)
        self.validation_ga_generations = max(int(self.validation_ga_generations), 1)
        self.validation_final_mc_sims = max(int(self.validation_final_mc_sims), 100)


@dataclass
class SpeedRunCandidateResult:
    candidate_id: str
    family: str | None
    pipeline_result: FullPipelineResult | None
    error: str | None = None


@dataclass
class SpeedRunResult:
    search_summary: SearchSummary
    candidates: list[SpeedRunCandidateResult]
    winner: SpeedRunCandidateResult | None
    winner_reason: str
    elapsed_seconds: float


def _score_key(row: dict) -> float:
    v = row.get("composite_score")
    return v if isinstance(v, (int, float)) and math.isfinite(v) else float("-inf")


def _rank_key(r: SpeedRunCandidateResult) -> tuple[int, float]:
    """Sorts ascending: (tier, -score). Tier 0 = READY, 1 = MARGINAL, 2 =
    anything else (NOT READY or failed) -- so the best USABLE candidate
    always sorts first regardless of raw score, and a high-scoring but
    NOT READY candidate never outranks a lower-scoring READY one."""
    if r.pipeline_result is None:
        return (2, float("inf"))
    tier = {"READY": 0, "MARGINAL": 1}.get(r.pipeline_result.verdict, 2)
    score = r.pipeline_result.final_mc.evaluation_pass_probability
    return (tier, -(score if math.isfinite(score) else float("-inf")))


def run_speed_run(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: SpeedRunConfig | None = None,
    progress_cb: ProgressCallback | None = None,
    instrument: str = "unknown",
    cancel_event: threading.Event | None = None,
) -> SpeedRunResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or SpeedRunConfig()
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Phase 1: wide discovery across every family ---------------------
    log("Phase 1/3: Wide discovery search across every strategy family (speed-tuned settings)...")
    space = generate_search_space(
        mode="family", family="all", max_candidates=cfg.max_candidates,
        seed=cfg.discovery_random_seed,
    )
    stage_cfg = SearchStageConfig(
        stage1_top_n=cfg.stage1_top_n,
        ga_population=cfg.ga_population,
        ga_generations=cfg.ga_generations,
        ga_search_sims=cfg.ga_search_sims,
        stage2_top_n=cfg.stage2_top_n,
        full_mc_sims=cfg.full_mc_sims,
        walk_forward_folds=cfg.walk_forward_folds,
        robustness_neighbors=cfg.robustness_neighbors,
        fitness_metric=cfg.fitness_metric,
        workers=cfg.discovery_workers,
        random_seed=cfg.discovery_random_seed,
        max_per_family_stage1=cfg.max_per_family_stage1,
    )
    db_dir = output_dir / "speed_run_search_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"speed_run_{int(t0)}.db"

    try:
        summary = run_search(
            df, risk, prop_rules, space, stage_cfg, str(db_path),
            instrument=instrument, timeframe="speed-run",
            progress_cb=log, cancel_event=cancel_event,
        )
    except SearchCancelled:
        # run_search (unlike run_speed_run itself) raises rather than
        # returning a cancelled-but-clean result when the cancel event is
        # already set going in -- catch it here so a Speed Run that's
        # stopped before or during discovery reports the same clean
        # "Cancelled" result as one stopped during validation (see the
        # cancel_event check right after Phase 1, below), instead of
        # surfacing as an unhandled exception.
        return SpeedRunResult(
            search_summary=SearchSummary(
                run_id=db_path.stem, mode="family", family="all",
                total_candidates=0, stage1_survivors=0, stage2_survivors=0, stage3_survivors=0,
                champion_candidate_id=None, elapsed_seconds=time.time() - t0,
                db_path=str(db_path), leaderboard=[],
            ),
            candidates=[], winner=None,
            winner_reason="Cancelled during discovery.", elapsed_seconds=time.time() - t0,
        )
    log(
        f"Phase 1 complete in {summary.elapsed_seconds:.1f}s: "
        f"{summary.total_candidates} candidate(s) -> {summary.stage1_survivors} Stage 1 -> "
        f"{summary.stage2_survivors} Stage 2 -> {summary.stage3_survivors} Stage 3 survivor(s)."
    )

    if cancel_event is not None and cancel_event.is_set():
        return SpeedRunResult(
            search_summary=summary, candidates=[], winner=None,
            winner_reason="Cancelled during discovery.", elapsed_seconds=time.time() - t0,
        )

    if not summary.leaderboard:
        log(
            "\nNo Stage 3 survivors -- nothing to validate. Widen Prop Rules / Risk "
            "settings, or try again (each run samples a different random subset of "
            "each family's grid), before assuming no edge exists here."
        )
        return SpeedRunResult(
            search_summary=summary, candidates=[], winner=None,
            winner_reason="No candidate survived Stage 3 of discovery.",
            elapsed_seconds=time.time() - t0,
        )

    # -- Phase 2: validate the top-K leaders through Full Pipeline -------
    top = sorted(summary.leaderboard, key=_score_key, reverse=True)[: cfg.top_k_to_validate]
    log(
        f"\nPhase 2/3: Validating top {len(top)} candidate(s) through Full Pipeline "
        f"(speed-tuned settings, up to {cfg.max_concurrent_validations} at once)..."
    )

    n_concurrent = max(1, min(cfg.max_concurrent_validations, len(top)))
    total_workers = stage_cfg.workers or (os.cpu_count() or 2)
    per_job_workers = max(1, total_workers // n_concurrent)

    fp_cfg = FullPipelineConfig(
        n_folds=cfg.validation_folds,
        ga_population=cfg.validation_ga_population,
        ga_generations=cfg.validation_ga_generations,
        ga_search_mc_sims=cfg.validation_ga_search_mc_sims,
        fitness_metric=cfg.fitness_metric,
        final_mc_sims=cfg.validation_final_mc_sims,
        holdout_frac=cfg.validation_holdout_frac,
        oos_check_folds=cfg.validation_folds,
        random_seed=cfg.random_seed,
        save_to_library=cfg.save_winner_to_library,
        parallel_search=True,
        parallel_search_max_workers=per_job_workers,
    )

    log_lock = threading.Lock()

    def validate_one(row: dict) -> SpeedRunCandidateResult:
        cid = row["candidate_id"]
        family = row.get("family")
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_speedrun_"))
        try:
            spec = _spec_from_record(row)
            strategy = build_strategy_from_spec(spec, tmp_dir)
            with log_lock:
                log(f"  Validating {cid} ({family or 'unknown family'})...")
            result = run_full_pipeline(
                df, strategy, risk, prop_rules, output_dir / "speed_run",
                fp_cfg, progress_cb=None, instrument=instrument,
                report_basename=f"speed_run_{cid}",
            )
            with log_lock:
                log(
                    f"  {cid}: verdict {result.verdict} "
                    f"(eval pass {result.final_mc.evaluation_pass_probability:.1f}%, "
                    f"payout {result.final_mc.first_payout_probability:.1f}%)"
                )
            return SpeedRunCandidateResult(candidate_id=cid, family=family, pipeline_result=result)
        except Exception as exc:  # noqa: BLE001 -- one candidate's failure must not sink the others
            with log_lock:
                log(f"  {cid}: FAILED validation -- {exc}")
            return SpeedRunCandidateResult(candidate_id=cid, family=family, pipeline_result=None, error=str(exc))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    results: list[SpeedRunCandidateResult] = []
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {pool.submit(validate_one, row): row for row in top}
        for fut in as_completed(futures):
            results.append(fut.result())
            if cancel_event is not None and cancel_event.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                break

    # -- Phase 3: pick a winner -------------------------------------------
    log("\nPhase 3/3: Selecting the best validated candidate...")
    ranked_results = sorted(results, key=_rank_key)
    best = ranked_results[0] if ranked_results else None
    if best is not None and best.pipeline_result is not None and best.pipeline_result.verdict in ("READY", "MARGINAL"):
        winner = best
        reason = (
            f"Best validated candidate: {winner.candidate_id} "
            f"({winner.family or 'unknown family'}) -- verdict {winner.pipeline_result.verdict}, "
            f"eval pass {winner.pipeline_result.final_mc.evaluation_pass_probability:.1f}%."
        )
    else:
        winner = None
        reason = "No validated candidate reached even a MARGINAL verdict. See individual results below."
    log(reason)

    elapsed = time.time() - t0
    log(f"\nSpeed Run complete in {elapsed / 60:.1f} minute(s).")
    return SpeedRunResult(
        search_summary=summary, candidates=results, winner=winner,
        winner_reason=reason, elapsed_seconds=elapsed,
    )
