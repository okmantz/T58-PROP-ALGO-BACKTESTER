"""
Search Lab batch runner -- Stages 1 through 5.

    Stage 0 (app.search.strategy_space)  generates the candidate pool.
    Stage 1  Cheap filter        -- one fast backtest per candidate, no Monte
                                     Carlo, run in parallel across CPU cores.
                                     Kills the vast majority of candidates in
                                     minutes, not hours.
    Stage 2  GA refinement       -- the app's EXISTING Iterative Refinement
                                     genetic algorithm (app.optimize.refinement,
                                     completely unmodified), applied to each
                                     Stage 1 survivor in parallel instead of
                                     to one hand-picked strategy at a time.
    Stage 3  Validation gate     -- full-fidelity Monte Carlo, multi-fold
                                     walk-forward (no re-tuning), the generic
                                     lookahead-bias detector, a cost-ladder
                                     stress test, and parameter-neighborhood
                                     robustness. A candidate must clear every
                                     gate to pass -- this is deliberately
                                     strict, because Stage 1/2 alone WILL
                                     surface noise at this scale.
    Stage 4  Leaderboard         -- every candidate at every stage is
                                     persisted to SQLite (app.search.results_db);
                                     Stage 3 survivors are ranked by a
                                     composite score that includes the
                                     Deflated Sharpe Ratio, which corrects
                                     for how many candidates were tried.
    Stage 5  Champion promotion  -- app.search.batch_runner.promote_champion()
                                     re-runs one chosen candidate through the
                                     app's existing, trusted
                                     generate_full_report() pipeline, so it
                                     "graduates" into the exact same report
                                     format every other strategy in this app
                                     produces.

Runs in "single" mode (one user-supplied strategy, re-validated through the
exact same funnel) or "family" mode (a generated grid) -- see
app.search.strategy_space. Both use this one pipeline; single-strategy mode
is not a separate code path.

Multiprocessing note: worker processes load the market data ONCE at pool
startup (via ProcessPoolExecutor's `initializer`) rather than having it
pickled through IPC on every one of potentially thousands of per-candidate
tasks. Workers only ever return small, JSON-safe dicts -- never DataFrames
or Trade objects -- back to the main process, which is the only process
that writes to the results database.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_cost_ladder
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.orchestration.resource_guard import safe_worker_count
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, compute_fitness, run_iterative_refinement
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.reports.generator import generate_full_report
from app.search.failure_triage import aggregate_failure_reasons
from app.search.family_diversity import enforce_family_diversity
from app.search.results_db import ResultsDB
from app.search.robustness import (
    deflated_sharpe_ratio, parameter_neighborhood_robustness, run_walk_forward,
)
from app.search.strategy_space import SearchSpace, build_strategy_from_spec
from app.strategy.lookahead_check import check_for_lookahead

ProgressCallback = Callable[[str], None]


def _record_search_candidates_to_dashboard(
    stage3_records: list[dict], instrument: str, timeframe: str, family: str | None,
) -> None:
    """Every Stage 3 candidate went through a real backtest + full Monte
    Carlo run -- exactly the same kind of result a single Run & Report (or
    Bulk Backtest) run produces -- so it's recorded into run_history the
    same way, purely so Search Lab activity shows up on the Dashboard
    instead of only Bulk Backtest ever doing so. Deliberately scoped to
    Stage 3 only (typically a handful to a few dozen candidates per run,
    never the hundreds/thousands seen at Stage 1) so this can't flood the
    history file. Best-effort: a recording failure here must never affect
    the search result itself."""
    try:
        from app.reports import run_history
    except Exception:
        return
    for rec in stage3_records:
        stats = rec.get("statistics") or {}
        if not stats or not stats.get("total_trades"):
            continue
        try:
            label = f"[Search Lab] {family or rec.get('family') or 'candidate'} {rec['candidate_id'][:8]}"
            report = {
                "strategy": {
                    "name": label, "source_type": rec.get("source_type", "manual"),
                    "instrument": instrument, "timeframe": timeframe,
                },
                "historical_backtest": {"statistics": stats},
                "prop_firm_single_run": rec.get("prop_summary") or {},
                "monte_carlo": rec.get("mc_summary") or {},
            }
            run_history.record_run(report, {"html": ""})
        except Exception:  # noqa: BLE001 -- dashboard visibility is a convenience, never core output
            continue


def _record_fields_from_spec(spec: dict) -> dict:
    """The subset of a candidate spec that gets persisted to / propagated
    through the results DB record for a given stage's output -- separated
    out so every stage task builds this the same way regardless of source
    type, rather than each one hand-rolling which of config/code_text/
    code_extension applies."""
    source_type = spec.get("source_type", "manual")
    if source_type == "manual":
        return {"source_type": "manual", "config": spec.get("config")}
    return {
        "source_type": source_type,
        "code_text": spec.get("code_text"),
        "code_extension": spec.get("code_extension"),
    }


def _spec_from_record(record: dict) -> dict:
    """Reconstructs a candidate spec dict from a stage's output record (or
    a row read back from ResultsDB) -- the inverse of
    _record_fields_from_spec, used to feed one stage's survivors into the
    next stage's task, and to rebuild a strategy for champion promotion."""
    return _record_fields_from_spec(record)


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------

@dataclass
class SearchStageConfig:
    # Stage 1 -- cheap filter
    min_trades: int = 20
    min_profit_factor: float = 1.05
    max_drawdown_buffer_mult: float = 1.5     # candidate's own DD must be <= prop max_drawdown_pct * this
    stage1_top_n: int = 40                    # survivors that advance to Stage 2

    # Stage 2 -- GA refinement (delegates to the existing RefinementConfig)
    ga_population: int = 10
    ga_generations: int = 4
    ga_search_sims: int = 300
    stage2_top_n: int = 10                    # survivors that advance to Stage 3

    # Stage 2 cost-stress: see RefinementConfig.cost_stress_* -- exposed here
    # so a family-wide search can bias its whole GA toward candidates that
    # survive worse execution, not just candidates that look best under the
    # default cost assumptions.
    cost_stress_enabled: bool = True
    cost_stress_multiplier: float = 2.0
    cost_stress_penalty_weight: float = 0.35

    # Stage 3 -- validation gate
    #
    # Hard floor, checked BEFORE the expensive part of Stage 3 (full-fidelity
    # Monte Carlo, the lookahead detector, walk-forward, and parameter-
    # neighborhood robustness all run once each per candidate).
    #
    # FIX (2026-09-04): this used to default to the exact same bar as Stage
    # 1 (min_trades=20, profit_factor>=1.05, net_profit>0 required) and was
    # applied to a PLAIN full-dataset backtest. But Stage 2's GA doesn't
    # optimize for full-dataset raw profit factor -- it optimizes chained
    # out-of-sample fitness (stage_cfg.fitness_metric, "eval_pass_probability"
    # by default: a Monte-Carlo/prop-rules-aware score computed on folded
    # OOS slices). A genome the GA legitimately selected for a strong
    # chained-OOS eval-pass score can easily land under 1.05 profit factor,
    # or even net-negative, on a single plain full-dataset backtest -- those
    # are two different numbers computed two different ways, not a
    # regression. Re-applying Stage 1's exact bar here silently discarded
    # candidates Stage 2 had already found real signal in, before Monte
    # Carlo -- the stage that actually scores the metric being optimized for
    # -- ever got to see them. Loosened to a genuine "protect against
    # wasting a full Monte Carlo run on something catastrophically broken"
    # floor: net profit is no longer required pre-MC (a candidate with a
    # small full-dataset loss but a real out-of-sample edge deserves its
    # Monte Carlo run), and the profit-factor floor is low enough to only
    # catch configs that are actually broken, not merely below Stage 1's bar.
    stage3_min_trades: int = 20
    stage3_min_profit_factor: float = 0.85
    stage3_max_drawdown_buffer_mult: float = 1.5
    stage3_require_positive_net: bool = False

    full_mc_sims: int = 3000
    walk_forward_folds: int = 4
    walk_forward_metric: str = "eval_pass_probability"
    walk_forward_min_efficiency: float = 0.4
    robustness_neighbors: int = 6
    robustness_perturbation_frac: float = 0.15
    robustness_min_stability: float = 0.4

    fitness_metric: str = "eval_pass_probability"
    workers: int | None = None                # None = os.cpu_count()
    random_seed: int = 42

    # Strategy Family Diversity -- caps how many Stage 1 survivors from
    # the SAME classified family (app.strategy.family_taxonomy) can
    # advance into Stage 2, so an "all families" or wide-grid search
    # can't let one family's sheer combinatorial size (e.g. a 10,000-
    # candidate parameter grid for one hypothesis) crowd out every other
    # family before the expensive stages even see them. None (default)
    # disables the cap -- exactly today's behavior, unchanged.
    max_per_family_stage1: int | None = None

    def __post_init__(self):
        self.min_trades = max(int(self.min_trades), 1)
        self.min_profit_factor = max(float(self.min_profit_factor), 0.0)
        self.stage1_top_n = max(int(self.stage1_top_n), 1)
        self.stage3_min_trades = max(int(self.stage3_min_trades), 1)
        self.stage3_min_profit_factor = max(float(self.stage3_min_profit_factor), 0.0)
        self.stage3_max_drawdown_buffer_mult = max(float(self.stage3_max_drawdown_buffer_mult), 0.1)
        self.ga_population = max(int(self.ga_population), 4)
        self.ga_generations = max(int(self.ga_generations), 1)
        self.stage2_top_n = max(int(self.stage2_top_n), 1)
        self.full_mc_sims = max(int(self.full_mc_sims), 100)
        self.walk_forward_folds = max(int(self.walk_forward_folds), 0)
        self.robustness_neighbors = max(int(self.robustness_neighbors), 0)


@dataclass
class SearchSummary:
    run_id: str
    mode: str
    family: str | None
    total_candidates: int
    stage1_survivors: int
    stage2_survivors: int
    stage3_survivors: int
    champion_candidate_id: str | None
    elapsed_seconds: float
    db_path: str
    leaderboard: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker process state & tasks (module-level so ProcessPoolExecutor can
# pickle/import them; state is per-process, populated once by _init_worker)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(df_pickle_path: str, risk_kwargs: dict, prop_kwargs: dict, tmp_dir_path: str) -> None:
    global _WORKER
    _WORKER["df"] = pd.read_pickle(df_pickle_path)
    _WORKER["risk"] = RiskConfig(**risk_kwargs)
    _WORKER["prop_rules"] = PropRules(**prop_kwargs)
    # Shared scratch directory for materializing Python-strategy candidates
    # to disk (PythonStrategy only accepts a file path). Every write uses a
    # uuid4 filename, so concurrent workers sharing this directory is safe.
    _WORKER["tmp_dir"] = Path(tmp_dir_path)


def _passes_stage1_filters(stats: dict, min_trades: int, min_profit_factor: float,
                            max_dd_limit: float, require_positive_net: bool = True) -> bool:
    """The Stage 1 pass/fail test, factored out so it can be re-applied
    against already-computed per-candidate statistics with progressively
    looser thresholds (see run_search's auto-relax step) without
    re-running any backtests."""
    if not stats:
        return False
    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
    net_ok = (stats.get("net_profit", 0.0) or 0.0) > 0 if require_positive_net else True
    return bool(n_trades >= min_trades and pf_val >= min_profit_factor and max_dd <= max_dd_limit and net_ok)


def _stage1_task(candidate_id: str, spec: dict, filters: dict) -> dict:
    """Stage 1: one fast backtest, no Monte Carlo. Runs in a worker process."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
    try:
        strategy = build_strategy_from_spec(spec, tmp_dir)
        bt = run_backtest(df, strategy, risk)
    except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the whole search
        return {**base, "error": str(exc), "passed_stage1": False}

    stats = bt.statistics.to_dict()
    if not bt.trades:
        return {**base, "statistics": stats, "error": "no trades generated on this data", "passed_stage1": False}

    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
    sharpe = stats.get("sharpe_ratio", 0.0) or 0.0

    passed = (
        n_trades >= filters["min_trades"]
        and pf_val >= filters["min_profit_factor"]
        and max_dd <= prop_rules.max_drawdown_pct * filters["max_drawdown_buffer_mult"]
        and stats.get("net_profit", 0.0) > 0
    )
    # Rewards edge (profit factor) AND sample size together, so a 3-trade
    # profit-factor-of-8 fluke doesn't outrank a 200-trade profit-factor-of-1.3
    # strategy that's actually been tested by the data.
    quick_score = pf_val * math.log(n_trades + 1)

    return {
        **base, "statistics": stats,
        "quick_score": quick_score, "sharpe": sharpe, "passed_stage1": bool(passed),
    }


def _stage2_task(
    candidate_id: str, spec: dict, refine_kwargs: dict, mc_search_sims: int,
    fitness_metric: str, seed: int,
) -> dict:
    """Stage 2: the app's existing GA refinement, run on one Stage 1 survivor."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    strategy = build_strategy_from_spec(spec, tmp_dir)
    mc_cfg = MonteCarloConfig(n_simulations=mc_search_sims, random_seed=seed)
    refine_cfg = RefinementConfig(
        fitness_metric=fitness_metric,
        population_size=refine_kwargs["population"],
        generations=refine_kwargs["generations"],
        search_monte_carlo_sims=mc_search_sims,
        random_seed=seed,
        cost_stress_enabled=refine_kwargs.get("cost_stress_enabled", True),
        cost_stress_multiplier=refine_kwargs.get("cost_stress_multiplier", 2.0),
        cost_stress_penalty_weight=refine_kwargs.get("cost_stress_penalty_weight", 0.35),
    )
    try:
        result = run_iterative_refinement(df, strategy, risk, prop_rules, mc_cfg, refine_cfg, progress_cb=None)
    except RefinementError as exc:
        # No tunable numeric parameters (rare for a generated skeleton;
        # possible for a hand-written single_config/strategy in "single"
        # mode) -- fall through with a plain backtest so it can still reach
        # Stage 3 rather than being silently dropped.
        base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
        bt = run_backtest(df, strategy, risk)
        stats = bt.statistics.to_dict()
        if not bt.trades:
            return {**base, "error": str(exc), "passed_stage2": False}
        pnls = [t.pnl for t in bt.trades]
        dates = [t.entry_time for t in bt.trades]
        single_run = simulate_account(pnls, dates, prop_rules)
        mc = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
        prop_summary = summarize_single_run(single_run)
        fitness = compute_fitness(stats, prop_summary, mc, fitness_metric)
        return {
            **base, "statistics": stats,
            "prop_summary": prop_summary, "fitness": fitness,
            "passed_stage2": math.isfinite(fitness), "ga_skipped_reason": str(exc),
        }

    best = result.best
    out_spec_fields = {"source_type": best.source_type}
    if best.source_type == "manual":
        out_spec_fields["config"] = best.config if best.config is not None else spec.get("config")
    else:
        out_spec_fields["code_text"] = best.code_text
        out_spec_fields["code_extension"] = best.code_extension
    return {
        "candidate_id": candidate_id, **out_spec_fields,
        "statistics": best.statistics,
        "prop_summary": best.prop_summary,
        "mc_summary": best.mc_summary,
        "fitness": best.fitness,
        "baseline_fitness": result.baseline.fitness,
        "genes_count": len(result.genes),
        "passed_stage2": math.isfinite(best.fitness),
    }


def _stage3_task(candidate_id: str, spec: dict, cfg: dict) -> dict:
    """Stage 3: the strict validation gate. Runs in a worker process."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
    notes: list[str] = []

    try:
        strategy = build_strategy_from_spec(spec, tmp_dir)
        bt = run_backtest(df, strategy, risk)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": str(exc), "passed_stage3_gate": False}

    stats = bt.statistics.to_dict()
    if not bt.trades:
        return {
            **base, "statistics": stats,
            "error": "no trades on full dataset", "passed_stage3_gate": False,
        }

    # Early-kill floor -- BEFORE the expensive Monte Carlo / lookahead /
    # walk-forward / robustness work below. See SearchStageConfig's
    # stage3_min_trades/stage3_min_profit_factor/stage3_max_drawdown_buffer_mult
    # docstring for why this exists.
    if not _passes_stage1_filters(
        stats,
        min_trades=cfg.get("stage3_min_trades", 20),
        min_profit_factor=cfg.get("stage3_min_profit_factor", 0.85),
        max_dd_limit=prop_rules.max_drawdown_pct * cfg.get("stage3_max_drawdown_buffer_mult", 1.5),
        require_positive_net=cfg.get("stage3_require_positive_net", False),
    ):
        return {
            **base, "statistics": stats,
            "error": (
                "failed Stage 3 early-kill floor (min_trades="
                f"{cfg.get('stage3_min_trades', 20)}, min_profit_factor="
                f"{cfg.get('stage3_min_profit_factor', 0.85):.2f}, "
                f"max_drawdown<={prop_rules.max_drawdown_pct * cfg.get('stage3_max_drawdown_buffer_mult', 1.5):.1f}%)"
                " -- skipped before Monte Carlo/walk-forward/robustness."
            ),
            "passed_stage3_gate": False,
        }

    trade_pnls = [t.pnl for t in bt.trades]
    trade_dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    prop_summary = summarize_single_run(single_run)

    mc_cfg = MonteCarloConfig(n_simulations=cfg["full_mc_sims"], random_seed=cfg["random_seed"])
    mc_result = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
    mc_summary = {
        "evaluation_pass_probability": mc_result.evaluation_pass_probability,
        "first_payout_probability": mc_result.first_payout_probability,
        "expected_payout": mc_result.expected_payout,
        "risk_of_ruin_pct": mc_result.risk_of_ruin_pct,
        "median_drawdown_pct": mc_result.median_drawdown_pct,
        "n_simulations": mc_result.n_simulations,
    }
    fitness = compute_fitness(stats, prop_summary, mc_result, cfg["fitness_metric"])

    cost_ladder = compute_cost_ladder(bt.trades)

    # A fresh strategy instance for the lookahead check, walk-forward, and
    # robustness passes below -- consistent with this app's existing
    # "always build fresh per use" convention for strategy instances (see
    # app.search.robustness.run_walk_forward's own docstring) rather than
    # reusing the one already run above.
    lookahead = check_for_lookahead(build_strategy_from_spec(spec, tmp_dir), df)
    lookahead_dict = {
        "checked": lookahead.checked, "bug_detected": lookahead.bug_detected,
        "skip_reason": lookahead.skip_reason,
    }
    if lookahead.bug_detected:
        notes.append("LOOKAHEAD BUG DETECTED -- excluded regardless of every other score.")

    wf_result = None
    wf_dict = None
    if cfg["walk_forward_folds"] >= 2:
        wf_result = run_walk_forward(
            df, lambda: build_strategy_from_spec(spec, tmp_dir), risk,
            n_folds=cfg["walk_forward_folds"], metric=cfg["walk_forward_metric"],
            stability_threshold=cfg["walk_forward_min_efficiency"],
            prop_rules=prop_rules, mc_cfg=mc_cfg,
        )
        if wf_result is not None:
            wf_dict = {
                "n_folds": wf_result.n_folds, "metric": wf_result.metric,
                "mean_train_metric": wf_result.mean_train_metric,
                "mean_test_metric": wf_result.mean_test_metric,
                "walk_forward_efficiency": wf_result.walk_forward_efficiency,
                "is_stable": wf_result.is_stable,
            }
            if not wf_result.is_stable:
                notes.append(
                    f"Walk-forward efficiency {wf_result.walk_forward_efficiency:.2f} below "
                    f"{cfg['walk_forward_min_efficiency']:.2f} threshold."
                )
        else:
            notes.append("Not enough data to walk-forward test -- treated as unproven, not failed.")

    robustness = None
    robustness_dict = None
    if cfg["robustness_neighbors"] > 0:
        robustness = parameter_neighborhood_robustness(
            spec, df, risk, prop_rules, mc_cfg,
            fitness_metric=cfg["fitness_metric"],
            perturbation_frac=cfg["robustness_perturbation_frac"],
            n_neighbors=cfg["robustness_neighbors"],
            seed=cfg["random_seed"],
            stability_threshold=cfg["robustness_min_stability"],
            tmp_dir=tmp_dir,
        )
        if robustness is not None:
            robustness_dict = {
                "n_neighbors_tested": robustness.n_neighbors_tested,
                "stability_ratio": robustness.stability_ratio,
                "mean_neighbor_fitness": robustness.mean_neighbor_fitness,
                "min_neighbor_fitness": robustness.min_neighbor_fitness,
                "is_stable": robustness.is_stable,
            }
            if not robustness.is_stable:
                notes.append(
                    f"Parameter-neighborhood stability ratio {robustness.stability_ratio:.2f} below "
                    f"{cfg['robustness_min_stability']:.2f} -- may be fit to noise in this window."
                )

    passed = (
        not lookahead.bug_detected
        and (wf_result is None or wf_result.is_stable)
        and (robustness is None or robustness.is_stable)
        and math.isfinite(fitness)
    )

    return {
        **base, "statistics": stats,
        "prop_summary": prop_summary, "mc_summary": mc_summary, "fitness": fitness,
        "sharpe": stats.get("sharpe_ratio", 0.0), "cost_ladder": cost_ladder,
        "lookahead": lookahead_dict, "walk_forward": wf_dict, "robustness": robustness_dict,
        "passed_stage3_gate": bool(passed), "gate_notes": "; ".join(notes),
    }


class SearchCancelled(Exception):
    """Raised when the caller sets ``cancel_event`` mid-run. Not an error --
    the UI catches this to report a clean user-requested stop rather than a
    crash. Any candidates already scored before the cancel point are still
    written to the results DB, so a stopped run isn't a wasted one."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_search(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    space: SearchSpace,
    stage_cfg: SearchStageConfig,
    db_path: str,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> SearchSummary:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def check_cancelled(pool) -> None:
        """Call between/within stage loops. On a stop request, cancels every
        not-yet-started future and drops out of the pool without waiting for
        the whole remaining batch, then raises SearchCancelled so the caller
        can stop cleanly instead of surfacing this as a crash."""
        if cancel_event is not None and cancel_event.is_set():
            log("\nStop requested -- cancelling remaining candidates and shutting down workers...")
            pool.shutdown(wait=False, cancel_futures=True)
            raise SearchCancelled("Search Lab run stopped by user.")

    run_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    workers = stage_cfg.workers or max(os.cpu_count() or 2, 1)
    workers = max(1, min(workers, len(space.candidates)))
    # Each worker process below loads its OWN full copy of `df` (see
    # _init_worker) -- on a large dataset (e.g. years of 1-minute bars),
    # os.cpu_count() workers each holding a full copy can exhaust system
    # memory well before it exhausts CPU, especially if another heavy job
    # (Evolution Lab, Full Pipeline) is ALSO running at the same time. Cap
    # to what's actually safe given this dataset's size and currently
    # available memory rather than trusting the caller's/CPU count blindly.
    # See app.orchestration.resource_guard for the full rationale.
    safe_workers = safe_worker_count(df, requested=workers, max_candidates_in_flight=len(space.candidates))
    if safe_workers < workers:
        log(
            f"Reducing worker processes from {workers} to {safe_workers} -- {len(df):,} bars is "
            f"large enough that {workers} full copies of it (one per worker) would risk exhausting "
            f"available memory. Install 'psutil' for a more precise estimate; for now this uses a "
            f"conservative fallback."
        )
    workers = safe_workers

    db = ResultsDB(db_path)
    db.create_run(run_id, space.mode, space.family, instrument, timeframe, len(space.candidates), asdict(stage_cfg))

    tmp_dir = Path(tempfile.mkdtemp(prefix="t58_search_"))
    df_path = tmp_dir / "data.pkl"
    df.to_pickle(df_path)
    risk_kwargs = asdict(risk)
    prop_kwargs = asdict(prop_rules)

    filters = {
        "min_trades": stage_cfg.min_trades,
        "min_profit_factor": stage_cfg.min_profit_factor,
        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult,
    }

    log(
        f"Search space ready: {len(space.candidates)} candidate(s) "
        f"({space.mode}{', family=' + space.family if space.family else ''}"
        f"{', sampled from ' + str(space.total_generated) if space.sampled else ''})."
    )

    stage1_records: list[dict] = []
    survivors1: list[dict] = []
    stage2_records: list[dict] = []
    survivors2: list[dict] = []
    stage3_records: list[dict] = []

    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(str(df_path), risk_kwargs, prop_kwargs, str(tmp_dir)),
            # Explicit "spawn" rather than the platform default (fork on
            # Linux/macOS): run_search() is routinely called from a
            # background thread rather than a process's main thread -- the
            # desktop GUI's Search Lab tab and the web app's search job
            # both do this so the UI/HTTP response isn't blocked for
            # minutes. forking a multi-threaded process is documented as
            # unsafe (can deadlock if another thread held a lock at fork
            # time) and Python 3.12+ warns about exactly this. spawn avoids
            # the hazard entirely at the cost of slightly slower worker
            # startup, which is negligible next to a Stage 1-3 run.
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            # ---------------- Stage 1: cheap filter ----------------
            log(f"Stage 1/5: cheap filter across {len(space.candidates)} candidate(s) on {workers} worker(s)...")
            futures = {
                pool.submit(_stage1_task, cid, spec, filters): cid
                for cid, spec in space.candidates.items()
            }
            done = 0
            log_every = max(1, len(futures) // 10)
            for fut in as_completed(futures):
                check_cancelled(pool)
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage1_records.append(rec)
                db.insert_candidate(run_id, rec["candidate_id"], "stage1", rec)
                done += 1
                if done % log_every == 0 or done == len(futures):
                    log(f"  Stage 1: {done}/{len(futures)} evaluated...")
            check_cancelled(pool)

            passed_stage1 = [r for r in stage1_records if r.get("passed_stage1")]
            diversity_dropped = 0
            if stage_cfg.max_per_family_stage1:
                passed_stage1, _dropped = enforce_family_diversity(
                    passed_stage1, stage_cfg.max_per_family_stage1, score_key="quick_score",
                )
                diversity_dropped = len(_dropped)
            survivors1 = sorted(
                passed_stage1,
                key=lambda r: r.get("quick_score", 0.0), reverse=True,
            )[: stage_cfg.stage1_top_n]
            log(
                f"Stage 1 complete: {len(survivors1)}/{len(stage1_records)} candidate(s) survived "
                f"the cheap filter and advance to Stage 2 (GA refinement)."
                + (f" ({diversity_dropped} further dropped by the family-diversity cap.)" if diversity_dropped else "")
            )
            stage1_triage = aggregate_failure_reasons(
                stage1_records, "Stage 1", "passed_stage1",
                min_trades=stage_cfg.min_trades, min_profit_factor=stage_cfg.min_profit_factor,
            )
            for line in stage1_triage.format_log_lines():
                log(line)
            if not survivors1:
                # Auto-relax: the original filters found nothing to work
                # with at all, which throws away the whole search rather
                # than giving Stage 2's GA a weaker starting point to try
                # to improve. Re-apply progressively looser thresholds
                # against the SAME already-computed Stage 1 statistics --
                # no re-backtesting -- so a strict default doesn't turn an
                # otherwise-viable search into a dead end. Stage 3's
                # validation gate is unaffected and stays exactly as
                # strict, so a candidate that only got through on relaxed
                # Stage 1 filters still has to earn its way past Stage 3
                # on its own merits.
                relax_rounds = [
                    {
                        "min_trades": max(5, stage_cfg.min_trades // 2),
                        "min_profit_factor": max(0.9, stage_cfg.min_profit_factor * 0.85),
                        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult * 1.5,
                        "require_positive_net": True,
                    },
                    {
                        "min_trades": max(3, stage_cfg.min_trades // 4),
                        "min_profit_factor": 0.7,
                        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult * 2.5,
                        "require_positive_net": False,
                    },
                ]
                relaxed_note = None
                for round_idx, relaxed in enumerate(relax_rounds, start=1):
                    max_dd_limit = prop_rules.max_drawdown_pct * relaxed["max_drawdown_buffer_mult"]
                    candidates_now = [
                        r for r in stage1_records
                        if _passes_stage1_filters(
                            r.get("statistics") or {}, relaxed["min_trades"], relaxed["min_profit_factor"],
                            max_dd_limit, relaxed["require_positive_net"],
                        )
                    ]
                    if candidates_now:
                        if stage_cfg.max_per_family_stage1:
                            candidates_now, _ = enforce_family_diversity(
                                candidates_now, stage_cfg.max_per_family_stage1, score_key="quick_score",
                            )
                        survivors1 = sorted(
                            candidates_now, key=lambda r: r.get("quick_score", 0.0), reverse=True,
                        )[: stage_cfg.stage1_top_n]
                        relaxed_note = (
                            f"Stage 1's original filters (min {stage_cfg.min_trades} trades, "
                            f"profit factor >= {stage_cfg.min_profit_factor:.2f}) found nothing, so it "
                            f"automatically loosened to min {relaxed['min_trades']} trades, profit factor "
                            f">= {relaxed['min_profit_factor']:.2f}"
                            + (", net profit not required" if not relaxed["require_positive_net"] else "")
                            + f" (auto-relax round {round_idx}/{len(relax_rounds)}) and found "
                            f"{len(survivors1)} candidate(s) to refine. These start weaker than the "
                            f"original filters wanted -- Stage 2's GA will try to tune them into "
                            f"something real, and Stage 3's validation gate is unchanged and still strict."
                        )
                        log(f"  Auto-relax round {round_idx}: loosened Stage 1 filters -> "
                            f"{len(survivors1)} candidate(s) now advance.")
                        break
                    log(f"  Auto-relax round {round_idx}: still 0 candidates even with loosened filters.")
                if relaxed_note:
                    log(relaxed_note)
            if not survivors1:
                log(
                    "No candidates survived Stage 1 even after automatically loosening the filters "
                    "twice -- nothing to refine or validate. This means the search space itself "
                    "(the strategy family, or the strategy you provided) doesn't produce a workable "
                    "number of trades on this data at all, not just a strictness setting. Consider "
                    "widening the search space, trying a different family, or checking that the "
                    "market data actually suits the strategy type."
                )
                db.finish_run(run_id, status="no_survivors")
                return SearchSummary(
                    run_id, space.mode, space.family, len(space.candidates), 0, 0, 0,
                    None, time.time() - t0, str(db_path), [],
                )

            # ---------------- Stage 2: GA refinement ----------------
            log(f"Stage 2/5: genetic-algorithm refinement on {len(survivors1)} surviving skeleton(s)...")
            refine_kwargs = {
                "population": stage_cfg.ga_population, "generations": stage_cfg.ga_generations,
                "cost_stress_enabled": stage_cfg.cost_stress_enabled,
                "cost_stress_multiplier": stage_cfg.cost_stress_multiplier,
                "cost_stress_penalty_weight": stage_cfg.cost_stress_penalty_weight,
            }
            futures = {
                pool.submit(
                    _stage2_task, r["candidate_id"], _spec_from_record(r), refine_kwargs,
                    stage_cfg.ga_search_sims, stage_cfg.fitness_metric, stage_cfg.random_seed,
                ): r["candidate_id"]
                for r in survivors1
            }
            done = 0
            for fut in as_completed(futures):
                check_cancelled(pool)
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage2_records.append(rec)
                db.insert_candidate(run_id, rec["candidate_id"], "stage2", rec)
                done += 1
                log(f"  Stage 2: {done}/{len(survivors1)} skeleton(s) refined...")
            check_cancelled(pool)

            survivors2 = sorted(
                (r for r in stage2_records if r.get("passed_stage2") and math.isfinite(r.get("fitness", float("-inf")))),
                key=lambda r: r.get("fitness", float("-inf")), reverse=True,
            )[: stage_cfg.stage2_top_n]
            log(f"Stage 2 complete: {len(survivors2)} candidate(s) advance to the Stage 3 validation gate.")
            if not survivors2:
                log("No candidates survived Stage 2 refinement.")
                db.finish_run(run_id, status="no_survivors")
                return SearchSummary(
                    run_id, space.mode, space.family, len(space.candidates), len(survivors1), 0, 0,
                    None, time.time() - t0, str(db_path), [],
                )

            # ---------------- Stage 3: validation gate ----------------
            log(
                f"Stage 3/5: validation gate (full Monte Carlo, walk-forward, lookahead check, "
                f"cost-ladder stress, parameter-neighborhood robustness) on {len(survivors2)} candidate(s)..."
            )
            stage3_cfg = {
                "full_mc_sims": stage_cfg.full_mc_sims, "random_seed": stage_cfg.random_seed,
                "fitness_metric": stage_cfg.fitness_metric,
                "stage3_min_trades": stage_cfg.stage3_min_trades,
                "stage3_min_profit_factor": stage_cfg.stage3_min_profit_factor,
                "stage3_max_drawdown_buffer_mult": stage_cfg.stage3_max_drawdown_buffer_mult,
                "stage3_require_positive_net": stage_cfg.stage3_require_positive_net,
                "walk_forward_folds": stage_cfg.walk_forward_folds,
                "walk_forward_metric": stage_cfg.walk_forward_metric,
                "walk_forward_min_efficiency": stage_cfg.walk_forward_min_efficiency,
                "robustness_neighbors": stage_cfg.robustness_neighbors,
                "robustness_perturbation_frac": stage_cfg.robustness_perturbation_frac,
                "robustness_min_stability": stage_cfg.robustness_min_stability,
            }
            futures = {
                pool.submit(_stage3_task, r["candidate_id"], _spec_from_record(r), stage3_cfg): r["candidate_id"]
                for r in survivors2
            }
            done = 0
            for fut in as_completed(futures):
                check_cancelled(pool)
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage3_records.append(rec)
                done += 1
                log(f"  Stage 3: {done}/{len(survivors2)} candidate(s) validated...")
            stage3_triage = aggregate_failure_reasons(
                stage3_records, "Stage 3", "passed_stage3_gate",
                min_trades=stage_cfg.stage3_min_trades, min_profit_factor=stage_cfg.stage3_min_profit_factor,
            )
            for line in stage3_triage.format_log_lines():
                log(line)
    except SearchCancelled:
        db.finish_run(run_id, status="cancelled")
        db.close()
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------------- Stage 4: deflate, rank, persist ----------------
    log("Stage 4/5: computing deflated Sharpe ratios and ranking the leaderboard...")
    trial_sharpes = [r.get("sharpe", 0.0) for r in stage1_records if isinstance(r.get("sharpe"), (int, float))]
    n_trials = len(stage1_records)

    for rec in stage3_records:
        sharpe = rec.get("sharpe", 0.0) or 0.0
        n_trade_returns = (rec.get("statistics") or {}).get("total_trades", 0)
        dsr = deflated_sharpe_ratio(sharpe, trial_sharpes, n_trials, n_trade_returns)
        rec["deflated_sharpe"] = {
            "observed_sharpe": dsr.observed_sharpe, "benchmark_sharpe": dsr.benchmark_sharpe,
            "probabilistic_sharpe": dsr.probabilistic_sharpe, "is_significant": dsr.is_significant,
            "n_trials": dsr.n_trials, "note": dsr.note,
        }
        mc = rec.get("mc_summary") or {}
        if rec.get("passed_stage3_gate"):
            rec["composite_score"] = (
                mc.get("evaluation_pass_probability", 0.0) * 0.35
                + mc.get("first_payout_probability", 0.0) * 0.25
                - mc.get("risk_of_ruin_pct", 0.0) * 0.15
                + dsr.probabilistic_sharpe * 100 * 0.25
            )
        else:
            rec["composite_score"] = -1.0
        db.insert_candidate(run_id, rec["candidate_id"], "stage3", rec)

    leaderboard = db.leaderboard(run_id, stage="stage3", top_n=25, only_passed=False)
    champion = next((r for r in leaderboard if r.get("passed_stage3_gate")), None)
    db.finish_run(run_id, status="completed")
    db.close()

    _record_search_candidates_to_dashboard(stage3_records, instrument, timeframe, space.family)

    elapsed = time.time() - t0
    n_passed = sum(1 for r in stage3_records if r.get("passed_stage3_gate"))
    log(
        f"Stage 5/5: search complete in {elapsed:.1f}s. "
        f"{n_passed}/{len(stage3_records)} candidate(s) passed every Stage 3 gate."
    )
    if champion:
        log(
            f"Champion candidate: {champion['candidate_id']} "
            f"(composite score {champion.get('composite_score', 0):.2f}, "
            f"PSR {champion.get('deflated_sharpe', {}).get('probabilistic_sharpe', 0):.2f})."
        )
    else:
        log(
            "No candidate passed every Stage 3 gate. That is an honest, useful result -- "
            "'nothing in this search beats the deflated chance benchmark' is a real finding, "
            "not a failed run. See the leaderboard for the closest calls and why each one failed."
        )

    return SearchSummary(
        run_id=run_id, mode=space.mode, family=space.family, total_candidates=len(space.candidates),
        stage1_survivors=len(survivors1), stage2_survivors=len(survivors2), stage3_survivors=len(stage3_records),
        champion_candidate_id=champion["candidate_id"] if champion else None,
        elapsed_seconds=elapsed, db_path=str(db_path), leaderboard=leaderboard,
    )


# ---------------------------------------------------------------------------
# Stage 5: champion promotion
# ---------------------------------------------------------------------------

def promote_champion(
    db_path: str, run_id: str, candidate_id: str, df: pd.DataFrame,
    risk: RiskConfig, prop_rules: PropRules, output_dir: str, mc_sims: int = 10000,
) -> dict:
    """
    Re-runs one chosen Stage 3 survivor through the app's EXISTING,
    trusted single-strategy report pipeline (the identical
    generate_full_report() call the normal Run & Report tab uses), plus a
    fresh full-dataset holdout check -- so the champion graduates into the
    exact same report format every other strategy in this app already
    produces, rather than a search-specific artifact nobody's used to
    reading yet.
    """
    with ResultsDB(db_path) as db:
        record = db.get_candidate(candidate_id, run_id=run_id, stage="stage3")
    if record is None:
        raise ValueError(f"Candidate '{candidate_id}' not found in run '{run_id}' at stage3.")

    spec = _spec_from_record(record)
    source_type = spec.get("source_type", "manual")
    if source_type == "manual" and not spec.get("config"):
        raise ValueError(f"Candidate '{candidate_id}' has no stored configuration to promote.")
    if source_type != "manual" and not spec.get("code_text"):
        raise ValueError(f"Candidate '{candidate_id}' has no stored source code to promote.")

    promote_tmp_dir = Path(tempfile.mkdtemp(prefix="t58_promote_"))
    try:
        strategy = build_strategy_from_spec(spec, promote_tmp_dir)
        bt_result = run_backtest(df, strategy, risk)
        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
        mc_cfg = MonteCarloConfig(n_simulations=mc_sims)
        mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)
        try:
            holdout = run_holdout_comparison(
                df, build_strategy_from_spec(spec, promote_tmp_dir), risk, holdout_frac=0.2,
            )
        except Exception:  # noqa: BLE001 -- a holdout that can't run isn't a reason to block promotion
            holdout = None

        if source_type == "manual":
            strategy_name = spec["config"].get("name", f"Search Champion {candidate_id}")
        else:
            strategy_name = f"Search Champion {candidate_id} ({source_type})"

        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_full_report(
            output_dir=output_dir,
            strategy_name=strategy_name,
            strategy_source_type=source_type,
            instrument="search-lab",
            timeframe="unknown",
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=prop_rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            holdout_comparison=holdout,
            risk_config=risk,
            price_df=df,
        )
        return {
            "candidate_id": candidate_id, "spec": spec, "config": spec.get("config"),
            "report_paths": paths,
        }
    finally:
        shutil.rmtree(promote_tmp_dir, ignore_errors=True)
