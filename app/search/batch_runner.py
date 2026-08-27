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
import os
import shutil
import tempfile
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
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, compute_fitness, run_iterative_refinement
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.reports.generator import generate_full_report
from app.search.results_db import ResultsDB
from app.search.robustness import (
    deflated_sharpe_ratio, parameter_neighborhood_robustness, run_walk_forward,
)
from app.search.strategy_space import SearchSpace
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy

ProgressCallback = Callable[[str], None]


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

    # Stage 3 -- validation gate
    full_mc_sims: int = 3000
    walk_forward_folds: int = 4
    walk_forward_metric: str = "profit_factor"
    walk_forward_min_efficiency: float = 0.4
    robustness_neighbors: int = 6
    robustness_perturbation_frac: float = 0.15
    robustness_min_stability: float = 0.4

    fitness_metric: str = "composite_prop_score"
    workers: int | None = None                # None = os.cpu_count()
    random_seed: int = 42

    def __post_init__(self):
        self.min_trades = max(int(self.min_trades), 1)
        self.min_profit_factor = max(float(self.min_profit_factor), 0.0)
        self.stage1_top_n = max(int(self.stage1_top_n), 1)
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


def _init_worker(df_pickle_path: str, risk_kwargs: dict, prop_kwargs: dict) -> None:
    global _WORKER
    _WORKER["df"] = pd.read_pickle(df_pickle_path)
    _WORKER["risk"] = RiskConfig(**risk_kwargs)
    _WORKER["prop_rules"] = PropRules(**prop_kwargs)


def _stage1_task(candidate_id: str, config: dict, filters: dict) -> dict:
    """Stage 1: one fast backtest, no Monte Carlo. Runs in a worker process."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    try:
        bt = run_backtest(df, ManualStrategy(config), risk)
    except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the whole search
        return {"candidate_id": candidate_id, "config": config, "error": str(exc), "passed_stage1": False}

    stats = bt.statistics.to_dict()
    if not bt.trades:
        return {
            "candidate_id": candidate_id, "config": config, "statistics": stats,
            "error": "no trades generated on this data", "passed_stage1": False,
        }

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
        "candidate_id": candidate_id, "config": config, "statistics": stats,
        "quick_score": quick_score, "sharpe": sharpe, "passed_stage1": bool(passed),
    }


def _stage2_task(
    candidate_id: str, config: dict, refine_kwargs: dict, mc_search_sims: int,
    fitness_metric: str, seed: int,
) -> dict:
    """Stage 2: the app's existing GA refinement, run on one Stage 1 survivor."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    strategy = ManualStrategy(config)
    mc_cfg = MonteCarloConfig(n_simulations=mc_search_sims, random_seed=seed)
    refine_cfg = RefinementConfig(
        fitness_metric=fitness_metric,
        population_size=refine_kwargs["population"],
        generations=refine_kwargs["generations"],
        search_monte_carlo_sims=mc_search_sims,
        random_seed=seed,
    )
    try:
        result = run_iterative_refinement(df, strategy, risk, prop_rules, mc_cfg, refine_cfg, progress_cb=None)
    except RefinementError as exc:
        # No tunable numeric parameters (rare for a generated skeleton;
        # possible for a hand-written single_config in "single" mode) --
        # fall through with a plain backtest so it can still reach Stage 3
        # rather than being silently dropped.
        bt = run_backtest(df, strategy, risk)
        stats = bt.statistics.to_dict()
        if not bt.trades:
            return {"candidate_id": candidate_id, "config": config, "error": str(exc), "passed_stage2": False}
        pnls = [t.pnl for t in bt.trades]
        dates = [t.entry_time for t in bt.trades]
        single_run = simulate_account(pnls, dates, prop_rules)
        mc = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
        prop_summary = summarize_single_run(single_run)
        fitness = compute_fitness(stats, prop_summary, mc, fitness_metric)
        return {
            "candidate_id": candidate_id, "config": config, "statistics": stats,
            "prop_summary": prop_summary, "fitness": fitness,
            "passed_stage2": math.isfinite(fitness), "ga_skipped_reason": str(exc),
        }

    best = result.best
    return {
        "candidate_id": candidate_id,
        "config": best.config if best.config is not None else config,
        "statistics": best.statistics,
        "prop_summary": best.prop_summary,
        "mc_summary": best.mc_summary,
        "fitness": best.fitness,
        "baseline_fitness": result.baseline.fitness,
        "genes_count": len(result.genes),
        "passed_stage2": math.isfinite(best.fitness),
    }


def _stage3_task(candidate_id: str, config: dict, cfg: dict) -> dict:
    """Stage 3: the strict validation gate. Runs in a worker process."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    notes: list[str] = []

    try:
        bt = run_backtest(df, ManualStrategy(config), risk)
    except Exception as exc:  # noqa: BLE001
        return {"candidate_id": candidate_id, "config": config, "error": str(exc), "passed_stage3_gate": False}

    stats = bt.statistics.to_dict()
    if not bt.trades:
        return {
            "candidate_id": candidate_id, "config": config, "statistics": stats,
            "error": "no trades on full dataset", "passed_stage3_gate": False,
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

    lookahead = check_for_lookahead(ManualStrategy(config), df)
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
            df, lambda: ManualStrategy(config), risk,
            n_folds=cfg["walk_forward_folds"], metric=cfg["walk_forward_metric"],
            stability_threshold=cfg["walk_forward_min_efficiency"],
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
            config, df, risk, prop_rules, mc_cfg,
            fitness_metric=cfg["fitness_metric"],
            perturbation_frac=cfg["robustness_perturbation_frac"],
            n_neighbors=cfg["robustness_neighbors"],
            seed=cfg["random_seed"],
            stability_threshold=cfg["robustness_min_stability"],
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
        "candidate_id": candidate_id, "config": config, "statistics": stats,
        "prop_summary": prop_summary, "mc_summary": mc_summary, "fitness": fitness,
        "sharpe": stats.get("sharpe_ratio", 0.0), "cost_ladder": cost_ladder,
        "lookahead": lookahead_dict, "walk_forward": wf_dict, "robustness": robustness_dict,
        "passed_stage3_gate": bool(passed), "gate_notes": "; ".join(notes),
    }


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
) -> SearchSummary:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    run_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    workers = stage_cfg.workers or max(os.cpu_count() or 2, 1)
    workers = max(1, min(workers, len(space.candidates)))

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
            max_workers=workers, initializer=_init_worker, initargs=(str(df_path), risk_kwargs, prop_kwargs),
        ) as pool:
            # ---------------- Stage 1: cheap filter ----------------
            log(f"Stage 1/5: cheap filter across {len(space.candidates)} candidate(s) on {workers} worker(s)...")
            futures = {
                pool.submit(_stage1_task, cid, cfg, filters): cid
                for cid, cfg in space.candidates.items()
            }
            done = 0
            log_every = max(1, len(futures) // 10)
            for fut in as_completed(futures):
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage1_records.append(rec)
                db.insert_candidate(run_id, rec["candidate_id"], "stage1", rec)
                done += 1
                if done % log_every == 0 or done == len(futures):
                    log(f"  Stage 1: {done}/{len(futures)} evaluated...")

            survivors1 = sorted(
                (r for r in stage1_records if r.get("passed_stage1")),
                key=lambda r: r.get("quick_score", 0.0), reverse=True,
            )[: stage_cfg.stage1_top_n]
            log(
                f"Stage 1 complete: {len(survivors1)}/{len(stage1_records)} candidate(s) survived "
                f"the cheap filter and advance to Stage 2 (GA refinement)."
            )
            if not survivors1:
                log(
                    "No candidates survived Stage 1 -- nothing to refine or validate. Consider "
                    "loosening the Stage 1 filters (min trades / min profit factor) or widening the "
                    "search space rather than trusting a forced 'winner' from an empty funnel."
                )
                db.finish_run(run_id, status="no_survivors")
                return SearchSummary(
                    run_id, space.mode, space.family, len(space.candidates), 0, 0, 0,
                    None, time.time() - t0, str(db_path), [],
                )

            # ---------------- Stage 2: GA refinement ----------------
            log(f"Stage 2/5: genetic-algorithm refinement on {len(survivors1)} surviving skeleton(s)...")
            refine_kwargs = {"population": stage_cfg.ga_population, "generations": stage_cfg.ga_generations}
            futures = {
                pool.submit(
                    _stage2_task, r["candidate_id"], r["config"], refine_kwargs,
                    stage_cfg.ga_search_sims, stage_cfg.fitness_metric, stage_cfg.random_seed,
                ): r["candidate_id"]
                for r in survivors1
            }
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage2_records.append(rec)
                db.insert_candidate(run_id, rec["candidate_id"], "stage2", rec)
                done += 1
                log(f"  Stage 2: {done}/{len(survivors1)} skeleton(s) refined...")

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
                "walk_forward_folds": stage_cfg.walk_forward_folds,
                "walk_forward_metric": stage_cfg.walk_forward_metric,
                "walk_forward_min_efficiency": stage_cfg.walk_forward_min_efficiency,
                "robustness_neighbors": stage_cfg.robustness_neighbors,
                "robustness_perturbation_frac": stage_cfg.robustness_perturbation_frac,
                "robustness_min_stability": stage_cfg.robustness_min_stability,
            }
            futures = {
                pool.submit(_stage3_task, r["candidate_id"], r["config"], stage3_cfg): r["candidate_id"]
                for r in survivors2
            }
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage3_records.append(rec)
                done += 1
                log(f"  Stage 3: {done}/{len(survivors2)} candidate(s) validated...")
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
    config = record.get("config")
    if not config:
        raise ValueError(f"Candidate '{candidate_id}' has no stored configuration to promote.")

    strategy = ManualStrategy(config)
    bt_result = run_backtest(df, strategy, risk)
    trade_pnls = [t.pnl for t in bt_result.trades]
    trade_dates = [t.entry_time for t in bt_result.trades]
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    mc_cfg = MonteCarloConfig(n_simulations=mc_sims)
    mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)
    try:
        holdout = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
    except Exception:  # noqa: BLE001 -- a holdout that can't run isn't a reason to block promotion
        holdout = None

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    paths = generate_full_report(
        output_dir=output_dir,
        strategy_name=config.get("name", f"Search Champion {candidate_id}"),
        strategy_source_type="manual",
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
    return {"candidate_id": candidate_id, "config": config, "report_paths": paths}
