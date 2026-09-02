"""
Evolution Lab -- natural-selection-based strategy discovery.

    RESEARCH (knowledge graph, informs which families/features get
              weighted into GENERATE)
        v
    GENERATE ~N STRATEGIES   (app.search.strategy_space, all families)
        v
    PRE-FILTER               (one cheap backtest: trades/profit-factor/DD)
        v
    BACKTEST                 (full dataset, already done above)
        v
    ROBUSTNESS FILTER        (app.search.robustness.parameter_neighborhood_robustness)
        v
    OOS FILTER                (app.search.robustness.run_walk_forward)
        v
    MONTE CARLO               (app.monte_carlo.engine.run_monte_carlo)
        v
    PROP SIMULATION            (app.prop.simulator.simulate_account)
        v
    CPCV / PBO                (app.validation.cpcv -- REAL combinatorial
                                purged CV + genuine multi-candidate PBO,
                                applied to the best `cpcv_top_n` survivors
                                only -- it's the most expensive stage)
        v
    STRESS TEST                (re-run at N-x execution costs)
        v
    CLUSTER                    (correlation-dedupe on daily P&L, so the
                                 top 10 aren't 10 near-identical variants
                                 of the same winner)
        v
    KEEP TOP N -> record to knowledge graph -> MUTATE -> repeat

This reuses this app's existing, already-tested building blocks end to
end (Search Lab's strategy generator, the walk-forward GA's mutation
operators, robustness/walk-forward/CPCV/PBO, Monte Carlo, the prop
simulator) rather than reimplementing any of them -- see the imports
below for exactly which module each stage delegates to. The new code
here is the composite PROP FITNESS scoring (app.evolution.prop_fitness),
the knowledge graph (app.evolution.knowledge_graph), and the generation
loop itself (this module).

Known scope limits (stated plainly rather than glossed over):
- Candidates are Manual Strategy Builder configs generated from
  app.search.strategy_space's families -- this does not mutate uploaded
  Python/PineScript/MQL5 files. Manual configs are what Search Lab
  already generates and mutates today, so this is the same scope as the
  system Owen asked to have "combined," not a new restriction.
- PRE-FILTER and the ROBUSTNESS/OOS/MONTE CARLO/PROP SIMULATION stage
  (by far the two most expensive, once-per-candidate stages) now run
  across a ProcessPoolExecutor pool, the same pattern
  app.search.batch_runner already uses for its own stage1/2/3 and
  app.orchestration.full_pipeline uses for its multi-strategy batch --
  see EvolutionConfig.parallel_workers. CPCV/PBO, the stress test, and
  clustering stay serial (they only run against the small `cpcv_top_n`
  survivor pool per generation, not the full population, so parallelizing
  them buys much less for the added complexity).
"""
from __future__ import annotations

import multiprocessing
import os
import random
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.evolution import checkpoint as evo_checkpoint
from app.evolution.knowledge_graph import DEFAULT_KG_PATH, KnowledgeGraph, feature_vector_for_spec
from app.evolution.prop_fitness import PropFitnessBreakdown, compute_prop_fitness
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import apply_genome, extract_genome
from app.optimize.refinement import _mutate, _stressed_risk_config
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.search.robustness import parameter_neighborhood_robustness, run_walk_forward
from app.search.strategy_space import build_strategy_from_spec, generate_search_space
from app.strategy.library import StrategyAlreadyExists, save_strategy_text, set_strategy_status
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv

ProgressCallback = "Callable[[str], None]"


# ---------------------------------------------------------------------------
# Worker process state & tasks (module-level so ProcessPoolExecutor can
# pickle/import them; state is per-process, populated once by
# _evo_init_worker -- same pattern app.search.batch_runner already uses).
# ---------------------------------------------------------------------------

_EVO_WORKER: dict = {}


def _evo_init_worker(df_pickle_path: str, risk_kwargs: dict, prop_kwargs: dict) -> None:
    global _EVO_WORKER
    _EVO_WORKER["df"] = pd.read_pickle(df_pickle_path)
    _EVO_WORKER["risk"] = RiskConfig(**risk_kwargs)
    _EVO_WORKER["prop_rules"] = PropRules(**prop_kwargs)


def _evo_prefilter_task(
    cid: str, spec: dict, meta: dict,
    min_trades: int, min_profit_factor: float, max_drawdown_pct: float, max_drawdown_buffer_mult: float,
):
    """One candidate's PRE-FILTER: build + backtest + the cheap pass/fail
    test, run in a worker process. Mirrors
    EvolutionRunner._prefilter_one's body exactly -- that instance method
    is what actually runs this when parallel_workers==1 or the pool is
    unavailable, so there's exactly one place the filter logic lives;
    this just re-parametrizes it without `self` so it can cross a
    process boundary. Returns a plain tuple (never raises) so one bad
    generated spec can't take down the whole prefilter batch."""
    df, risk = _EVO_WORKER["df"], _EVO_WORKER["risk"]
    try:
        strategy = build_strategy_from_spec(spec)
        bt = run_backtest(df, strategy, risk)
    except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the pool
        return (cid, spec, meta, None, ["build_or_backtest_error"], str(exc)[:300], None)
    if not bt.trades:
        return (cid, spec, meta, None, ["no_trades"], None, None)

    stats = bt.statistics.to_dict()
    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0

    reasons = []
    if n_trades < min_trades:
        reasons.append("min_trades")
    if pf_val < min_profit_factor:
        reasons.append("profit_factor")
    if max_dd > max_drawdown_pct * max_drawdown_buffer_mult:
        reasons.append("max_drawdown")
    if stats.get("net_profit", 0.0) <= 0:
        reasons.append("unprofitable")

    if reasons:
        return (cid, spec, meta, None, reasons, None, stats)
    return (cid, spec, meta, bt, [], None, stats)


def _evo_full_eval_task(
    cid: str, spec: dict, meta: dict, bt,
    mc_sims: int, robustness_perturbation_frac: float, robustness_neighbors: int,
    robustness_min_stability: float, walk_forward_folds: int, walk_forward_metric: str,
    min_trades_target_for_fitness: int, random_seed: int,
):
    """One PRE-FILTER survivor's ROBUSTNESS / OOS / MONTE CARLO / PROP
    SIMULATION scoring, run in a worker process. Mirrors
    EvolutionRunner._full_eval_one's body exactly (same one-source-of-
    truth reasoning as _evo_prefilter_task above)."""
    df, risk, prop_rules = _EVO_WORKER["df"], _EVO_WORKER["risk"], _EVO_WORKER["prop_rules"]
    stats = bt.statistics.to_dict()
    trade_pnls = [t.pnl for t in bt.trades]
    trade_dates = [t.entry_time for t in bt.trades]

    mc_cfg = MonteCarloConfig(n_simulations=mc_sims, random_seed=random_seed)
    mc = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
    mc_summary = {
        "evaluation_pass_probability": mc.evaluation_pass_probability,
        "first_payout_probability": mc.first_payout_probability,
    }
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    summarize_single_run(single_run)  # surfaces prop-sim issues early; summary itself not needed downstream here

    robustness_dict = None
    try:
        robustness = parameter_neighborhood_robustness(
            spec, df, risk, prop_rules, mc_cfg,
            fitness_metric="eval_pass_probability",
            perturbation_frac=robustness_perturbation_frac,
            n_neighbors=robustness_neighbors,
            seed=random_seed,
            stability_threshold=robustness_min_stability,
        )
        if robustness is not None:
            robustness_dict = {"stability_ratio": robustness.stability_ratio, "is_stable": robustness.is_stable}
    except Exception:
        pass

    wf_dict = None
    try:
        wf = run_walk_forward(
            df, lambda spec=spec: build_strategy_from_spec(spec), risk,
            n_folds=walk_forward_folds, metric=walk_forward_metric,
            prop_rules=prop_rules, mc_cfg=mc_cfg,
        )
        if wf is not None:
            wf_dict = {"walk_forward_efficiency": wf.walk_forward_efficiency, "is_stable": wf.is_stable}
    except Exception:
        pass

    fitness = compute_prop_fitness(
        stats, mc_summary, robustness_dict, wf_dict, trade_pnls,
        min_trades_target=min_trades_target_for_fitness,
    )
    return EvolutionCandidateRecord(
        candidate_id=cid, spec=spec, meta=meta, stats=stats, mc_summary=mc_summary,
        robustness=robustness_dict, walk_forward=wf_dict, fitness=fitness, trade_pnls=trade_pnls,
        trades=bt.trades,
    )


@dataclass
class EvolutionConfig:
    population_size: int = 60
    elite_keep: int = 10
    families: list[str] | None = None          # None = every family in list_families()
    grid_points_per_gene: int = 3

    # Pre-filter (Stage 1, cheap)
    min_trades: int = 20
    min_profit_factor: float = 1.05
    max_drawdown_buffer_mult: float = 1.5

    # Robustness / OOS
    robustness_neighbors: int = 4
    robustness_perturbation_frac: float = 0.15
    robustness_min_stability: float = 0.4
    walk_forward_folds: int = 4
    walk_forward_metric: str = "eval_pass_probability"

    # Monte Carlo
    mc_sims: int = 1000

    # CPCV / PBO -- only run against the best `cpcv_top_n` survivors, since
    # genuine CPCV re-backtests every candidate up to `cpcv_max_paths` times.
    cpcv_top_n: int = 8
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_max_paths: int = 10
    cpcv_metric: str = "eval_pass_probability"

    # Stress test
    stress_cost_multiplier: float = 2.0

    # Cluster (dedupe near-identical survivors before picking the top N)
    cluster_correlation_threshold: float = 0.85

    # Mutation / next generation
    mutation_rate: float = 0.35
    mutation_strength: float = 0.25
    random_immigrant_frac: float = 0.3

    min_trades_target_for_fitness: int = 30
    max_generations: int | None = None          # None = run until stop() is called
    random_seed: int = 42

    # Parallelism for PRE-FILTER and the ROBUSTNESS/OOS/MONTE CARLO/PROP
    # SIMULATION stage -- the two stages that run once per candidate and
    # dominate a generation's wall-clock time. None (default) auto-picks
    # os.cpu_count() (minimum 1); set to 1 to force the old fully serial
    # behavior (e.g. for debugging, or a machine where spawning worker
    # processes is undesirable). The pool is created once per run (not
    # once per generation) and reused across generations to avoid paying
    # worker-startup cost repeatedly.
    parallel_workers: int | None = None

    save_to_library: bool = True
    library_status: str = "draft"
    knowledge_graph_path: str = str(DEFAULT_KG_PATH)

    # Checkpoint / resume -- so STOP then START again continues from the
    # last completed generation (same elites, leaderboard, journal)
    # instead of starting a brand new run from scratch. Resuming is
    # refused (with a clear log message, falling back to a fresh run)
    # if the market data being started with doesn't match what the
    # checkpoint was built from -- see app.evolution.checkpoint.
    resume_from_checkpoint: bool = True
    checkpoint_path: str = str(evo_checkpoint.default_checkpoint_path())
    tested_log_path: str = str(evo_checkpoint.default_tested_log_path())

    # Auto-relax -- if this many generations IN A ROW produce zero
    # PRE-FILTER survivors, the pre-filter thresholds are automatically
    # loosened once (same idea as Search Lab's own Stage 1 auto-relax)
    # instead of the run silently grinding forever with an empty
    # leaderboard and no indication why.
    auto_relax_after_empty_generations: int = 3


@dataclass
class EvolutionCandidateRecord:
    candidate_id: str
    spec: dict
    meta: dict
    stats: dict | None = None
    mc_summary: dict | None = None
    robustness: dict | None = None
    walk_forward: dict | None = None
    pbo: float | None = None
    cpcv_degradation: float | None = None
    stressed_ok: bool | None = None
    fitness: object = None                     # PropFitnessBreakdown
    trade_pnls: list = field(default_factory=list)
    trades: list = field(default_factory=list)  # raw Trade objects, for date-aligned cluster correlation

    def to_checkpoint_dict(self) -> dict:
        """Serializes everything needed to redisplay this record and seed
        future generations -- NOT the raw `trades` objects (not JSON-
        serializable and only needed transiently for same-generation
        cluster-dedupe correlation), and pnls are capped since a
        checkpoint is meant to be a small, fast-to-load file, not a full
        trade-by-trade record (the strategy itself is always saved to
        the Strategy Library separately, in full, if save_to_library is on)."""
        return {
            "candidate_id": self.candidate_id,
            "spec": self.spec,
            "meta": self.meta,
            "stats": self.stats,
            "mc_summary": self.mc_summary,
            "robustness": self.robustness,
            "walk_forward": self.walk_forward,
            "pbo": self.pbo,
            "cpcv_degradation": self.cpcv_degradation,
            "stressed_ok": self.stressed_ok,
            "fitness": self.fitness.to_dict() if self.fitness is not None else None,
            "trade_pnls": self.trade_pnls[:500],
        }


def _record_from_dict(d: dict) -> EvolutionCandidateRecord:
    fitness = PropFitnessBreakdown(**d["fitness"]) if d.get("fitness") else None
    return EvolutionCandidateRecord(
        candidate_id=d.get("candidate_id", ""),
        spec=d.get("spec") or {},
        meta=d.get("meta") or {},
        stats=d.get("stats"),
        mc_summary=d.get("mc_summary"),
        robustness=d.get("robustness"),
        walk_forward=d.get("walk_forward"),
        pbo=d.get("pbo"),
        cpcv_degradation=d.get("cpcv_degradation"),
        stressed_ok=d.get("stressed_ok"),
        fitness=fitness,
        trade_pnls=d.get("trade_pnls") or [],
        trades=[],
    )


def _daily_pnl_series(trades) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    rows = [(pd.Timestamp(t.exit_time or t.entry_time).normalize(), t.pnl) for t in trades]
    s = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    return s.groupby(level=0).sum()


class EvolutionRunner:
    """Owns one Evolution Lab run: a background thread cycling through
    generations until stop() is called or max_generations is reached.
    Not thread-safe against being start()ed twice concurrently -- callers
    (the UI) are expected to check .is_running first, same convention as
    every other background job in this app."""

    def __init__(
        self,
        df: pd.DataFrame,
        risk: RiskConfig,
        prop_rules: PropRules,
        cfg: EvolutionConfig | None = None,
        progress_cb=None,
    ):
        self.df = df
        self.risk = risk
        self.prop_rules = prop_rules
        self.cfg = cfg or EvolutionConfig()
        self.progress_cb = progress_cb
        self.knowledge_graph = KnowledgeGraph(Path(self.cfg.knowledge_graph_path))

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self.is_running = False
        self.generation = 0
        self.leaderboard: list[EvolutionCandidateRecord] = []      # current top N (all-time best seen)
        self.journal: list[str] = []                                 # numbered HYPOTHESIS-style entries
        self._elites: list[tuple[dict, dict]] = []                    # seeds mutated children next gen
        self._consecutive_empty_generations = 0
        self.resumed = False                                          # set True if a checkpoint was loaded
        self._pool: ProcessPoolExecutor | None = None
        self._pool_tmp_dir: tempfile.TemporaryDirectory | None = None

        self._load_checkpoint_if_compatible()

    # -- worker pool (PRE-FILTER + full-eval parallelism) ------------------
    def _ensure_pool(self) -> ProcessPoolExecutor | None:
        """Lazily creates the ProcessPoolExecutor used by _prefilter and
        _full_eval, reused across every generation of this run (workers
        are expensive to start, cheap to keep alive). Returns None (never
        raises) if parallel_workers resolves to 1 or the pool fails to
        start for any reason -- both _prefilter and _full_eval fall back
        to their serial per-candidate loop in that case, so a machine
        where spawning worker processes doesn't work still runs, just
        without the speedup."""
        if self._pool is not None:
            return self._pool
        workers = self.cfg.parallel_workers
        if workers is None:
            workers = os.cpu_count() or 1
        if workers <= 1:
            return None
        try:
            self._pool_tmp_dir = tempfile.TemporaryDirectory(prefix="t58_evolution_")
            df_path = Path(self._pool_tmp_dir.name) / "data.pkl"
            self.df.to_pickle(df_path)
            self._pool = ProcessPoolExecutor(
                max_workers=workers, initializer=_evo_init_worker,
                initargs=(str(df_path), asdict(self.risk), asdict(self.prop_rules)),
                # spawn, not the platform default fork: this runner always
                # lives inside a background thread (see start()), and
                # forking a multi-threaded process is unsafe/deprecated --
                # same reasoning as app.search.batch_runner's own pool.
                mp_context=multiprocessing.get_context("spawn"),
            )
            return self._pool
        except Exception:
            self._log(
                "Could not start a worker process pool for this run -- "
                "continuing single-process (slower, but still fully "
                "functional)."
            )
            self._shutdown_pool()
            return None

    def _shutdown_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None
        if self._pool_tmp_dir is not None:
            try:
                self._pool_tmp_dir.cleanup()
            except Exception:
                pass
            self._pool_tmp_dir = None

    # -- checkpoint / resume ----------------------------------------------
    def _load_checkpoint_if_compatible(self) -> None:
        """Loads generation/elites/leaderboard/journal from disk if the
        config asks to resume AND the checkpoint was built from the same
        market data this runner is starting with. Otherwise starts clean
        (still logging why, so a mismatched-data situation isn't silent)."""
        if not self.cfg.resume_from_checkpoint:
            return
        saved = evo_checkpoint.load_checkpoint(Path(self.cfg.checkpoint_path))
        if saved is None:
            return
        current_fp = evo_checkpoint.data_fingerprint(self.df)
        if saved.data_fingerprint and saved.data_fingerprint != current_fp:
            self._log(
                f"Found a saved Evolution Lab checkpoint (generation {saved.generation}, "
                f"{len(saved.leaderboard)} on its leaderboard) but it was built from different "
                f"market data than what's loaded now -- starting a fresh run instead of resuming "
                f"against mismatched data. (Use the same data file to resume that run.)"
            )
            return
        try:
            self.generation = saved.generation
            self._elites = [(e["spec"], e["meta"]) for e in saved.elites]
            self.leaderboard = [_record_from_dict(r) for r in saved.leaderboard]
            self.journal = list(saved.journal)
            self.resumed = True
            self._log(
                f"Resuming Evolution Lab from checkpoint: generation {self.generation}, "
                f"{len(self.leaderboard)} on the leaderboard, {len(self.journal)} journal "
                f"entries carried over. Click STOP at any time -- progress keeps saving after "
                f"every generation."
            )
        except Exception:
            self._log("Found a saved Evolution Lab checkpoint but couldn't load it cleanly -- starting fresh.")
            self.generation = 0
            self._elites = []
            self.leaderboard = []
            self.journal = []
            self.resumed = False

    def _save_checkpoint(self, next_generation: int) -> None:
        try:
            ckpt = evo_checkpoint.EvolutionCheckpoint(
                generation=next_generation,
                elites=[{"spec": s, "meta": m} for s, m in self._elites],
                leaderboard=[r.to_checkpoint_dict() for r in self.leaderboard],
                journal=list(self.journal),
                data_fingerprint=evo_checkpoint.data_fingerprint(self.df),
                saved_at=pd.Timestamp.now("UTC").isoformat(),
            )
            evo_checkpoint.save_checkpoint(ckpt, Path(self.cfg.checkpoint_path))
        except Exception:
            pass  # checkpointing is best-effort -- must never break the run itself

    def reset(self) -> None:
        """Discards the on-disk checkpoint and tested-candidates log so the
        next START begins a genuinely fresh run. Only safe to call while
        not running."""
        evo_checkpoint.clear_checkpoint(Path(self.cfg.checkpoint_path))
        evo_checkpoint.clear_tested_log(Path(self.cfg.tested_log_path))
        self.generation = 0
        self._elites = []
        self.leaderboard = []
        self.journal = []
        self.resumed = False

    # -- public controls ------------------------------------------------
    def start(self) -> None:
        if self.is_running:
            return
        self._stop_flag.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "generation": self.generation,
            "leaderboard_size": len(self.leaderboard),
            "resumed": self.resumed,
        }

    def tested_candidates(self, limit: int = 500) -> list[dict]:
        """The "what was actually tested" record -- every candidate the
        PRE-FILTER stage has backtested this run (and prior resumed runs
        against the same data), pass or fail, with the reason it was
        rejected if it was. Read from disk so it survives a restart same
        as the checkpoint does."""
        return evo_checkpoint.read_tested_rows(Path(self.cfg.tested_log_path), limit=limit)

    # -- internals --------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    def _run_loop(self) -> None:
        gen = self.generation
        try:
            while not self._stop_flag.is_set():
                if self.cfg.max_generations is not None and gen >= self.cfg.max_generations:
                    break
                self.generation = gen
                t0 = time.time()
                self._log(f"===== GENERATION {gen} =====")

                population = self._generate_population(gen, self._elites)
                self._log(f"GENERATE: {len(population)} candidates.")
                if self._stop_flag.is_set():
                    break

                stage1_survivors, rejection_counts = self._prefilter(population, gen)
                self._log(f"PRE-FILTER + BACKTEST: {len(stage1_survivors)}/{len(population)} survived.")
                if not stage1_survivors:
                    self._handle_empty_prefilter(gen, rejection_counts)
                if self._stop_flag.is_set() or not stage1_survivors:
                    self._finish_empty_generation(gen, population)
                    self._save_checkpoint(next_generation=gen + 1)
                    gen += 1
                    continue
                self._consecutive_empty_generations = 0

                evaluated = self._full_eval(stage1_survivors)
                self._log(f"ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION: {len(evaluated)} candidates scored.")
                if self._stop_flag.is_set() or not evaluated:
                    self._finish_empty_generation(gen, population, evaluated)
                    self._save_checkpoint(next_generation=gen + 1)
                    gen += 1
                    continue

                cpcv_pool = self._cpcv_and_pbo(evaluated)
                self._log(f"CPCV / PBO: re-scored top {len(cpcv_pool)} candidates.")

                stress_survivors = self._stress_test(cpcv_pool)
                self._log(f"STRESS TEST ({self.cfg.stress_cost_multiplier:g}x costs): "
                          f"{len(stress_survivors)}/{len(cpcv_pool)} still fitness-positive.")

                clustered = self._cluster(stress_survivors or cpcv_pool)
                self._log(f"CLUSTER: {len(clustered)} distinct candidates remain.")

                clustered.sort(key=lambda r: r.fitness.final_score, reverse=True)
                new_elites = clustered[: self.cfg.elite_keep]

                self._record_generation_to_knowledge_graph(evaluated, {r.candidate_id for r in new_elites})
                self._append_tested_log_full_eval(evaluated, gen)
                self._update_leaderboard(new_elites)
                self._maybe_save_to_library(new_elites)
                self._write_journal_entry(gen, population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, new_elites)

                elapsed = time.time() - t0
                self._log(f"Generation {gen} complete in {elapsed:.1f}s. Best fitness so far: "
                          f"{self.leaderboard[0].fitness.final_score:.2f}" if self.leaderboard else f"Generation {gen} complete in {elapsed:.1f}s.")

                self._elites = [(r.spec, r.meta) for r in new_elites]
                self._save_checkpoint(next_generation=gen + 1)
                gen += 1
        except Exception:
            self._log("Evolution Lab crashed:\n" + traceback.format_exc())
        finally:
            self._shutdown_pool()
            self.is_running = False
            self._log("Evolution Lab stopped. Progress is saved -- clicking START again resumes from here.")

    def _handle_empty_prefilter(self, gen: int, rejection_counts: dict) -> None:
        """Logs WHY nothing survived (instead of just "0 survived", which
        gives no way to tell "your data/settings genuinely can't produce
        a passing strategy" from "something's silently broken"), and
        auto-relaxes the pre-filter thresholds once every
        `auto_relax_after_empty_generations` consecutive empty
        generations -- same philosophy as Search Lab's own Stage 1
        auto-relax (app.search.batch_runner), so a run doesn't grind for
        hours with a leaderboard stuck at 0 for no visible reason."""
        breakdown = ", ".join(f"{k}: {v}" for k, v in rejection_counts.items() if v) or "no candidates built successfully"
        self._log(f"  Rejection breakdown -- {breakdown}.")
        self._consecutive_empty_generations += 1
        if self._consecutive_empty_generations >= self.cfg.auto_relax_after_empty_generations:
            old_trades, old_pf = self.cfg.min_trades, self.cfg.min_profit_factor
            self.cfg.min_trades = max(5, int(self.cfg.min_trades * 0.6))
            self.cfg.min_profit_factor = max(1.0, round(self.cfg.min_profit_factor * 0.9, 3))
            self.cfg.max_drawdown_buffer_mult = round(self.cfg.max_drawdown_buffer_mult * 1.25, 3)
            self._log(
                f"  AUTO-RELAX: {self._consecutive_empty_generations} generations in a row produced zero "
                f"pre-filter survivors, so the thresholds were automatically loosened -- min trades "
                f"{old_trades} -> {self.cfg.min_trades}, min profit factor {old_pf:.2f} -> "
                f"{self.cfg.min_profit_factor:.2f}, drawdown buffer x{self.cfg.max_drawdown_buffer_mult:.2f}. "
                f"If generations keep coming back empty even after this, the market data or the selected "
                f"families likely can't produce a profitable strategy at all on this instrument/timeframe."
            )
            self._consecutive_empty_generations = 0

    def _finish_empty_generation(self, gen, population, evaluated=None) -> None:
        self._log(f"Generation {gen}: nothing survived far enough to update the leaderboard -- continuing to the next generation.")

    # -- GENERATE ---------------------------------------------------------
    def _generate_population(self, gen: int, elites: list[tuple[dict, dict]]) -> list[tuple[str, dict, dict]]:
        """Returns a list of (candidate_id, spec, meta). Generation 0 is
        pure random family sampling. Later generations mix mutated
        children of the previous top N with a fresh slice of random
        immigrants for diversity (same random-immigrant idea the
        walk-forward GA already uses, applied at the population level)."""
        seed = self.cfg.random_seed + gen
        n_immigrants = self.cfg.population_size if not elites else max(1, int(self.cfg.population_size * self.cfg.random_immigrant_frac))
        space = generate_search_space(
            mode="family", family=(None if not self.cfg.families else "all"),
            max_candidates=n_immigrants, seed=seed, grid_points_per_gene=self.cfg.grid_points_per_gene,
        )
        out = [(cid, spec, space.meta[cid]) for cid, spec in space.candidates.items()]
        if self.cfg.families:
            out = [row for row in out if row[2]["family"] in self.cfg.families] or out

        if elites:
            rng = random.Random(seed)
            n_children = max(0, self.cfg.population_size - len(out))
            per_elite = max(1, n_children // len(elites))
            for spec, meta in elites:
                config = spec.get("config")
                if not config:
                    continue
                genes = extract_genome(config)
                if not genes:
                    continue
                base_genome = [g.base_value for g in genes]
                for _ in range(per_elite):
                    child_genome = _mutate(base_genome, genes, self.cfg.mutation_rate, self.cfg.mutation_strength, rng)
                    child_config = apply_genome(config, genes, child_genome)
                    child_spec = {"source_type": "manual", "config": child_config}
                    cid = f"{meta.get('family', 'mutant')}-gen{gen}-{rng.randrange(10**8):08x}"
                    out.append((cid, child_spec, {"family": meta.get("family", "mutant"), "params": {}, "mutated_from": meta.get("family")}))
        return out[: max(self.cfg.population_size, len(out))]

    # -- PRE-FILTER + BACKTEST --------------------------------------------
    def _prefilter(self, population: list[tuple[str, dict, dict]], gen: int):
        """Runs every candidate in `population` through build+backtest+the
        cheap pass/fail test. Dispatches across the worker pool (see
        _ensure_pool) when one is available, falling back to a plain
        serial loop -- in-process, calling the exact same
        _evo_prefilter_task function each candidate would run in a
        worker -- if the pool couldn't be started or parallel_workers==1.
        Either path produces identical survivors/rejection_counts/
        tested_rows; only the wall-clock time differs."""
        survivors = []
        rejection_counts = {
            "build_or_backtest_error": 0, "no_trades": 0, "min_trades": 0,
            "profit_factor": 0, "max_drawdown": 0, "unprofitable": 0,
        }
        tested_rows = []

        def _consume(cid, spec, meta, bt, reasons, error, stats):
            # Mirrors _evo_prefilter_task's four possible return shapes
            # exactly (see its docstring/body): a build/backtest error, a
            # zero-trade candidate, a candidate that ran but failed one or
            # more cheap filters, or a genuine survivor.
            if error is not None:
                rejection_counts["build_or_backtest_error"] += 1
                tested_rows.append(self._tested_row(cid, meta, gen, passed=False, reasons=reasons, error=error))
                return
            if stats is None:
                rejection_counts["no_trades"] += 1
                tested_rows.append(self._tested_row(cid, meta, gen, passed=False, reasons=reasons))
                return
            n_trades = stats.get("total_trades", 0)
            pf = stats.get("profit_factor", 0.0)
            pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
            max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
            tested_rows.append(self._tested_row(
                cid, meta, gen, passed=not reasons, reasons=reasons,
                n_trades=n_trades, profit_factor=pf_val, net_profit=stats.get("net_profit"),
                max_drawdown_pct=max_dd,
            ))
            if reasons:
                for r in reasons:
                    rejection_counts[r] += 1
            else:
                survivors.append((cid, spec, meta, bt))

        pool = self._ensure_pool()
        if pool is None:
            # Serial fallback still calls _evo_prefilter_task (single
            # source of truth for the filter logic) -- it just runs it
            # in-process instead of in a worker, so _EVO_WORKER (normally
            # populated once per worker process by _evo_init_worker) needs
            # seeding here too.
            _EVO_WORKER["df"], _EVO_WORKER["risk"] = self.df, self.risk
            for cid, spec, meta in population:
                if self._stop_flag.is_set():
                    break
                cid_out, spec_out, meta_out, bt, reasons, error, stats = _evo_prefilter_task(
                    cid, spec, meta, self.cfg.min_trades, self.cfg.min_profit_factor,
                    self.prop_rules.max_drawdown_pct, self.cfg.max_drawdown_buffer_mult,
                )
                _consume(cid_out, spec_out, meta_out, bt, reasons, error, stats)
        else:
            futures = {
                pool.submit(
                    _evo_prefilter_task, cid, spec, meta, self.cfg.min_trades, self.cfg.min_profit_factor,
                    self.prop_rules.max_drawdown_pct, self.cfg.max_drawdown_buffer_mult,
                ): (cid, spec, meta)
                for cid, spec, meta in population
            }
            for future in as_completed(futures):
                if self._stop_flag.is_set():
                    break
                cid, spec, meta = futures[future]
                try:
                    _, _, _, bt, reasons, error, stats = future.result()
                except Exception as exc:  # noqa: BLE001 -- a dead worker must not kill the generation
                    _consume(cid, spec, meta, None, ["build_or_backtest_error"], str(exc)[:300], None)
                    continue
                _consume(cid, spec, meta, bt, reasons, error, stats)

        evo_checkpoint.append_tested_rows(tested_rows, Path(self.cfg.tested_log_path))
        return survivors, rejection_counts

    def _tested_row(self, cid: str, meta: dict, gen: int, passed: bool, reasons: list[str],
                     n_trades=None, profit_factor=None, net_profit=None, max_drawdown_pct=None,
                     error: str | None = None) -> dict:
        """One row of the durable "what was tested" log -- deliberately
        flat/small (no spec/config blob) so the log stays cheap to append
        to and to read back for a multi-hour, many-thousand-candidate
        run; the full spec for anything that mattered (an elite/leaderboard
        entry) is separately captured in the checkpoint and, when
        save_to_library is on, the Strategy Library."""
        return {
            "generation": gen,
            "candidate_id": cid,
            "family": meta.get("family", "?"),
            "stage": "prefilter",
            "passed": passed,
            "reasons": reasons,
            "n_trades": n_trades,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "max_drawdown_pct": max_drawdown_pct,
            "error": error,
        }

    def _append_tested_log_full_eval(self, evaluated: list["EvolutionCandidateRecord"], gen: int) -> None:
        """Appends a second row for every candidate that made it past
        PRE-FILTER into the expensive robustness/OOS/Monte Carlo/prop-sim
        stage, this time with the PROP FITNESS score -- so "what was
        tested" shows not just the cheap pass/fail but how far a
        candidate actually got and how good it turned out to be."""
        rows = []
        for r in evaluated:
            rows.append({
                "generation": gen,
                "candidate_id": r.candidate_id,
                "family": r.meta.get("family", "?"),
                "stage": "full_eval",
                "passed": True,
                "reasons": [],
                "n_trades": (r.stats or {}).get("total_trades"),
                "profit_factor": (r.stats or {}).get("profit_factor"),
                "net_profit": (r.stats or {}).get("net_profit"),
                "max_drawdown_pct": (r.stats or {}).get("max_drawdown_pct"),
                "fitness_score": r.fitness.final_score if r.fitness else None,
                "pass_probability": (r.mc_summary or {}).get("evaluation_pass_probability"),
                "error": None,
            })
        evo_checkpoint.append_tested_rows(rows, Path(self.cfg.tested_log_path))

    # -- ROBUSTNESS + OOS + MONTE CARLO + PROP SIMULATION ------------------
    def _full_eval(self, stage1_survivors) -> list[EvolutionCandidateRecord]:
        """Runs every PRE-FILTER survivor through ROBUSTNESS / OOS / MONTE
        CARLO / PROP SIMULATION scoring. Same dispatch-to-pool-or-fall-
        back-serial shape as _prefilter (see its docstring) -- both paths
        call _evo_full_eval_task, so there's one place this logic lives."""
        args = (
            self.cfg.mc_sims, self.cfg.robustness_perturbation_frac, self.cfg.robustness_neighbors,
            self.cfg.robustness_min_stability, self.cfg.walk_forward_folds, self.cfg.walk_forward_metric,
            self.cfg.min_trades_target_for_fitness, self.cfg.random_seed,
        )
        records: list[EvolutionCandidateRecord] = []
        eval_pool = self._ensure_pool()
        if eval_pool is None:
            _EVO_WORKER["df"], _EVO_WORKER["risk"], _EVO_WORKER["prop_rules"] = self.df, self.risk, self.prop_rules
            for cid, spec, meta, bt in stage1_survivors:
                if self._stop_flag.is_set():
                    break
                try:
                    records.append(_evo_full_eval_task(cid, spec, meta, bt, *args))
                except Exception:
                    self._log(f"  full-eval error on {cid}:\n" + traceback.format_exc())
        else:
            futures = {
                eval_pool.submit(_evo_full_eval_task, cid, spec, meta, bt, *args): cid
                for cid, spec, meta, bt in stage1_survivors
            }
            for future in as_completed(futures):
                if self._stop_flag.is_set():
                    break
                cid = futures[future]
                try:
                    records.append(future.result())
                except Exception:  # noqa: BLE001 -- a dead worker must not kill the generation
                    self._log(f"  full-eval error on {cid}:\n" + traceback.format_exc())
        return records

    # -- CPCV / PBO ---------------------------------------------------------
    def _cpcv_and_pbo(self, evaluated: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        pool = sorted(evaluated, key=lambda r: r.fitness.final_score, reverse=True)[: self.cfg.cpcv_top_n]
        if len(pool) < 2:
            return pool

        pbo_value = None
        try:
            pbo_result = compute_pbo(
                self.df, [r.spec for r in pool], self.risk,
                n_groups=self.cfg.cpcv_n_groups, n_test_groups=self.cfg.cpcv_n_test_groups,
                metric=self.cfg.cpcv_metric, max_paths=self.cfg.cpcv_max_paths,
                prop_rules=self.prop_rules,
            )
            pbo_value = pbo_result.pbo
        except Exception:
            pass

        for r in pool:
            if self._stop_flag.is_set():
                break
            cpcv_degradation = None
            try:
                cpcv_result = run_cpcv(
                    self.df, lambda spec=r.spec: build_strategy_from_spec(spec), self.risk,
                    n_groups=self.cfg.cpcv_n_groups, n_test_groups=self.cfg.cpcv_n_test_groups,
                    metric=self.cfg.cpcv_metric, max_paths=self.cfg.cpcv_max_paths,
                    prop_rules=self.prop_rules,
                )
                cpcv_degradation = cpcv_result.degradation
            except CPCVError:
                pass
            except Exception:
                pass
            r.pbo = pbo_value
            r.cpcv_degradation = cpcv_degradation
            r.fitness = compute_prop_fitness(
                r.stats, r.mc_summary, r.robustness, r.walk_forward, r.trade_pnls,
                pbo=pbo_value, cpcv_degradation=cpcv_degradation,
                min_trades_target=self.cfg.min_trades_target_for_fitness,
            )
        return pool

    # -- STRESS TEST ---------------------------------------------------------
    def _stress_test(self, pool: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        stressed_risk = _stressed_risk_config(self.risk, self.cfg.stress_cost_multiplier)
        survivors = []
        for r in pool:
            if self._stop_flag.is_set():
                break
            try:
                strategy = build_strategy_from_spec(r.spec)
                bt = run_backtest(self.df, strategy, stressed_risk)
                r.stressed_ok = bool(bt.trades) and bt.statistics.net_profit > 0
            except Exception:
                r.stressed_ok = False
            if r.stressed_ok:
                survivors.append(r)
        return survivors

    # -- CLUSTER -------------------------------------------------------------
    def _cluster(self, pool: list[EvolutionCandidateRecord]) -> list[EvolutionCandidateRecord]:
        """Correlation-dedupe on DAILY P&L (date-aligned, zero-filled on
        days either strategy didn't trade) -- comparing raw trade-by-trade
        pnl lists positionally would be meaningless since two strategies
        rarely have the same number of trades on the same days."""
        ranked = sorted(pool, key=lambda r: r.fitness.final_score, reverse=True)
        kept: list[EvolutionCandidateRecord] = []
        kept_series: list[pd.Series] = []
        for r in ranked:
            series = _daily_pnl_series(r.trades)
            is_duplicate = False
            for other in kept_series:
                if series.empty or other.empty:
                    continue
                idx = series.index.union(other.index)
                a = series.reindex(idx, fill_value=0.0)
                b = other.reindex(idx, fill_value=0.0)
                if len(idx) < 5 or a.std() == 0 or b.std() == 0:
                    continue
                corr = a.corr(b)
                if corr is not None and corr >= self.cfg.cluster_correlation_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(r)
                kept_series.append(series)
        return kept

    # -- knowledge graph / leaderboard / library / journal -------------------
    def _record_generation_to_knowledge_graph(self, evaluated: list[EvolutionCandidateRecord], elite_ids: set) -> None:
        for r in evaluated:
            fv = feature_vector_for_spec(r.spec, r.meta)
            outcome = {
                "passed": r.candidate_id in elite_ids,
                "final_score": r.fitness.final_score if r.fitness else None,
                "generation": self.generation,
            }
            try:
                self.knowledge_graph.record(fv, outcome)
            except Exception:
                pass

    def _update_leaderboard(self, new_elites: list[EvolutionCandidateRecord]) -> None:
        combined = {r.candidate_id: r for r in (self.leaderboard + new_elites)}
        ranked = sorted(combined.values(), key=lambda r: r.fitness.final_score, reverse=True)
        self.leaderboard = ranked[: self.cfg.elite_keep]

    def _maybe_save_to_library(self, new_elites: list[EvolutionCandidateRecord]) -> None:
        if not self.cfg.save_to_library:
            return
        for r in new_elites:
            config = r.spec.get("config")
            if not config:
                continue
            import json
            text = json.dumps(config, indent=2)
            base_name = f"evolab_gen{self.generation}_{r.meta.get('family', 'strategy')}_{r.candidate_id[-6:]}"
            filename = f"{base_name}.json"
            try:
                try:
                    save_strategy_text(text, filename, "manual", overwrite=False)
                except StrategyAlreadyExists:
                    continue
                set_strategy_status("manual", filename, self.cfg.library_status)
            except Exception:
                pass

    def _write_journal_entry(self, gen, population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, new_elites) -> None:
        n = len(self.journal) + 1
        winner = new_elites[0] if new_elites else None
        confidence = "LOW"
        if winner is not None and winner.fitness and winner.fitness.final_score > 0:
            n_similar_records = self.knowledge_graph.query_similar(feature_vector_for_spec(winner.spec, winner.meta), top_k=20)
            n_similar = sum(1 for s, _ in n_similar_records if s >= 0.6)
            stable = bool(winner.robustness and winner.robustness.get("is_stable"))
            if stable and n_similar >= 5:
                confidence = "HIGH"
            elif stable or n_similar >= 2:
                confidence = "MEDIUM"

        lines = [
            f"HYPOTHESIS #{n} (generation {gen})",
            "",
            f"TEST: {len(population)} strategy variants",
            f"RESULT: {len(stage1_survivors)} survived initial screening",
            f"OOS/ROBUSTNESS: {len(evaluated)} survived",
            f"CPCV: {len(cpcv_pool)} re-scored",
            f"STRESS: {len(stress_survivors)} survived {self.cfg.stress_cost_multiplier:g}x costs",
        ]
        if winner is not None:
            lines.append(f"WINNER: {winner.candidate_id}  (PROP FITNESS {winner.fitness.final_score:.2f})")
            lines.append(f"CONFIDENCE: {confidence}")
            lines.append("")
            lines.append(self.knowledge_graph.describe(feature_vector_for_spec(winner.spec, winner.meta)))
        else:
            lines.append("WINNER: none this generation")
            lines.append("CONFIDENCE: LOW")
        entry = "\n".join(lines)
        self.journal.append(entry)
        self._log("\n" + entry + "\n")
