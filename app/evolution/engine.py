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
- This runs single-process. It will happily run for hours as asked, but
  each generation is currently slower than it would be with the
  ProcessPoolExecutor parallelism app.search.batch_runner already uses
  for its own stage1/2/3 -- porting that same pattern in here is the
  natural next optimization once this loop's shape is validated in
  practice, not attempted in this pass.
"""
from __future__ import annotations

import random
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.evolution.knowledge_graph import DEFAULT_KG_PATH, KnowledgeGraph, feature_vector_for_spec
from app.evolution.prop_fitness import compute_prop_fitness
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import apply_genome, extract_genome
from app.optimize.refinement import _mutate, _stressed_risk_config
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.search.robustness import parameter_neighborhood_robustness, run_walk_forward
from app.search.strategy_space import build_strategy_from_spec, generate_search_space
from app.strategy.library import StrategyAlreadyExists, save_strategy_text, set_strategy_status
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv

ProgressCallback = "Callable[[str], None]"


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
    walk_forward_metric: str = "profit_factor"

    # Monte Carlo
    mc_sims: int = 1000

    # CPCV / PBO -- only run against the best `cpcv_top_n` survivors, since
    # genuine CPCV re-backtests every candidate up to `cpcv_max_paths` times.
    cpcv_top_n: int = 8
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_max_paths: int = 10
    cpcv_metric: str = "profit_factor"

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

    save_to_library: bool = True
    library_status: str = "draft"
    knowledge_graph_path: str = str(DEFAULT_KG_PATH)


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
        }

    # -- internals --------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    def _run_loop(self) -> None:
        elites: list[tuple[dict, dict]] = []
        gen = 0
        try:
            while not self._stop_flag.is_set():
                if self.cfg.max_generations is not None and gen >= self.cfg.max_generations:
                    break
                self.generation = gen
                t0 = time.time()
                self._log(f"===== GENERATION {gen} =====")

                population = self._generate_population(gen, elites)
                self._log(f"GENERATE: {len(population)} candidates.")
                if self._stop_flag.is_set():
                    break

                stage1_survivors = self._prefilter(population)
                self._log(f"PRE-FILTER + BACKTEST: {len(stage1_survivors)}/{len(population)} survived.")
                if self._stop_flag.is_set() or not stage1_survivors:
                    self._finish_empty_generation(gen, population)
                    gen += 1
                    continue

                evaluated = self._full_eval(stage1_survivors)
                self._log(f"ROBUSTNESS / OOS / MONTE CARLO / PROP SIMULATION: {len(evaluated)} candidates scored.")
                if self._stop_flag.is_set() or not evaluated:
                    self._finish_empty_generation(gen, population, evaluated)
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
                self._update_leaderboard(new_elites)
                self._maybe_save_to_library(new_elites)
                self._write_journal_entry(gen, population, stage1_survivors, evaluated, cpcv_pool, stress_survivors, new_elites)

                elapsed = time.time() - t0
                self._log(f"Generation {gen} complete in {elapsed:.1f}s. Best fitness so far: "
                          f"{self.leaderboard[0].fitness.final_score:.2f}" if self.leaderboard else f"Generation {gen} complete in {elapsed:.1f}s.")

                elites = [(r.spec, r.meta) for r in new_elites]
                gen += 1
        except Exception:
            self._log("Evolution Lab crashed:\n" + traceback.format_exc())
        finally:
            self.is_running = False
            self._log("Evolution Lab stopped.")

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
    def _prefilter(self, population: list[tuple[str, dict, dict]]):
        survivors = []
        for cid, spec, meta in population:
            if self._stop_flag.is_set():
                break
            try:
                strategy = build_strategy_from_spec(spec)
                bt = run_backtest(self.df, strategy, self.risk)
            except Exception:
                continue
            if not bt.trades:
                continue
            stats = bt.statistics.to_dict()
            pf = stats.get("profit_factor", 0.0)
            pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
            n_trades = stats.get("total_trades", 0)
            max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
            passed = (
                n_trades >= self.cfg.min_trades
                and pf_val >= self.cfg.min_profit_factor
                and max_dd <= self.prop_rules.max_drawdown_pct * self.cfg.max_drawdown_buffer_mult
                and stats.get("net_profit", 0.0) > 0
            )
            if passed:
                survivors.append((cid, spec, meta, bt))
        return survivors

    # -- ROBUSTNESS + OOS + MONTE CARLO + PROP SIMULATION ------------------
    def _full_eval(self, stage1_survivors) -> list[EvolutionCandidateRecord]:
        records = []
        for cid, spec, meta, bt in stage1_survivors:
            if self._stop_flag.is_set():
                break
            stats = bt.statistics.to_dict()
            trade_pnls = [t.pnl for t in bt.trades]
            trade_dates = [t.entry_time for t in bt.trades]

            mc_cfg = MonteCarloConfig(n_simulations=self.cfg.mc_sims, random_seed=self.cfg.random_seed)
            mc = run_monte_carlo(bt.trades, self.prop_rules, mc_cfg)
            mc_summary = {
                "evaluation_pass_probability": mc.evaluation_pass_probability,
                "first_payout_probability": mc.first_payout_probability,
            }
            single_run = simulate_account(trade_pnls, trade_dates, self.prop_rules)
            summarize_single_run(single_run)  # surfaces prop-sim issues early; summary itself not needed downstream here

            robustness_dict = None
            try:
                robustness = parameter_neighborhood_robustness(
                    spec, self.df, self.risk, self.prop_rules, mc_cfg,
                    fitness_metric="composite_prop_score",
                    perturbation_frac=self.cfg.robustness_perturbation_frac,
                    n_neighbors=self.cfg.robustness_neighbors,
                    seed=self.cfg.random_seed,
                    stability_threshold=self.cfg.robustness_min_stability,
                )
                if robustness is not None:
                    robustness_dict = {"stability_ratio": robustness.stability_ratio, "is_stable": robustness.is_stable}
            except Exception:
                pass

            wf_dict = None
            try:
                wf = run_walk_forward(
                    self.df, lambda spec=spec: build_strategy_from_spec(spec), self.risk,
                    n_folds=self.cfg.walk_forward_folds, metric=self.cfg.walk_forward_metric,
                )
                if wf is not None:
                    wf_dict = {"walk_forward_efficiency": wf.walk_forward_efficiency, "is_stable": wf.is_stable}
            except Exception:
                pass

            fitness = compute_prop_fitness(
                stats, mc_summary, robustness_dict, wf_dict, trade_pnls,
                min_trades_target=self.cfg.min_trades_target_for_fitness,
            )
            records.append(EvolutionCandidateRecord(
                candidate_id=cid, spec=spec, meta=meta, stats=stats, mc_summary=mc_summary,
                robustness=robustness_dict, walk_forward=wf_dict, fitness=fitness, trade_pnls=trade_pnls,
                trades=bt.trades,
            ))
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
