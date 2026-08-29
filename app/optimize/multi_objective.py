"""
Multi-objective optimization -- Pareto front across several metrics at
once (e.g. Sharpe Ratio, Max Drawdown, Evaluation Pass Probability),
instead of Iterative Refinement's single scalar fitness.

app.optimize.refinement's GA collapses everything into one number
(composite_prop_score by default) before it ever compares two
candidates. That's the right tool when you already know how you want to
trade those objectives off against each other. It's the wrong tool when
you don't -- collapsing "high Sharpe, low drawdown, high eval-pass
probability" into one weighted score BAKES IN a trade-off you didn't
actually choose, and the GA will happily sacrifice one for the other on
your behalf without telling you it happened.

This module runs a standard NSGA-II-style search instead: candidates are
compared by Pareto dominance (A dominates B only if A is at least as good
as B on EVERY objective and strictly better on at least one), sorted into
non-dominated "fronts", and selected via front-rank + crowding distance
(which additionally rewards spreading out along the front rather than
clustering). The result is a POOL of candidates -- the Pareto front --
each of which is the best available trade-off for some weighting of the
objectives; nothing in the front is strictly worse than anything else in
it. Picking a final winner from the front is a judgment call for the
person, which is exactly the point.

Reuses the exact same gene discovery/apply machinery (and therefore
supports the same manual/python/pinescript/mql5 source types) as
Iterative Refinement.
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import (
    _build_adapter,
    _crossover,
    _mutate,
    _random_gene_value,
    preflight_signal_check,
)
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy

ProgressCallback = Callable[[str], None]

# Every objective this module knows how to read off a candidate's
# evaluated stats/prop-summary/Monte-Carlo result, and which DIRECTION is
# "better" for each. "max" objectives are negated internally so every
# objective in NSGA-II's dominance/crowding math can be treated uniformly
# as "bigger is better".
OBJECTIVE_DIRECTIONS: dict[str, str] = {
    "sharpe_ratio": "max",
    "sortino_ratio": "max",
    "profit_factor": "max",
    "net_profit": "max",
    "win_rate": "max",
    "max_drawdown_pct": "min",
    "eval_pass_probability": "max",
    "first_payout_probability": "max",
    "risk_of_ruin_pct": "min",
    "expected_payout": "max",
}

DEFAULT_OBJECTIVES = ["sharpe_ratio", "max_drawdown_pct", "eval_pass_probability"]


def _raw_objective_value(objective: str, stats: dict, prop_summary: dict | None, mc: MonteCarloResult | None) -> float:
    if objective in stats:
        v = stats.get(objective, 0.0)
    elif mc is not None and hasattr(mc, objective):
        v = getattr(mc, objective)
    elif prop_summary is not None and objective in prop_summary:
        v = prop_summary.get(objective, 0.0)
    else:
        v = 0.0
    if v == float("inf"):
        return 10.0
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0.0
    return float(v)


def _objective_vector(objectives: list[str], stats: dict, prop_summary: dict | None, mc: MonteCarloResult | None) -> tuple:
    """Returns objective values with 'min' objectives negated, so higher is always better for every entry."""
    out = []
    for obj in objectives:
        v = _raw_objective_value(obj, stats, prop_summary, mc)
        direction = OBJECTIVE_DIRECTIONS.get(obj, "max")
        out.append(-v if direction == "min" else v)
    return tuple(out)


@dataclass
class MOCandidate:
    genome: list
    objective_values: tuple  # raw (non-negated) values, in objectives order -- for display
    _sort_values: tuple = field(repr=False, default=())  # negated-for-min values, used internally for dominance/crowding
    rank: int = 0
    crowding_distance: float = 0.0
    statistics: dict | None = None
    config: dict | None = None
    code_text: str | None = None
    code_extension: str | None = None
    feasible: bool = True  # False if the candidate produced zero trades


@dataclass
class MultiObjectiveConfig:
    objectives: list = field(default_factory=lambda: list(DEFAULT_OBJECTIVES))
    population_size: int = 20
    generations: int = 8
    mutation_rate: float = 0.35
    mutation_strength: float = 0.25
    random_immigrants_frac: float = 0.15
    search_monte_carlo_sims: int = 300
    random_seed: int | None = 42

    def __post_init__(self):
        for obj in self.objectives:
            if obj not in OBJECTIVE_DIRECTIONS:
                raise RefinementError(
                    f"Unknown objective '{obj}'. Supported objectives: {sorted(OBJECTIVE_DIRECTIONS)}"
                )
        if len(self.objectives) < 2:
            raise RefinementError("Multi-objective optimization requires at least 2 objectives.")
        self.population_size = max(int(self.population_size), 8)
        self.generations = max(int(self.generations), 1)


@dataclass
class MOGenerationSummary:
    generation: int
    front_0_size: int
    n_fronts: int


@dataclass
class MultiObjectiveResult:
    config: MultiObjectiveConfig
    source_type: str
    pareto_front: list  # list[MOCandidate], rank 0 of the FINAL population
    final_population: list  # list[MOCandidate], all ranks
    generation_history: list  # list[MOGenerationSummary]
    elapsed_seconds: float
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# NSGA-II core: fast non-dominated sort + crowding distance
# ---------------------------------------------------------------------------

def _dominates(a: tuple, b: tuple) -> bool:
    at_least_as_good = all(x >= y for x, y in zip(a, b))
    strictly_better = any(x > y for x, y in zip(a, b))
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population: list[MOCandidate]) -> list[list[int]]:
    n = len(population)
    dominated_by = [set() for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(population[p]._sort_values, population[q]._sort_values):
                dominated_by[p].add(q)
            elif _dominates(population[q]._sort_values, population[p]._sort_values):
                domination_count[p] += 1
        if domination_count[p] == 0:
            population[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def crowding_distance(population: list[MOCandidate], front: list[int]) -> None:
    if not front:
        return
    n_obj = len(population[front[0]]._sort_values)
    for idx in front:
        population[idx].crowding_distance = 0.0
    for m in range(n_obj):
        front_sorted = sorted(front, key=lambda idx: population[idx]._sort_values[m])
        population[front_sorted[0]].crowding_distance = float("inf")
        population[front_sorted[-1]].crowding_distance = float("inf")
        lo = population[front_sorted[0]]._sort_values[m]
        hi = population[front_sorted[-1]]._sort_values[m]
        span = (hi - lo) or 1.0
        for k in range(1, len(front_sorted) - 1):
            prev_v = population[front_sorted[k - 1]]._sort_values[m]
            next_v = population[front_sorted[k + 1]]._sort_values[m]
            population[front_sorted[k]].crowding_distance += (next_v - prev_v) / span


def _crowded_tournament(population: list[MOCandidate], rng: random.Random, k: int = 2) -> MOCandidate:
    contenders = rng.sample(population, min(k, len(population)))
    best = contenders[0]
    for c in contenders[1:]:
        if (c.rank, -c.crowding_distance) < (best.rank, -best.crowding_distance):
            best = c
    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_multi_objective_refinement(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    mo_config: MultiObjectiveConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> MultiObjectiveResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = mo_config or MultiObjectiveConfig()
    t0 = time.time()
    warnings: list[str] = []

    tmp_dir: Path | None = None
    if strategy.source_type == "python":
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_mo_"))

    try:
        preflight_signal_check(df, strategy, risk, "Multi-Objective search")

        genes, build = _build_adapter(strategy, tmp_dir)
        if not genes:
            raise RefinementError(
                "This strategy has no tunable numeric parameters -- multi-objective "
                "optimization has nothing to search over."
            )

        rng = random.Random(cfg.random_seed)
        search_mc_cfg = MonteCarloConfig(
            n_simulations=cfg.search_monte_carlo_sims,
            method=mc_config.method, block_size=mc_config.block_size,
            slippage_stress_pct=mc_config.slippage_stress_pct, random_seed=mc_config.random_seed,
        )

        def evaluate(genome: list) -> MOCandidate:
            candidate_strategy = build(genome)
            bt = run_backtest(df, candidate_strategy, risk)
            if not bt.trades:
                worst = tuple(-1e9 for _ in cfg.objectives)
                return MOCandidate(genome=list(genome), objective_values=tuple(0.0 for _ in cfg.objectives),
                                    _sort_values=worst, feasible=False, statistics=bt.statistics.to_dict())
            stats = bt.statistics.to_dict()
            pnls = [t.pnl for t in bt.trades]
            dates = [t.entry_time for t in bt.trades]
            single_run = simulate_account(pnls, dates, prop_rules)
            mc = run_monte_carlo(bt.trades, prop_rules, search_mc_cfg)
            prop_summary = summarize_single_run(single_run)
            sort_values = _objective_vector(cfg.objectives, stats, prop_summary, mc)
            raw_values = tuple(_raw_objective_value(o, stats, prop_summary, mc) for o in cfg.objectives)
            return MOCandidate(genome=list(genome), objective_values=raw_values, _sort_values=sort_values,
                                feasible=True, statistics=stats)

        log(f"Analyzing strategy parameters... found {len(genes)} tunable parameter(s), "
            f"optimizing for {cfg.objectives}.")

        population = [evaluate([g.base_value for g in genes])]
        while len(population) < cfg.population_size:
            population.append(evaluate([_random_gene_value(g, rng) for g in genes]))

        fronts = fast_non_dominated_sort(population)
        for f in fronts:
            crowding_distance(population, f)

        generation_history = [MOGenerationSummary(generation=0, front_0_size=len(fronts[0]), n_fronts=len(fronts))]
        log(f"Generation 0: front-0 size={len(fronts[0])}/{len(population)}, {len(fronts)} fronts total.")

        for gen in range(1, cfg.generations + 1):
            n_immigrants = max(1, round(cfg.population_size * cfg.random_immigrants_frac))
            n_bred = max(cfg.population_size - n_immigrants, 0)
            offspring: list[MOCandidate] = []
            for _ in range(n_bred):
                pa = _crowded_tournament(population, rng)
                pb = _crowded_tournament(population, rng)
                child_genome = _crossover(pa.genome, pb.genome, rng)
                child_genome = _mutate(child_genome, genes, cfg.mutation_rate, cfg.mutation_strength, rng)
                offspring.append(evaluate(child_genome))
            for _ in range(n_immigrants):
                offspring.append(evaluate([_random_gene_value(g, rng) for g in genes]))

            combined = population + offspring
            fronts = fast_non_dominated_sort(combined)
            for f in fronts:
                crowding_distance(combined, f)

            next_population: list[MOCandidate] = []
            for f in fronts:
                if len(next_population) + len(f) <= cfg.population_size:
                    next_population.extend(combined[i] for i in f)
                else:
                    remaining = cfg.population_size - len(next_population)
                    f_sorted = sorted(f, key=lambda i: combined[i].crowding_distance, reverse=True)
                    next_population.extend(combined[i] for i in f_sorted[:remaining])
                    break
            population = next_population

            fronts = fast_non_dominated_sort(population)
            for f in fronts:
                crowding_distance(population, f)

            generation_history.append(MOGenerationSummary(generation=gen, front_0_size=len(fronts[0]), n_fronts=len(fronts)))
            log(f"Generation {gen}/{cfg.generations}: front-0 size={len(fronts[0])}/{len(population)}, "
                f"{len(fronts)} fronts total.")

        pareto_front = [population[i] for i in fronts[0]]
        for c in pareto_front:
            if not c.feasible:
                warnings.append("At least one Pareto-front candidate produced zero trades; treat it as infeasible, not optimal.")
                break

        # Attach config/code snapshots only for the final front (keeps memory bounded).
        for c in pareto_front:
            candidate_strategy = build(c.genome)
            if strategy.source_type == "manual":
                from app.optimize.parameter_space import apply_genome
                c.config = apply_genome(strategy.config, genes, c.genome)
            else:
                from app.optimize.code_parameter_space import patched_source_for_strategy
                c.code_text, c.code_extension = patched_source_for_strategy(strategy, genes, c.genome)

        elapsed = time.time() - t0
        log(f"Multi-objective optimization complete in {elapsed:.1f}s. Pareto front size: {len(pareto_front)}.")

        return MultiObjectiveResult(
            config=cfg,
            source_type=strategy.source_type,
            pareto_front=pareto_front,
            final_population=population,
            generation_history=generation_history,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
