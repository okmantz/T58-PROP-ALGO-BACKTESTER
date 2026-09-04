"""
Iterative Refinement engine.

A small, dependency-free genetic algorithm that searches a strategy's
numeric parameter space for the configuration that scores best on a chosen
fitness metric, evaluated on the SAME historical dataset used for the
normal backtest. Works across all four strategy sources this app supports:

  Manual        -- every tunable numeric leaf in the config dict
                   (see app.optimize.parameter_space)
  Python        -- every top-level SCREAMING_SNAKE_CASE numeric constant
                   (see app.optimize.code_parameter_space)
  PineScript    -- every input.int()/input.float() default, plus the
                   T58_SL_PIPS/T58_TP_PIPS directives
  MQL5          -- every literal iMA()/iRSI() period, plus the same
                   T58_SL_PIPS/T58_TP_PIPS directives

Each "generation":
  1. keeps the top `elite_count` candidates from the current population unchanged
  2. breeds the rest via tournament-selected crossover + mutation
  3. injects a small fraction of fresh, fully-random candidates ("random
     immigrants") to keep the search from collapsing onto one local optimum
  4. re-evaluates the new population (backtest -> prop simulation ->
     a *cheap* Monte Carlo pass) and scores it

Across generations the population's parameters converge toward whatever
scores highest on the fitness metric -- the "genetic algorithm-like
optimization" Owen asked for. After the last generation, the single
best-ever candidate found across the whole search is re-evaluated once
more with the SAME Monte Carlo fidelity (n_simulations) as the main
pipeline, plus an out-of-sample holdout check, so the final report's
headline numbers are as trustworthy as the first report's.

IMPORTANT -- this is an in-sample search. It will always find *something*
that looks better on the exact historical window it was run against, even
if that improvement is pure noise. The holdout check and the prominent
overfitting-risk note in the generated report exist specifically to catch
that. Iterative Refinement should never be treated as proof a strategy
is better -- only as a faster way to generate candidates worth falsifying.
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.code_parameter_space import discover_code_genes, materialize_code_strategy, patched_source_for_strategy
from app.optimize.parameter_space import GeneMeta, RefinementError, apply_genome, extract_genome
from app.prop.simulator import AccountSimResult, PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy
from app.strategy.manual import ManualStrategy

ProgressCallback = Callable[[str], None]

CODE_SOURCE_TYPES = {"python", "pinescript", "mql5"}
SUPPORTED_SOURCE_TYPES = {"manual"} | CODE_SOURCE_TYPES

CODE_EXTENSIONS = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}

# Human-readable labels for the fitness-metric dropdown in the UI. Keys
# here are the exact strings accepted by RefinementConfig.fitness_metric.
#
# Most of these metrics (eval_pass_probability, prop_guide_score,
# composite_prop_score, first_payout_probability, expected_payout) are only
# meaningful if you're actually being scored against a prop firm's
# evaluation/funded-account rules (see app.prop.simulator.PropRules) -- they
# read PropRules and the Monte Carlo account simulation to score candidates.
#
# "net_profit" is the odd one out: compute_fitness() below reads it straight
# off the backtest's own statistics and never touches PropRules or the MC
# account simulation at all. That makes it the right choice for long-term /
# buy-and-hold-style retail trading where there is no prop firm evaluation to
# pass -- you just want the configuration that made the most money over the
# full test period, full stop.
FITNESS_METRICS: dict[str, str] = {
    "eval_pass_probability": "Eval Pass Probability -- reach target before hitting a limit (recommended for prop firms)",
    "prop_guide_score": "Prop-Oriented Guide Score",
    "composite_prop_score": "Composite Prop Score",
    "first_payout_probability": "First Payout Probability",
    "expected_payout": "Expected Payout ($)",
    "net_profit": "Long-Term Net Profit ($) -- for long-term trading, no prop firm",
    "profit_factor": "Profit Factor",
    "sharpe_ratio": "Sharpe Ratio",
}


@dataclass
class RefinementConfig:
    enabled: bool = False
    fitness_metric: str = "eval_pass_probability"
    population_size: int = 10
    generations: int = 5
    elite_count: int = 2
    mutation_rate: float = 0.35        # probability each gene mutates in a bred child
    mutation_strength: float = 0.25    # mutation step size, as a fraction of that gene's search range
    random_immigrants_frac: float = 0.15  # fraction of each new generation that is freshly randomized
    search_monte_carlo_sims: int = 500    # cheap MC used while searching, for speed
    random_seed: int | None = 42
    # Cost-stress: while ranking candidates, ALSO backtest each one at
    # spread_pips/slippage_pips/commission_per_trade multiplied by
    # cost_stress_multiplier, and blend that stressed-cost fitness into the
    # score the GA actually selects on (see _apply_cost_stress below). This
    # is what makes the search prefer strategies whose edge SURVIVES worse
    # execution, rather than strategies that only look good under the
    # default (fairly forgiving) cost assumptions -- without this, a
    # strategy that is pure curve-fit to optimistic fills scores identically
    # to a strategy with a real edge, right up until Stage 3's cost-ladder
    # check (which only reports, it doesn't feed back into what gets bred).
    cost_stress_enabled: bool = True
    cost_stress_multiplier: float = 2.0
    cost_stress_penalty_weight: float = 0.35   # 0 = ignore stress entirely, 1 = fully penalize any degradation

    def __post_init__(self):
        self.population_size = max(int(self.population_size), 4)
        self.generations = max(int(self.generations), 1)
        self.elite_count = max(1, min(int(self.elite_count), self.population_size - 1))
        self.mutation_rate = min(max(float(self.mutation_rate), 0.0), 1.0)
        self.mutation_strength = min(max(float(self.mutation_strength), 0.01), 1.0)
        self.random_immigrants_frac = min(max(float(self.random_immigrants_frac), 0.0), 0.9)
        self.search_monte_carlo_sims = max(int(self.search_monte_carlo_sims), 50)
        self.cost_stress_multiplier = max(float(self.cost_stress_multiplier), 1.0)
        self.cost_stress_penalty_weight = min(max(float(self.cost_stress_penalty_weight), 0.0), 1.0)
        if self.fitness_metric not in FITNESS_METRICS:
            raise RefinementError(f"Unknown fitness metric '{self.fitness_metric}'.")


@dataclass
class Candidate:
    generation: int
    genome: list
    fitness: float
    source_type: str = "manual"
    config: dict | None = None              # Manual Strategy config dict (manual only)
    code_text: str | None = None            # patched source text (python/pinescript/mql5 only)
    code_extension: str | None = None       # ".py" / ".pine" / ".mq5" (code strategies only)
    statistics: dict | None = None          # BacktestStatistics.to_dict()
    prop_summary: dict | None = None        # summarize_single_run(...)
    mc_summary: dict | None = None          # a few key Monte Carlo fields (cheap to keep for every candidate)
    # Full objects are only kept for the baseline and the final best
    # candidate (see _evaluate keep_full=) -- keeping them for every
    # candidate in every generation would mean holding population_size *
    # generations full equity curves in memory at once.
    bt_result: BacktestResult | None = None
    mc_result: MonteCarloResult | None = None
    single_run: AccountSimResult | None = None


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float   # average normalized population std-dev across genes; falls as the GA converges


@dataclass
class RefinementResult:
    refinement_config: RefinementConfig
    fitness_metric: str
    source_type: str
    genes: list  # list[GeneMeta] (manual) or list[CodeGene] (python/pinescript/mql5)
    baseline: Candidate
    best: Candidate
    generation_history: list  # list[GenerationSummary]
    leaderboard: list  # list[Candidate], final generation, sorted best-first
    holdout_comparison: dict | None
    elapsed_seconds: float
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _band(value: float, floor: float, ceiling: float) -> float:
    """0 at/below `floor`, ramps linearly to 1 at/above `ceiling`. Small
    helper for scoring a raw metric against one of the guide's target
    bands without repeating the same clamp-and-scale logic for each one."""
    if not math.isfinite(value):
        return 0.0
    if value <= floor:
        return 0.0
    if value >= ceiling:
        return 1.0
    return (value - floor) / (ceiling - floor)


def _prop_guide_score(stats: dict, mc: MonteCarloResult) -> float:
    """Scores a strategy against the PROP-ORIENTED STRATEGY GENERATION
    GUIDE's "ideal performance profile" (its section 4 target table)
    instead of collapsing everything to net profit, or even just eval-pass
    probability alone.

    Two strategies with identical eval-pass probability can still be very
    different bets to fund: one might clear the bar with a P95 drawdown
    that hugs the firm's actual limit, a handful of oversized trades, or
    too few trades to trust the number at all -- exactly the failure
    patterns the guide calls out by name in its "do not optimize for
    these alone" section. This blends the guide's own target bands (win
    rate 50-65%, profit factor >= 1.3, 100+ trades, P95 drawdown under
    roughly half of a typical prop firm's overall drawdown limit) into
    the pass/payout-probability objective the guide is actually chasing,
    so the GA can't win by exploiting the numbers this metric doesn't
    look at.

    Deliberately NOT normalized to a clean 0-1 range -- what matters for
    the GA/refinement's tournament selection is relative ordering between
    candidates, not the absolute scale."""
    eval_pass = mc.evaluation_pass_probability / 100.0
    payout = mc.first_payout_probability / 100.0
    ruin_penalty = mc.risk_of_ruin_pct / 100.0

    pf = stats.get("profit_factor", 0.0)
    pf_score = _band(pf if math.isfinite(pf) else 0.0, 1.0, 1.5)

    trade_score = _band(stats.get("total_trades", 0), 30, 150)

    win_rate = stats.get("win_rate", 0.0)
    if 50.0 <= win_rate <= 65.0:
        win_rate_score = 1.0
    elif win_rate < 50.0:
        win_rate_score = _band(win_rate, 30.0, 50.0)
    else:
        win_rate_score = max(0.0, 1.0 - (win_rate - 65.0) / 25.0)

    # P95 drawdown vs a typical ~10%-max-drawdown prop firm: the guide
    # wants P95 drawdown under roughly 50-60% of the firm's actual limit.
    # compute_fitness doesn't have the active PropRules in scope here, so
    # this uses that typical 10% figure as a stand-in rather than the
    # exact configured limit -- close enough to penalize a strategy that
    # runs uncomfortably close to ANY reasonable drawdown limit, without
    # needing to thread PropRules through every call site of this metric.
    dd_score = 1.0 - _band(mc.p95_drawdown_pct, 6.0, 10.0)

    return (
        eval_pass * 0.40
        + payout * 0.25
        + pf_score * 0.10
        + trade_score * 0.08
        + win_rate_score * 0.07
        + dd_score * 0.10
        - ruin_penalty * 0.15
    )


def compute_fitness(stats: dict, prop_summary: dict | None, mc: MonteCarloResult, metric: str) -> float:
    if metric == "net_profit":
        return float(stats.get("net_profit", 0.0))
    if metric == "profit_factor":
        pf = stats.get("profit_factor", 0.0)
        return 10.0 if pf == float("inf") else float(pf)
    if metric == "sharpe_ratio":
        return float(stats.get("sharpe_ratio", 0.0))
    if metric == "eval_pass_probability":
        return float(mc.evaluation_pass_probability)
    if metric == "first_payout_probability":
        return float(mc.first_payout_probability)
    if metric == "expected_payout":
        return float(mc.expected_payout)
    if metric == "composite_prop_score":
        return float(
            mc.evaluation_pass_probability * 0.5
            + mc.first_payout_probability * 0.3
            - mc.risk_of_ruin_pct * 0.2
        )
    if metric == "prop_guide_score":
        return _prop_guide_score(stats, mc)
    raise RefinementError(f"Unknown fitness metric '{metric}'.")


def _stressed_risk_config(risk: RiskConfig, multiplier: float) -> RiskConfig:
    """A copy of `risk` with every execution-cost assumption (spread,
    slippage, commission) scaled up by `multiplier` -- used to re-backtest
    a candidate under deliberately worse fills, never to change position
    sizing or account rules."""
    return replace(
        risk,
        spread_pips=risk.spread_pips * multiplier,
        slippage_pips=risk.slippage_pips * multiplier,
        commission_per_trade=risk.commission_per_trade * multiplier,
    )


def apply_cost_stress_penalty(nominal_fitness: float, stressed_fitness: float, weight: float) -> float:
    """
    Blends a stressed-cost fitness value into a nominal one, penalizing
    degradation without rewarding a stressed run that (by noise) scores
    slightly ABOVE nominal. Metric-agnostic: works the same whether
    `metric` is a raw dollar figure, a ratio, or a 0-1 probability, because
    the penalty is expressed as a FRACTION of the nominal score, not an
    absolute offset.

        weight=0.0 -> stress is ignored entirely (returns nominal_fitness)
        weight=1.0 -> full erosion under stress drives fitness to exactly 0

    A candidate that is already unprofitable/invalid at nominal cost
    (nominal_fitness <= 0, or non-finite) is returned unchanged -- there is
    no meaningful "how much of the edge survived" to measure once there
    was no edge at nominal cost either, and stressing it further would
    double-penalize a candidate Stage 1/the GA's own selection pressure
    already rejects on nominal grounds.
    """
    if weight <= 0 or not math.isfinite(nominal_fitness) or nominal_fitness <= 0:
        return nominal_fitness
    if not math.isfinite(stressed_fitness):
        degradation = 1.0
    else:
        degradation = max(0.0, (nominal_fitness - stressed_fitness) / abs(nominal_fitness))
        degradation = min(degradation, 1.0)
    return nominal_fitness - weight * degradation * nominal_fitness


def _mc_summary(mc: MonteCarloResult) -> dict:
    return {
        "evaluation_pass_probability": mc.evaluation_pass_probability,
        "first_payout_probability": mc.first_payout_probability,
        "failure_before_payout_probability": mc.failure_before_payout_probability,
        "expected_payout": mc.expected_payout,
        "risk_of_ruin_pct": mc.risk_of_ruin_pct,
        "median_drawdown_pct": mc.median_drawdown_pct,
        "n_simulations": mc.n_simulations,
    }


def _evaluate(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_cfg: MonteCarloConfig,
    metric: str,
    keep_full: bool = False,
    cost_stress_multiplier: float | None = None,
    cost_stress_penalty_weight: float = 0.0,
):
    """Runs one full backtest -> prop sim -> Monte Carlo pass for one strategy instance.

    When `cost_stress_multiplier` is given (and `cost_stress_penalty_weight` >
    0), ALSO re-runs the same strategy on the same data with spread/
    slippage/commission scaled up by that multiplier, and blends that
    stressed-cost result into the returned fitness via
    apply_cost_stress_penalty(). The returned `statistics`/`prop_summary`/
    `mc_summary` always describe the NOMINAL run (so reports keep showing
    real, un-stressed numbers) -- only the scalar fitness the GA selects on
    is cost-stress-adjusted.
    """
    bt_result = run_backtest(df, strategy, risk)

    if not bt_result.trades:
        return (
            float("-inf"),
            bt_result.statistics.to_dict(),
            None,
            None,
            bt_result if keep_full else None,
            None,
            None,
        )

    trade_pnls = [t.pnl for t in bt_result.trades]
    trade_dates = [t.entry_time for t in bt_result.trades]
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
    mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)
    prop_summary = summarize_single_run(single_run)

    fitness = compute_fitness(bt_result.statistics.to_dict(), prop_summary, mc_result, metric)
    if not math.isfinite(fitness):
        fitness = float("-inf")

    if cost_stress_multiplier and cost_stress_penalty_weight > 0 and math.isfinite(fitness):
        stressed_risk = _stressed_risk_config(risk, cost_stress_multiplier)
        stressed_bt = run_backtest(df, strategy, stressed_risk)
        if stressed_bt.trades:
            stressed_pnls = [t.pnl for t in stressed_bt.trades]
            stressed_dates = [t.entry_time for t in stressed_bt.trades]
            stressed_single_run = simulate_account(stressed_pnls, stressed_dates, prop_rules)
            stressed_mc = run_monte_carlo(stressed_bt.trades, prop_rules, mc_cfg)
            stressed_fitness = compute_fitness(
                stressed_bt.statistics.to_dict(), summarize_single_run(stressed_single_run), stressed_mc, metric,
            )
        else:
            stressed_fitness = float("-inf")
        fitness = apply_cost_stress_penalty(fitness, stressed_fitness, cost_stress_penalty_weight)

    return (
        fitness,
        bt_result.statistics.to_dict(),
        prop_summary,
        _mc_summary(mc_result),
        bt_result if keep_full else None,
        mc_result if keep_full else None,
        single_run if keep_full else None,
    )


# ---------------------------------------------------------------------------
# Genetic operators (source-type agnostic: only touch .lo/.hi/.is_int/.base_value,
# which GeneMeta and CodeGene both expose)
# ---------------------------------------------------------------------------

def _random_gene_value(gene, rng: random.Random) -> float:
    v = rng.uniform(gene.lo, gene.hi)
    return float(round(v)) if gene.is_int else float(v)


def _crossover(genome_a: list, genome_b: list, rng: random.Random) -> list:
    return [genome_a[i] if rng.random() < 0.5 else genome_b[i] for i in range(len(genome_a))]


def _mutate(genome: list, genes: list, rate: float, strength: float, rng: random.Random) -> list:
    out = list(genome)
    for i, gene in enumerate(genes):
        if rng.random() < rate:
            span = gene.hi - gene.lo
            delta = rng.uniform(-strength, strength) * span
            v = min(max(out[i] + delta, gene.lo), gene.hi)
            out[i] = float(round(v)) if gene.is_int else float(v)
    return out


def _tournament_select(population: list[Candidate], rng: random.Random, k: int = 3) -> Candidate:
    k = min(k, len(population))
    contenders = rng.sample(population, k)
    return max(contenders, key=lambda c: c.fitness)


def _diversity(population: list[Candidate], genes: list) -> float:
    if not genes or len(population) < 2:
        return 0.0
    total = 0.0
    for i, gene in enumerate(genes):
        span = (gene.hi - gene.lo) or 1.0
        vals = [c.genome[i] for c in population]
        mean_v = sum(vals) / len(vals)
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        total += (var ** 0.5) / span
    return total / len(genes)


def _summarize_generation(gen: int, population: list[Candidate], genes: list) -> GenerationSummary:
    finite = [c.fitness for c in population if math.isfinite(c.fitness)]
    best = max((c.fitness for c in population), default=float("-inf"))
    mean = (sum(finite) / len(finite)) if finite else float("-inf")
    worst = min(finite) if finite else float("-inf")
    return GenerationSummary(
        generation=gen, best_fitness=best, mean_fitness=mean, worst_fitness=worst,
        diversity=_diversity(population, genes),
    )


_NO_PARAMS_MESSAGE = {
    "manual": (
        "No tunable numeric parameters were found in this strategy configuration. "
        "Iterative Refinement optimizes Manual Strategy Builder parameters "
        "(indicator periods, comparison thresholds, stop loss / take profit / "
        "trailing stop / break-even values). Add at least one indicator-based "
        "condition, or a Fixed or ATR-based stop/target, and try again."
    ),
    "python": (
        "No tunable numeric parameters were found in this Python strategy. Iterative "
        "Refinement optimizes any top-level SCREAMING_SNAKE_CASE numeric constant "
        "(e.g. `EMA_FAST = 10`, `STOP_LOSS_PIPS = 20`) -- add at least one such "
        "constant and reference it inside generate_signals() to make it tunable."
    ),
    "pinescript": (
        "No tunable numeric parameters were found in this PineScript strategy. "
        "Iterative Refinement optimizes input.int()/input.float() values and the "
        "// T58_SL_PIPS= / // T58_TP_PIPS= directives -- add at least one of these."
    ),
    "mql5": (
        "No tunable numeric parameters were found in this MQL5 strategy. Iterative "
        "Refinement optimizes literal iMA()/iRSI() period arguments and the "
        "// T58_SL_PIPS= / // T58_TP_PIPS= directives -- add at least one of these."
    ),
}


def _build_adapter(strategy: Strategy, tmp_dir: Path | None):
    """
    Returns (genes, build_fn) where build_fn(genome) -> a fresh Strategy
    instance of the same source type as `strategy` with that genome applied.
    """
    source_type = strategy.source_type
    if source_type == "manual":
        genes = extract_genome(strategy.config)

        def build(genome: list):
            return ManualStrategy(apply_genome(strategy.config, genes, genome))

        return genes, build

    if source_type in CODE_SOURCE_TYPES:
        genes = discover_code_genes(strategy)

        def build(genome: list):
            return materialize_code_strategy(strategy, genes, genome, tmp_dir)

        return genes, build

    raise RefinementError(
        f"Iterative Refinement does not support strategy source type '{source_type}'."
    )


def preflight_signal_check(
    df: pd.DataFrame, strategy: Strategy, risk: RiskConfig, feature_name: str,
) -> None:
    """
    Runs ONE cheap, unmodified backtest of `strategy` on the FULL `df`
    before any fold-splitting or GA/NSGA-II search begins, and raises a
    clear RefinementError if it produces zero trades.

    Why this exists: Walk-Forward Optimization, Multi-Objective search,
    and the Walk-Forward-Aware GA all score every candidate (across every
    fold and every generation) the same way -- backtest it and read off
    stats. If the UNMODIFIED baseline strategy already produces zero
    trades on the WHOLE dataset, every single candidate downstream is
    guaranteed to also produce zero trades (folds are strict subsets of
    the same data, and no amount of numeric-parameter tuning fixes a
    strategy that structurally never fires on this data/timeframe). The
    old behavior was to grind through every fold and every generation
    anyway and hand back a report that's all zeros / -inf / "infeasible"
    with no explanation -- expensive AND confusing. This catches it in
    under a second, before any of that work starts.

    This is deliberately NOT run for Iterative Refinement's own baseline
    (run_iterative_refinement already computes and reports that baseline
    as part of its normal flow) -- only for the three heavier fold/
    population-based searches that would otherwise waste real time
    re-discovering the same "zero trades" fact many times over.
    """
    try:
        bt = run_backtest(df, strategy, risk)
    except Exception:
        # Let the caller's own error handling deal with a strategy that
        # can't even run once -- this check is only about "runs fine but
        # never fires," not about strategies that crash outright.
        return

    if bt.trades:
        return
    raise RefinementError(
        f"{feature_name} can't proceed: the strategy, unmodified, produced "
        f"ZERO trades on the entire dataset ({len(df)} bars, "
        f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}) before any "
        f"optimization even began. Every fold and every candidate downstream "
        f"would also score zero trades -- that's not a search-quality problem, "
        f"it's this strategy never firing on this data at all, so the search "
        f"was stopped instead of grinding through folds/generations for a "
        f"guaranteed-empty result.\n\n"
        f"Common causes, roughly in order of likelihood:\n"
        f"  - The strategy filters entries to specific hours-of-day (a London/"
        f"NY \"session\" window) but this data is daily bars or otherwise has "
        f"no real intraday hour information -- every bar's hour is constant, "
        f"so an hour-of-day filter excludes 100% of bars. Check the strategy "
        f"source for an hour/session filter if this data isn't intraday.\n"
        f"  - Not enough bars for the strategy's slowest indicator to warm up "
        f"(e.g. a 200-period moving average on a dataset with only a few "
        f"hundred bars).\n"
        f"  - The data's price scale, symbol, or timeframe doesn't match what "
        f"the strategy was written/tuned for.\n"
        f"  - The strategy's entry conditions are just very strict for this "
        f"particular instrument/period.\n\n"
        f"Try running a plain Run & Report (Step 5) on this same data/strategy "
        f"pair first -- if that also shows 0 trades, the fix is in the "
        f"strategy or the data pairing, not in this search."
    )



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_iterative_refinement(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    refinement_config: RefinementConfig,
    progress_cb: ProgressCallback | None = None,
) -> RefinementResult:
    """
    strategy: an already-built Strategy instance (ManualStrategy,
    PythonStrategy, PineScriptStrategy, or MQL5Strategy) -- e.g. whatever
    the UI/CLI already constructed for the normal run.
    mc_config: the SAME MonteCarloConfig used for the main pipeline run --
    its n_simulations is used for the baseline and the final best-candidate
    evaluation; the search phase uses refinement_config.search_monte_carlo_sims
    instead, for speed.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if strategy.source_type not in SUPPORTED_SOURCE_TYPES:
        raise RefinementError(
            f"Iterative Refinement does not support strategy source type "
            f"'{strategy.source_type}'."
        )

    t0 = time.time()
    cfg = refinement_config
    warnings: list[str] = []
    source_type = strategy.source_type

    tmp_dir: Path | None = None
    if source_type == "python":
        # PythonStrategy only accepts a file path, so mutated candidates are
        # written to throwaway temp .py files here, cleaned up in `finally`.
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_refine_"))

    try:
        genes, build = _build_adapter(strategy, tmp_dir)
        if not genes:
            raise RefinementError(_NO_PARAMS_MESSAGE.get(source_type, _NO_PARAMS_MESSAGE["manual"]))

        rng = random.Random(cfg.random_seed)
        search_mc_cfg = replace(mc_config, n_simulations=cfg.search_monte_carlo_sims)

        def snapshot(genome: list) -> tuple[dict | None, str | None, str | None]:
            """Only computed for the baseline and best-ever candidate (see keep_full)."""
            if source_type == "manual":
                return apply_genome(strategy.config, genes, genome), None, None
            code_text, ext = patched_source_for_strategy(strategy, genes, genome)
            return None, code_text, ext

        def evaluate(genome: list, generation: int, keep_full: bool = False) -> Candidate:
            candidate_strategy = build(genome)
            fitness, stats, prop_summary, mc_summary, bt_full, mc_full, single_full = _evaluate(
                df, candidate_strategy, risk, prop_rules, search_mc_cfg, cfg.fitness_metric, keep_full=keep_full,
                cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
                cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
            )
            config, code_text, code_ext = (None, None, None)
            if keep_full:
                config, code_text, code_ext = snapshot(genome)
            return Candidate(
                generation=generation, genome=list(genome), fitness=fitness, source_type=source_type,
                config=config, code_text=code_text, code_extension=code_ext,
                statistics=stats, prop_summary=prop_summary, mc_summary=mc_summary,
                bt_result=bt_full, mc_result=mc_full, single_run=single_full,
            )

        log(f"Analyzing strategy parameters... found {len(genes)} tunable parameter(s).")

        baseline_genome = [g.base_value for g in genes]
        baseline = evaluate(baseline_genome, 0, keep_full=True)
        if not math.isfinite(baseline.fitness):
            warnings.append(
                "The current (baseline) configuration produced no trades on this data "
                "-- there is no known-good baseline to compare the search against."
            )
        log(f"Baseline fitness ({FITNESS_METRICS[cfg.fitness_metric]}): {baseline.fitness:.3f}")

        population: list[Candidate] = [baseline]
        while len(population) < cfg.population_size:
            population.append(evaluate([_random_gene_value(g, rng) for g in genes], 0))

        best_ever = max(population, key=lambda c: c.fitness)
        generation_history: list[GenerationSummary] = [_summarize_generation(0, population, genes)]
        log(
            f"Generation 0 (initial population of {cfg.population_size}): "
            f"best={generation_history[0].best_fitness:.3f}  mean={generation_history[0].mean_fitness:.3f}"
        )

        for gen in range(1, cfg.generations + 1):
            population.sort(key=lambda c: c.fitness, reverse=True)
            elites = population[: cfg.elite_count]
            next_pop: list[Candidate] = list(elites)

            n_immigrants = max(1, round(cfg.population_size * cfg.random_immigrants_frac))
            n_bred = max(cfg.population_size - len(elites) - n_immigrants, 0)

            for _ in range(n_bred):
                parent_a = _tournament_select(population, rng)
                parent_b = _tournament_select(population, rng)
                child_genome = _crossover(parent_a.genome, parent_b.genome, rng)
                child_genome = _mutate(child_genome, genes, cfg.mutation_rate, cfg.mutation_strength, rng)
                next_pop.append(evaluate(child_genome, gen))

            while len(next_pop) < cfg.population_size:
                next_pop.append(evaluate([_random_gene_value(g, rng) for g in genes], gen))

            population = next_pop
            gen_summary = _summarize_generation(gen, population, genes)
            generation_history.append(gen_summary)

            gen_best = max(population, key=lambda c: c.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best

            log(
                f"Generation {gen}/{cfg.generations}: best={gen_summary.best_fitness:.3f}  "
                f"mean={gen_summary.mean_fitness:.3f}  diversity={gen_summary.diversity:.3f}  "
                f"(best-ever={best_ever.fitness:.3f})"
            )

        log("Running full-fidelity Monte Carlo on the best-ever configuration...")
        best_strategy = build(best_ever.genome)
        final_fitness, final_stats, final_prop, final_mc_summary, final_bt, final_mc, final_single = _evaluate(
            df, best_strategy, risk, prop_rules, mc_config, cfg.fitness_metric, keep_full=True,
            cost_stress_multiplier=cfg.cost_stress_multiplier if cfg.cost_stress_enabled else None,
            cost_stress_penalty_weight=cfg.cost_stress_penalty_weight if cfg.cost_stress_enabled else 0.0,
        )
        final_config, final_code_text, final_code_ext = snapshot(best_ever.genome)
        best_final = Candidate(
            generation=best_ever.generation, genome=best_ever.genome, fitness=final_fitness,
            source_type=source_type, config=final_config, code_text=final_code_text, code_extension=final_code_ext,
            statistics=final_stats, prop_summary=final_prop, mc_summary=final_mc_summary,
            bt_result=final_bt, mc_result=final_mc, single_run=final_single,
        )

        holdout_comparison = None
        if final_bt is not None and final_bt.trades:
            log("Running out-of-sample holdout check on the optimized configuration...")
            try:
                holdout_comparison = run_holdout_comparison(
                    df, build(best_ever.genome), risk, holdout_frac=0.2,
                )
            except Exception:
                warnings.append(
                    "Holdout check on the optimized configuration could not be "
                    "completed (not enough data to split)."
                )

        leaderboard = sorted(population, key=lambda c: c.fitness, reverse=True)
        elapsed = time.time() - t0
        log(f"Iterative Refinement complete in {elapsed:.1f}s.")

        return RefinementResult(
            refinement_config=cfg,
            fitness_metric=cfg.fitness_metric,
            source_type=source_type,
            genes=genes,
            baseline=baseline,
            best=best_final,
            generation_history=generation_history,
            leaderboard=leaderboard,
            holdout_comparison=holdout_comparison,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
