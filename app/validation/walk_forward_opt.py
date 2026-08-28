"""
Walk-Forward Optimization (first-class workflow).

app.backtest.engine.run_holdout_comparison() and app.search.robustness.
run_walk_forward() both already answer "does this exact, already-chosen
strategy configuration keep working across time?" -- neither one
re-tunes anything. This module answers a different, harder question:
"if I re-optimize on each rolling/anchored in-sample window the way I
actually would in practice, and only ever trade the following
out-of-sample window with whatever that optimization picked, what does
my real equity curve look like?"

That is the classic definition of walk-forward optimization/analysis:

    [-------- train 1 --------][-- test 1 --]
              [-------- train 2 --------][-- test 2 --]
                        [-------- train 3 --------][-- test 3 --]
                                  ...

Each fold's optimizer never sees that fold's own test window. The test
windows are then stitched together in chronological order into ONE
continuous out-of-sample equity curve -- this is the number that matters,
because it is the closest thing to "what would have actually happened if
I had been running this process live," as opposed to a single lucky (or
unlucky) in-sample fit.

Two window modes:
  "rolling"  -- each fold's train window is a fixed-size slice that
                slides forward (older data ages out). Better suited to
                a strategy/instrument you believe has a regime-dependent
                edge.
  "anchored" -- each fold's train window starts at bar 0 and grows
                (nothing ages out). Better suited to a strategy you
                believe has a stable, non-regime-dependent edge, so more
                data is strictly better.

The per-fold optimizer reuses the exact same genetic-algorithm machinery
Iterative Refinement (app.optimize.refinement) already uses -- same
GeneMeta/CodeGene discovery, same crossover/mutation/tournament
selection -- just re-run fresh on each fold's train window, with a
smaller population/generation count by default since it now runs once
per fold.
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

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest
from app.backtest.execution import Trade
from app.backtest.risk import RiskConfig
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import (
    Candidate,
    RefinementConfig,
    _build_adapter,
    _crossover,
    _mutate,
    _random_gene_value,
    _tournament_select,
    compute_fitness,
)
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy

ProgressCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# Fold splitting -- shared by walk_forward_opt.py and optimize/walkforward_ga.py
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    fold_index: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_period: tuple
    test_period: tuple


def build_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    window_mode: str = "rolling",
    train_frac: float = 0.6,
    embargo_bars: int = 0,
) -> list[Fold]:
    """
    Splits `df` chronologically into `n_folds` (train, test) windows.

    train_frac: for "rolling" mode, the fraction of each fold's total
    window that is training data (the rest is that fold's test slice).
    For "anchored" mode, train_frac instead sets the SIZE of the first
    fold's test slice as a fraction of the total remaining data per fold
    (the train window itself always starts at bar 0 and grows).

    embargo_bars: number of bars dropped from the START of each test
    slice (purged) to reduce contamination from indicator warm-up state
    computed at the train/test boundary (e.g. a slow moving average
    whose value at the boundary still depends on train-window bars).
    """
    n = len(df)
    if n_folds < 1 or n < (n_folds + 1) * 20:
        return []

    fold_span = n // (n_folds + 1)
    if fold_span < 10:
        return []

    folds: list[Fold] = []
    for i in range(n_folds):
        if window_mode == "anchored":
            train_end = fold_span * (i + 1)
            test_start = train_end
        else:  # rolling
            window_end = fold_span * (i + 2)
            train_end = int(window_end * train_frac)
            train_end = max(train_end, fold_span // 2)
            test_start = train_end
            window_start = max(0, window_end - fold_span * 2)
            train_start = max(window_start, 0)

        test_start = min(test_start + embargo_bars, n)
        test_end = min(fold_span * (i + 2), n)

        if window_mode == "anchored":
            train_slice = df.iloc[0:train_end].reset_index(drop=True)
        else:
            train_slice = df.iloc[train_start:train_end].reset_index(drop=True)
        test_slice = df.iloc[test_start:test_end].reset_index(drop=True)

        if len(train_slice) < 10 or len(test_slice) < 5:
            continue

        folds.append(Fold(
            fold_index=i,
            train_df=train_slice,
            test_df=test_slice,
            train_period=(str(train_slice["timestamp"].iloc[0]), str(train_slice["timestamp"].iloc[-1])),
            test_period=(str(test_slice["timestamp"].iloc[0]), str(test_slice["timestamp"].iloc[-1])),
        ))
    return folds


# ---------------------------------------------------------------------------
# Per-fold optimization (a small, fresh GA run per fold's train window)
# ---------------------------------------------------------------------------

def _optimize_on_window(
    train_df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    refine_cfg: RefinementConfig,
    tmp_dir: Path | None,
    rng: random.Random,
) -> tuple[list, list[float] | None, str | None]:
    """
    Runs a self-contained GA search on `train_df` only. Returns
    (best_genome, genes_as_plain_list_or_None, warning_or_None).
    genes is None (best_genome == []) when the strategy has no tunable
    numeric parameters -- callers should then just use the strategy as-is
    for every fold's test window (still a valid, if degenerate, walk-
    forward run: nothing to re-optimize, so the OOS chain is really just
    a walk-forward HOLDOUT chain for a fixed strategy).
    """
    try:
        genes, build = _build_adapter(strategy, tmp_dir)
    except RefinementError as exc:
        return [], None, str(exc)
    if not genes:
        return [], None, None

    def evaluate(genome: list) -> float:
        candidate_strategy = build(genome)
        bt = run_backtest(train_df, candidate_strategy, risk)
        if not bt.trades:
            return float("-inf")
        pnls = [t.pnl for t in bt.trades]
        dates = [t.entry_time for t in bt.trades]
        single_run = simulate_account(pnls, dates, prop_rules)
        from app.monte_carlo.engine import run_monte_carlo
        mc = run_monte_carlo(bt.trades, prop_rules, replace(mc_config, n_simulations=refine_cfg.search_monte_carlo_sims))
        prop_summary = summarize_single_run(single_run)
        fitness = compute_fitness(bt.statistics.to_dict(), prop_summary, mc, refine_cfg.fitness_metric)
        return fitness if math.isfinite(fitness) else float("-inf")

    baseline_genome = [g.base_value for g in genes]
    population = [(baseline_genome, evaluate(baseline_genome))]
    while len(population) < refine_cfg.population_size:
        g = [_random_gene_value(gene, rng) for gene in genes]
        population.append((g, evaluate(g)))

    best_genome, best_fitness = max(population, key=lambda p: p[1])

    for _gen in range(refine_cfg.generations):
        population.sort(key=lambda p: p[1], reverse=True)
        elites = population[: refine_cfg.elite_count]
        next_pop = list(elites)
        n_immigrants = max(1, round(refine_cfg.population_size * refine_cfg.random_immigrants_frac))
        n_bred = max(refine_cfg.population_size - len(elites) - n_immigrants, 0)

        class _Cand:  # minimal shim so we can reuse refinement._tournament_select unmodified
            __slots__ = ("genome", "fitness")

            def __init__(self, genome, fitness):
                self.genome = genome
                self.fitness = fitness

        cand_pop = [_Cand(g, f) for g, f in population]
        for _ in range(n_bred):
            pa = _tournament_select(cand_pop, rng)
            pb = _tournament_select(cand_pop, rng)
            child = _crossover(pa.genome, pb.genome, rng)
            child = _mutate(child, genes, refine_cfg.mutation_rate, refine_cfg.mutation_strength, rng)
            next_pop.append((child, evaluate(child)))
        while len(next_pop) < refine_cfg.population_size:
            g = [_random_gene_value(gene, rng) for gene in genes]
            next_pop.append((g, evaluate(g)))

        population = next_pop
        gen_best_genome, gen_best_fitness = max(population, key=lambda p: p[1])
        if gen_best_fitness > best_fitness:
            best_genome, best_fitness = gen_best_genome, gen_best_fitness

    return best_genome, [g.base_value for g in genes], None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardOptFold:
    fold_index: int
    train_period: tuple
    test_period: tuple
    train_bars: int
    test_bars: int
    train_fitness: float
    best_genome: list
    test_statistics: dict | None
    test_trade_count: int


@dataclass
class WalkForwardOptResult:
    window_mode: str
    n_folds_requested: int
    folds: list  # list[WalkForwardOptFold]
    combined_trades: list  # list[Trade], chronological, chained across all folds' OOS windows
    combined_equity_curve: pd.DataFrame
    combined_statistics: BacktestStatistics
    in_sample_reference_fitness: float | None  # fitness of ONE GA run on the whole df, for comparison
    out_of_sample_efficiency: float | None      # mean per-fold OOS fitness proxy / in_sample_reference_fitness
    elapsed_seconds: float
    warnings: list = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        return {
            "window_mode": self.window_mode,
            "n_folds_completed": len(self.folds),
            "combined_statistics": self.combined_statistics.to_dict(),
            "out_of_sample_efficiency": self.out_of_sample_efficiency,
            "in_sample_reference_fitness": self.in_sample_reference_fitness,
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "train_period": f.train_period,
                    "test_period": f.test_period,
                    "train_bars": f.train_bars,
                    "test_bars": f.test_bars,
                    "train_fitness": f.train_fitness,
                    "test_trade_count": f.test_trade_count,
                    "test_net_profit": (f.test_statistics or {}).get("net_profit"),
                    "test_profit_factor": (f.test_statistics or {}).get("profit_factor"),
                }
                for f in self.folds
            ],
            "warnings": self.warnings,
        }


def _rebuild_equity_curve(trades: list[Trade], initial_balance: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame({"timestamp": [], "equity": []})
    ordered = sorted(trades, key=lambda t: t.exit_time)
    equity = initial_balance
    rows = [{"timestamp": ordered[0].entry_time, "equity": initial_balance}]
    for t in ordered:
        equity += t.pnl if math.isfinite(t.pnl) else 0.0
        rows.append({"timestamp": t.exit_time, "equity": equity})
    return pd.DataFrame(rows)


def run_walk_forward_optimization(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    n_folds: int = 5,
    window_mode: str = "rolling",
    train_frac: float = 0.6,
    embargo_bars: int = 0,
    refine_cfg: RefinementConfig | None = None,
    random_seed: int | None = 42,
    progress_cb: ProgressCallback | None = None,
) -> WalkForwardOptResult:
    """
    Runs a fresh, small GA search on each fold's train window, applies the
    winning configuration UNCHANGED to that fold's test window, and chains
    every fold's test-window trades into one continuous out-of-sample
    equity curve + statistics set. This is the number to trust over a
    single in-sample backtest or a single 80/20 holdout split.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    t0 = time.time()
    warnings: list[str] = []
    refine_cfg = refine_cfg or RefinementConfig(population_size=8, generations=3, search_monte_carlo_sims=200)
    rng = random.Random(random_seed)

    raw_folds = build_folds(df, n_folds=n_folds, window_mode=window_mode, train_frac=train_frac, embargo_bars=embargo_bars)
    if not raw_folds:
        raise RefinementError(
            "Not enough bars to build the requested number of walk-forward folds. "
            "Use fewer folds or a larger dataset."
        )

    tmp_dir: Path | None = None
    if strategy.source_type == "python":
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_wfo_"))

    fold_results: list[WalkForwardOptFold] = []
    all_test_trades: list[Trade] = []
    per_fold_fitness: list[float] = []

    try:
        _, build_for_result = _build_adapter(strategy, tmp_dir) if strategy.source_type in (
            "manual", "python", "pinescript", "mql5"
        ) else (None, None)

        for fold in raw_folds:
            log(f"Fold {fold.fold_index + 1}/{len(raw_folds)}: optimizing on {len(fold.train_df)} train bars...")
            best_genome, gene_base_values, warn = _optimize_on_window(
                fold.train_df, strategy, risk, prop_rules, mc_config, refine_cfg, tmp_dir, rng,
            )
            if warn:
                warnings.append(f"Fold {fold.fold_index}: {warn}")

            if gene_base_values is None:
                # No tunable parameters -- just run the strategy as-is on the test window.
                test_strategy = strategy
                train_fitness = float("nan")
            else:
                genes, build = _build_adapter(strategy, tmp_dir)
                test_strategy = build(best_genome)
                # Re-derive the train fitness for reporting (cheap MC).
                train_bt = run_backtest(fold.train_df, test_strategy, risk)
                if train_bt.trades:
                    pnls = [t.pnl for t in train_bt.trades]
                    dates = [t.entry_time for t in train_bt.trades]
                    single_run = simulate_account(pnls, dates, prop_rules)
                    from app.monte_carlo.engine import run_monte_carlo
                    mc = run_monte_carlo(train_bt.trades, prop_rules, replace(mc_config, n_simulations=refine_cfg.search_monte_carlo_sims))
                    train_fitness = compute_fitness(train_bt.statistics.to_dict(), summarize_single_run(single_run), mc, refine_cfg.fitness_metric)
                else:
                    train_fitness = float("-inf")

            test_bt = run_backtest(fold.test_df, test_strategy, risk)
            per_fold_fitness.append(train_fitness if math.isfinite(train_fitness) else 0.0)
            all_test_trades.extend(test_bt.trades)

            fold_results.append(WalkForwardOptFold(
                fold_index=fold.fold_index,
                train_period=fold.train_period,
                test_period=fold.test_period,
                train_bars=len(fold.train_df),
                test_bars=len(fold.test_df),
                train_fitness=train_fitness,
                best_genome=best_genome,
                test_statistics=test_bt.statistics.to_dict(),
                test_trade_count=len(test_bt.trades),
            ))
            log(
                f"  train_fitness={train_fitness:.3f}  "
                f"test_trades={len(test_bt.trades)}  test_net_profit=${test_bt.statistics.net_profit:,.2f}"
            )

        combined_equity = _rebuild_equity_curve(all_test_trades, risk.initial_balance)
        combined_stats = compute_statistics(all_test_trades, combined_equity, initial_balance=risk.initial_balance) \
            if len(combined_equity) else compute_statistics([], pd.DataFrame({"timestamp": [], "equity": []}), risk.initial_balance)

        # In-sample reference: one GA run on the FULL dataset, for an
        # apples-to-apples "in-sample vs chained-OOS" efficiency ratio.
        in_sample_fitness = None
        try:
            full_genome, full_gene_bases, _w = _optimize_on_window(
                df, strategy, risk, prop_rules, mc_config, refine_cfg, tmp_dir, random.Random(random_seed),
            )
            if full_gene_bases is not None:
                genes, build = _build_adapter(strategy, tmp_dir)
                full_strategy = build(full_genome)
                full_bt = run_backtest(df, full_strategy, risk)
                if full_bt.trades:
                    pnls = [t.pnl for t in full_bt.trades]
                    dates = [t.entry_time for t in full_bt.trades]
                    single_run = simulate_account(pnls, dates, prop_rules)
                    from app.monte_carlo.engine import run_monte_carlo
                    mc = run_monte_carlo(full_bt.trades, prop_rules, replace(mc_config, n_simulations=refine_cfg.search_monte_carlo_sims))
                    in_sample_fitness = compute_fitness(full_bt.statistics.to_dict(), summarize_single_run(single_run), mc, refine_cfg.fitness_metric)
        except Exception:
            in_sample_fitness = None

        mean_oos_fitness = float(np.mean(per_fold_fitness)) if per_fold_fitness else None
        oos_efficiency = None
        if in_sample_fitness and math.isfinite(in_sample_fitness) and in_sample_fitness != 0 and mean_oos_fitness is not None:
            oos_efficiency = max(min(mean_oos_fitness / in_sample_fitness, 5.0), -5.0)

        elapsed = time.time() - t0
        log(f"Walk-forward optimization complete in {elapsed:.1f}s across {len(fold_results)} fold(s).")

        return WalkForwardOptResult(
            window_mode=window_mode,
            n_folds_requested=n_folds,
            folds=fold_results,
            combined_trades=all_test_trades,
            combined_equity_curve=combined_equity,
            combined_statistics=combined_stats,
            in_sample_reference_fitness=in_sample_fitness,
            out_of_sample_efficiency=oos_efficiency,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
