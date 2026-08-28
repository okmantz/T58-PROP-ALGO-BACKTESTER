"""
Parameter sensitivity sweeps and heatmaps.

app.search.robustness.parameter_neighborhood_robustness() already answers
"are nearby parameter values roughly as good?" with a single scalar
stability ratio -- useful for an automated pass/fail gate, but it doesn't
show WHERE a strategy's edge lives or WHAT the drop-off looks like. This
module produces the actual curves/grids so a person can look at them:

  compute_1d_sensitivity() -- for a chosen parameter, evaluate the
      strategy at N evenly-spaced values across +/- pct_range of its
      current value (clipped to that parameter's normal search bounds),
      holding every other parameter fixed. Flags a "cliff" wherever the
      metric drops sharply between adjacent steps, vs. a "plateau" where
      it degrades gradually.

  compute_2d_heatmap() -- the same idea for a PAIR of parameters at once,
      producing a 2D grid suitable for a heatmap chart. This is the more
      informative view when two parameters interact (e.g. a fast/slow
      moving-average pair), since a 1D sweep of either one alone can look
      robust while the pair together sits on a narrow ridge.

Works across all source types (manual/python/pinescript/mql5) via the
same gene discovery Iterative Refinement already uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import _build_adapter, compute_fitness
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.strategy.base import Strategy


def _metric_for(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str,
) -> float:
    bt = run_backtest(df, strategy, risk)
    if not bt.trades:
        return float("-inf")
    if metric in ("net_profit", "profit_factor", "sharpe_ratio", "win_rate", "expectancy", "max_drawdown_pct"):
        stats = bt.statistics.to_dict()
        v = stats.get(metric, 0.0)
        if v == float("inf"):
            return 10.0
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
    # Anything else (composite_prop_score, eval_pass_probability, ...) needs the full prop+MC pipeline.
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(pnls, dates, prop_rules)
    mc = run_monte_carlo(bt.trades, prop_rules, mc_config)
    prop_summary = summarize_single_run(single_run)
    fitness = compute_fitness(stats := bt.statistics.to_dict(), prop_summary, mc, metric)
    return fitness if math.isfinite(fitness) else float("-inf")


@dataclass
class Sensitivity1DResult:
    gene_label: str
    base_value: float
    base_metric: float
    values: list
    metric_values: list
    metric: str
    max_pct_drop_between_adjacent_steps: float
    cliff_detected: bool
    cliff_threshold: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _sweep_values(base_value: float, lo: float, hi: float, is_int: bool, pct_range: float, n_steps: int) -> list[float]:
    span = max(abs(base_value) * pct_range, (hi - lo) * 0.02 if hi > lo else 1.0)
    sweep_lo = max(base_value - span, lo)
    sweep_hi = min(base_value + span, hi)
    if sweep_hi <= sweep_lo:
        sweep_lo, sweep_hi = lo, hi
    raw = np.linspace(sweep_lo, sweep_hi, max(n_steps, 3))
    if is_int:
        raw = sorted(set(int(round(v)) for v in raw))
        return [float(v) for v in raw]
    return [float(v) for v in raw]


def compute_1d_sensitivity(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str = "profit_factor",
    pct_range: float = 0.5,
    n_steps: int = 9,
    max_params: int = 8,
    cliff_threshold_pct: float = 40.0,
    tmp_dir: Path | None = None,
) -> list[Sensitivity1DResult]:
    """
    Sweeps every tunable numeric parameter (up to max_params, in discovery
    order) independently across +/- pct_range of its current value,
    holding all other parameters at their base value.
    """
    genes, build = _build_adapter(strategy, tmp_dir)
    if not genes:
        raise RefinementError(
            "This strategy has no tunable numeric parameters to run a sensitivity sweep on."
        )

    results: list[Sensitivity1DResult] = []
    for gi, gene in enumerate(genes[:max_params]):
        values = _sweep_values(gene.base_value, gene.lo, gene.hi, gene.is_int, pct_range, n_steps)
        metric_values = []
        for v in values:
            genome = [g.base_value for g in genes]
            genome[gi] = v
            candidate = build(genome)
            metric_values.append(_metric_for(df, candidate, risk, prop_rules, mc_config, metric))

        base_metric = _metric_for(df, build([g.base_value for g in genes]), risk, prop_rules, mc_config, metric)

        finite_vals = [m for m in metric_values if math.isfinite(m)]
        max_drop_pct = 0.0
        for a, b in zip(metric_values, metric_values[1:]):
            if math.isfinite(a) and math.isfinite(b) and a > 0:
                drop = (a - b) / abs(a) * 100.0
                max_drop_pct = max(max_drop_pct, drop)
            elif math.isfinite(a) and not math.isfinite(b):
                max_drop_pct = max(max_drop_pct, 100.0)

        results.append(Sensitivity1DResult(
            gene_label=gene.label,
            base_value=gene.base_value,
            base_metric=base_metric,
            values=values,
            metric_values=metric_values,
            metric=metric,
            max_pct_drop_between_adjacent_steps=max_drop_pct,
            cliff_detected=max_drop_pct >= cliff_threshold_pct,
            cliff_threshold=cliff_threshold_pct,
        ))
    return results


@dataclass
class Sensitivity2DResult:
    gene_a_label: str
    gene_b_label: str
    a_values: list
    b_values: list
    grid: list  # list[list[float]], grid[i][j] = metric at (a_values[i], b_values[j])
    base_metric: float
    metric: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compute_2d_heatmap(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    gene_label_a: str,
    gene_label_b: str,
    metric: str = "profit_factor",
    pct_range: float = 0.5,
    n_steps: int = 7,
    tmp_dir: Path | None = None,
) -> Sensitivity2DResult:
    genes, build = _build_adapter(strategy, tmp_dir)
    if not genes:
        raise RefinementError("This strategy has no tunable numeric parameters.")

    label_to_idx = {g.label: i for i, g in enumerate(genes)}
    if gene_label_a not in label_to_idx or gene_label_b not in label_to_idx:
        raise RefinementError(
            f"Unknown parameter label(s). Available: {sorted(label_to_idx)}"
        )
    ia, ib = label_to_idx[gene_label_a], label_to_idx[gene_label_b]
    ga, gb = genes[ia], genes[ib]

    a_values = _sweep_values(ga.base_value, ga.lo, ga.hi, ga.is_int, pct_range, n_steps)
    b_values = _sweep_values(gb.base_value, gb.lo, gb.hi, gb.is_int, pct_range, n_steps)

    base_genome = [g.base_value for g in genes]
    base_metric = _metric_for(df, build(base_genome), risk, prop_rules, mc_config, metric)

    grid: list[list[float]] = []
    for av in a_values:
        row = []
        for bv in b_values:
            genome = list(base_genome)
            genome[ia] = av
            genome[ib] = bv
            row.append(_metric_for(df, build(genome), risk, prop_rules, mc_config, metric))
        grid.append(row)

    return Sensitivity2DResult(
        gene_a_label=gene_label_a,
        gene_b_label=gene_label_b,
        a_values=a_values,
        b_values=b_values,
        grid=grid,
        base_metric=base_metric,
        metric=metric,
    )


def list_tunable_parameters(strategy: Strategy, tmp_dir: Path | None = None) -> list[str]:
    """Convenience helper for a UI/CLI to populate a parameter picker."""
    genes, _build = _build_adapter(strategy, tmp_dir)
    return [g.label for g in genes]
