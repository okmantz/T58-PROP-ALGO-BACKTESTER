"""
Parameter Robustness Score + heatmap.

The GA in this app (app.optimize.refinement / app.optimize.walkforward_ga)
searches parameter space aggressively, which is exactly what makes one
question unavoidable: did it find a GOOD parameter, or a LUCKY one?

    EMA 36 -> terrible
    EMA 37 -> amazing      <- the GA's "winner"
    EMA 38 -> terrible

That shape is an optimization cliff: the result is fit to noise in this
exact historical window, not to a tradeable effect, and it will not
survive contact with one more day of real data. What a real edge looks
like instead:

    EMA 30 -> good
    EMA 34 -> great
    EMA 37 -> great        <- the GA's "winner" -- STILL great, but so is
    EMA 40 -> good            everything else nearby. That's a plateau.

app.validation.sensitivity already computes the raw curves and grids that
show this (compute_1d_sensitivity for a single parameter's sweep,
compute_2d_heatmap for a pair). This module is the layer on top that a
person actually wants to look at: it turns those curves/grids into ONE
Parameter Robustness Score (0-100), flags which specific parameters sit
on a cliff, and formats the 2D grid as a plain "param A / param B /
Pass %" table -- ready to render as-is, no reshaping required.

Deliberately reuses compute_1d_sensitivity / compute_2d_heatmap /
list_tunable_parameters UNMODIFIED rather than recomputing sweeps a
second way -- this module is pure interpretation of numbers those
functions already produce, so a strategy's sensitivity curves and its
robustness score can never silently disagree with each other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.prop.simulator import PropRules
from app.strategy.base import Strategy
from app.validation.sensitivity import (
    Sensitivity1DResult, Sensitivity2DResult, compute_1d_sensitivity, compute_2d_heatmap,
    list_tunable_parameters,
)

# eval_pass_probability (and first_payout_probability) are already
# genuine 0-100 PASS RATES straight out of the Monte Carlo engine, which
# is exactly the "Pass %" framing the product spec's own example table
# uses -- unlike profit_factor/sharpe/net_profit, which are the RIGHT
# default for app.validation.sensitivity's more general-purpose sweeps
# but aren't naturally a "pass %" at all. Override `metric=` for a
# different lens (e.g. "first_payout_probability" to score robustness of
# getting PAID, not just of passing the evaluation).
DEFAULT_PASS_METRIC = "eval_pass_probability"
DEFAULT_PASS_THRESHOLD_PCT = 50.0
CLIFF_SCORE_PENALTY = 20.0


@dataclass
class ParameterHeatmapRow:
    """One flattened row of a 2D heatmap -- the exact 'SL / TP / Pass %'
    table shape from the product spec, so a UI can render a plain table
    directly with zero reshaping (in addition to, or instead of, a chart
    widget consuming the 2D grid on ParameterPairRobustness.heatmap)."""
    param_a_label: str
    param_a_value: float
    param_b_label: str
    param_b_value: float
    pass_pct: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ParameterPairRobustness:
    heatmap: Sensitivity2DResult
    rows: list
    pass_threshold_pct: float
    fraction_of_grid_passing: float   # 0-100
    worst_cell_pass_pct: float
    best_cell_pass_pct: float
    is_plateau: bool                  # the (approximate) base cell AND every immediate neighbor clear the threshold
    is_cliff: bool                    # the base cell clears the threshold but at least one immediate neighbor doesn't

    def to_dict(self) -> dict:
        return {
            "heatmap": self.heatmap.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "pass_threshold_pct": self.pass_threshold_pct,
            "fraction_of_grid_passing": self.fraction_of_grid_passing,
            "worst_cell_pass_pct": self.worst_cell_pass_pct,
            "best_cell_pass_pct": self.best_cell_pass_pct,
            "is_plateau": self.is_plateau,
            "is_cliff": self.is_cliff,
        }


@dataclass
class ParameterRobustnessResult:
    parameter_robustness_score: float           # 0-100 composite across every parameter (+ pair heatmap(s)) checked
    per_parameter: list                          # list[Sensitivity1DResult], unmodified
    pair_heatmaps: list                          # list[ParameterPairRobustness]
    n_parameters_checked: int
    n_cliffs_detected: int
    metric: str
    pass_threshold_pct: float
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "parameter_robustness_score": self.parameter_robustness_score,
            "per_parameter": [p.to_dict() for p in self.per_parameter],
            "pair_heatmaps": [p.to_dict() for p in self.pair_heatmaps],
            "n_parameters_checked": self.n_parameters_checked,
            "n_cliffs_detected": self.n_cliffs_detected,
            "metric": self.metric,
            "pass_threshold_pct": self.pass_threshold_pct,
            "notes": self.notes,
        }


def _closest_index(values: list[float], target: float) -> int:
    return int(np.argmin([abs(v - target) for v in values]))


def _summarize_pair(heatmap: Sensitivity2DResult, pass_threshold_pct: float) -> ParameterPairRobustness:
    grid = heatmap.grid
    a_values, b_values = heatmap.a_values, heatmap.b_values
    flat = [v for row in grid for v in row if isinstance(v, (int, float)) and math.isfinite(v)]
    fraction_passing = (sum(1 for v in flat if v >= pass_threshold_pct) / len(flat) * 100.0) if flat else 0.0
    worst = float(min(flat)) if flat else 0.0
    best = float(max(flat)) if flat else 0.0

    rows = [
        ParameterHeatmapRow(
            param_a_label=heatmap.gene_a_label, param_a_value=a_values[i],
            param_b_label=heatmap.gene_b_label, param_b_value=b_values[j],
            pass_pct=grid[i][j],
        )
        for i in range(len(a_values)) for j in range(len(b_values))
    ]

    # The sweep is centered on each parameter's actual current value (see
    # app.validation.sensitivity._sweep_values), so the middle grid index
    # is a reasonable stand-in for "the winning combination" without
    # needing to re-thread the raw gene values through this module.
    ci, cj = len(a_values) // 2, len(b_values) // 2
    center_val = grid[ci][cj] if a_values and b_values else 0.0
    center_passes = math.isfinite(center_val) and center_val >= pass_threshold_pct

    neighbor_vals = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = ci + di, cj + dj
        if 0 <= ni < len(a_values) and 0 <= nj < len(b_values):
            neighbor_vals.append(grid[ni][nj])
    neighbors_pass = [math.isfinite(v) and v >= pass_threshold_pct for v in neighbor_vals]

    is_plateau = bool(center_passes and neighbor_vals and all(neighbors_pass))
    is_cliff = bool(center_passes and neighbor_vals and not all(neighbors_pass))

    return ParameterPairRobustness(
        heatmap=heatmap, rows=rows, pass_threshold_pct=pass_threshold_pct,
        fraction_of_grid_passing=fraction_passing, worst_cell_pass_pct=worst, best_cell_pass_pct=best,
        is_plateau=is_plateau, is_cliff=is_cliff,
    )


def compute_parameter_robustness(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    metric: str = DEFAULT_PASS_METRIC,
    pass_threshold_pct: float = DEFAULT_PASS_THRESHOLD_PCT,
    max_params: int = 6,
    pct_range: float = 0.5,
    n_steps_1d: int = 9,
    n_steps_2d: int = 7,
    cliff_threshold_pct: float = 40.0,
    n_heatmap_pairs: int = 1,
    tmp_dir: Path | None = None,
) -> ParameterRobustnessResult:
    """
    Runs a 1D sensitivity sweep on every tunable parameter (up to
    `max_params`), then a full 2D heatmap on the `n_heatmap_pairs` most
    sensitive pair(s) of those parameters (ranked by their own 1D sweep's
    biggest observed drop -- the parameters most likely to be hiding a
    cliff), and rolls all of it into a single Parameter Robustness Score.

    `metric` defaults to a genuine 0-100 pass RATE (eval_pass_probability)
    so `pass_threshold_pct` reads naturally as "at least this % of
    resampled accounts still pass the evaluation at this parameter
    value" -- pass a different metric (e.g. a raw profit_factor) together
    with a threshold on THAT metric's own scale if that's the more
    relevant lens for a given strategy.
    """
    genes_available = list_tunable_parameters(strategy, tmp_dir)
    if not genes_available:
        raise RefinementError(
            "This strategy has no tunable numeric parameters to score for robustness."
        )

    per_param = compute_1d_sensitivity(
        df, strategy, risk, prop_rules, mc_config, metric=metric,
        pct_range=pct_range, n_steps=n_steps_1d, max_params=max_params,
        cliff_threshold_pct=cliff_threshold_pct, tmp_dir=tmp_dir,
    )

    per_param_scores: list[float] = []
    n_cliffs = 0
    for r in per_param:
        finite_vals = [v for v in r.metric_values if math.isfinite(v)]
        pass_fraction = (
            sum(1 for v in finite_vals if v >= pass_threshold_pct) / len(finite_vals)
        ) if finite_vals else 0.0
        penalty = CLIFF_SCORE_PENALTY if r.cliff_detected else 0.0
        score_i = min(max(pass_fraction * 100.0 - penalty, 0.0), 100.0)
        per_param_scores.append(score_i)
        if r.cliff_detected:
            n_cliffs += 1

    # Rank by biggest observed adjacent-step drop -- the parameters most
    # likely to be hiding a cliff get the (more expensive) 2D treatment.
    ranked = sorted(per_param, key=lambda r: r.max_pct_drop_between_adjacent_steps, reverse=True)
    pair_labels = [r.gene_label for r in ranked]

    pair_heatmaps: list[ParameterPairRobustness] = []
    n_pairs_possible = max(0, len(pair_labels) - 1)
    for i in range(min(n_heatmap_pairs, n_pairs_possible)):
        a_label, b_label = pair_labels[i], pair_labels[i + 1]
        heatmap = compute_2d_heatmap(
            df, strategy, risk, prop_rules, mc_config,
            gene_label_a=a_label, gene_label_b=b_label, metric=metric,
            pct_range=pct_range, n_steps=n_steps_2d, tmp_dir=tmp_dir,
        )
        pair_heatmaps.append(_summarize_pair(heatmap, pass_threshold_pct))

    overall = float(np.mean(per_param_scores)) if per_param_scores else 0.0
    if pair_heatmaps:
        # A pair can reveal a ridge-shaped interaction neither parameter's
        # own 1D sweep shows alone, so blend the heatmap's own grid
        # pass-fraction in as an equally-weighted additional vote rather
        # than letting the 1D scores alone decide.
        overall = float(np.mean([overall] + [ph.fraction_of_grid_passing for ph in pair_heatmaps]))
    overall = round(min(max(overall, 0.0), 100.0), 1)

    notes: list[str] = []
    if n_cliffs:
        cliff_labels = ", ".join(r.gene_label for r in per_param if r.cliff_detected)
        notes.append(
            f"Optimization cliff detected on: {cliff_labels}. A real edge should not depend on one "
            "exact value for these -- treat the current setting as suspect until a wider search "
            "confirms a plateau nearby."
        )
    if any(ph.is_cliff for ph in pair_heatmaps):
        notes.append(
            "At least one parameter-pair heatmap shows the current combination sitting right at the "
            "edge of a cliff rather than inside a plateau -- see is_cliff/is_plateau on that heatmap."
        )
    if not notes:
        notes.append("No optimization cliffs detected across the parameters checked.")

    return ParameterRobustnessResult(
        parameter_robustness_score=overall,
        per_parameter=per_param,
        pair_heatmaps=pair_heatmaps,
        n_parameters_checked=len(per_param),
        n_cliffs_detected=n_cliffs,
        metric=metric,
        pass_threshold_pct=pass_threshold_pct,
        notes=notes,
    )
