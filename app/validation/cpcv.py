"""
Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest
Overfitting (PBO) -- Bailey, Borwein, Lopez de Prado & Zhu, "The
Probability of Backtest Overfitting" (2017), and Lopez de Prado,
"Advances in Financial Machine Learning" (2018), ch. 12.

Why this exists (and how it differs from what's already in the app):
  app.search.robustness.run_walk_forward() checks ONE fixed configuration
  across a handful of sequential rolling folds -- useful, but it only
  samples one specific partition of the data into train/test.
  app.search.robustness.deflated_sharpe_ratio() corrects a single
  champion's Sharpe for how many candidates were tried, but doesn't tell
  you which train/test partitions the champion's edge actually survives.

CPCV instead:
  1. Splits the dataset into N contiguous, equal-size groups.
  2. Enumerates every combination of k groups as the "test" set for that
     path (C(N, k) total paths), with the remaining N-k groups as train.
     Because both the test AND train sets differ on every path, this
     samples far more of the possible ways the data could have been
     partitioned than one holdout split or one rolling walk-forward ever
     does, without requiring more historical data.
  3. Purges/embargoes a few bars adjacent to every train/test boundary,
     so a slow indicator's warm-up state can't leak information across
     the split.

Two entry points:
  run_cpcv()   -- evaluates ONE fixed strategy across every CPCV path and
                  reports the distribution of its out-of-sample metric
                  (mean/median/std/pct of paths that are net losers).
                  This is the right tool when you already have a single
                  strategy you want to stress-test against partition
                  choice.
  compute_pbo() -- the genuine, textbook Probability of Backtest
                  Overfitting: given a POOL of candidates (e.g. a Search
                  Lab leaderboard, or several Iterative Refinement
                  candidates), for every CPCV path it ranks all
                  candidates by their IN-SAMPLE performance, takes
                  whichever candidate looked best in-sample, and checks
                  where that same candidate ranks OUT-OF-sample. PBO is
                  the fraction of paths where the in-sample "winner"
                  finished in the bottom half out-of-sample -- i.e. the
                  probability that picking the best backtest result was
                  the same as picking noise.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.search.strategy_space import StrategySpaceError, build_strategy_from_spec


class CPCVError(Exception):
    """Raised when CPCV cannot proceed (e.g. not enough data for the requested grouping)."""


def _metric_value(stats: dict, metric: str, trades: list | None = None, prop_rules=None, mc_cfg=None) -> float:
    """Same convention as app.search.robustness._metric_value: pulls
    ordinary metrics off the backtest stats dict, but "eval_pass_probability"
    (probability of hitting the profit target before a daily-loss/
    max-drawdown/consistency limit) isn't in that dict -- it only exists
    after a Monte Carlo run -- so this runs a small per-path Monte Carlo
    for it instead, whenever prop_rules is available."""
    if metric == "eval_pass_probability" and prop_rules is not None:
        from app.monte_carlo.engine import eval_pass_probability_for_trades
        return eval_pass_probability_for_trades(trades or [], prop_rules, mc_cfg)
    v = stats.get(metric, 0.0)
    if v == float("inf"):
        return 10.0
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0.0
    return float(v)


def _group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    size = n // n_groups
    bounds = []
    start = 0
    for g in range(n_groups):
        end = start + size if g < n_groups - 1 else n
        bounds.append((start, end))
        start = end
    return bounds


def _slice_with_embargo(df: pd.DataFrame, lo: int, hi: int, embargo: int, n: int) -> pd.DataFrame:
    lo2 = min(lo + embargo, hi)
    hi2 = max(hi - embargo, lo2)
    return df.iloc[lo2:hi2].reset_index(drop=True)


# ---------------------------------------------------------------------------
# run_cpcv -- single fixed strategy, distribution of OOS metric across paths
# ---------------------------------------------------------------------------

@dataclass
class CPCVPathResult:
    path_index: int
    test_group_indices: tuple
    train_bars: int
    test_bars: int
    in_sample_metric: float
    out_of_sample_metric: float


@dataclass
class CPCVResult:
    metric: str
    n_groups: int
    n_test_groups: int
    n_paths: int
    paths: list  # list[CPCVPathResult]
    mean_oos_metric: float
    median_oos_metric: float
    std_oos_metric: float
    pct_paths_oos_negative: float
    pct_paths_oos_below_is: float
    mean_is_metric: float
    degradation: float  # mean_is_metric - mean_oos_metric
    is_robust: bool
    robustness_threshold: float

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "paths"}
        d["paths"] = [p.__dict__ for p in self.paths]
        return d


def run_cpcv(
    df: pd.DataFrame,
    strategy_builder,
    risk: RiskConfig,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    metric: str = "eval_pass_probability",
    robustness_threshold: float = 0.5,
    max_paths: int | None = 30,
    prop_rules=None,
    mc_cfg=None,
) -> CPCVResult:
    """
    strategy_builder: zero-argument callable returning a FRESH Strategy
    instance each call (same convention as app.search.robustness.
    run_walk_forward) -- required because some strategy sources cache
    state keyed to the data they last saw.
    """
    if metric == "eval_pass_probability" and prop_rules is None:
        metric = "profit_factor"  # can't run per-path Monte Carlo without prop rules

    n = len(df)
    if n_groups < 3 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise CPCVError("n_groups must be >= 3 and 1 <= n_test_groups < n_groups.")
    if n < n_groups * 20:
        raise CPCVError(
            f"Not enough bars ({n}) to split into {n_groups} groups of a "
            "meaningful size. Use fewer groups or more data."
        )

    embargo = max(int(n * embargo_frac / n_groups), 0)
    bounds = _group_bounds(n, n_groups)
    all_combos = list(itertools.combinations(range(n_groups), n_test_groups))
    if max_paths is not None and len(all_combos) > max_paths:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), size=max_paths, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]

    paths: list[CPCVPathResult] = []
    for path_idx, test_groups in enumerate(all_combos):
        test_group_set = set(test_groups)
        test_frames, train_frames = [], []
        for g, (lo, hi) in enumerate(bounds):
            if g in test_group_set:
                test_frames.append(_slice_with_embargo(df, lo, hi, embargo, n))
            else:
                train_frames.append(_slice_with_embargo(df, lo, hi, embargo, n))

        test_df = pd.concat(test_frames, ignore_index=True) if test_frames else df.iloc[0:0]
        train_df = pd.concat(train_frames, ignore_index=True) if train_frames else df.iloc[0:0]
        if len(test_df) < 5 or len(train_df) < 10:
            continue

        train_bt = run_backtest(train_df, strategy_builder(), risk)
        test_bt = run_backtest(test_df, strategy_builder(), risk)
        is_val = _metric_value(train_bt.statistics.to_dict(), metric, train_bt.trades, prop_rules, mc_cfg)
        oos_val = _metric_value(test_bt.statistics.to_dict(), metric, test_bt.trades, prop_rules, mc_cfg)

        paths.append(CPCVPathResult(
            path_index=path_idx,
            test_group_indices=tuple(test_groups),
            train_bars=len(train_df),
            test_bars=len(test_df),
            in_sample_metric=is_val,
            out_of_sample_metric=oos_val,
        ))

    if not paths:
        raise CPCVError("No CPCV path produced enough train/test bars to evaluate.")

    oos_vals = np.array([p.out_of_sample_metric for p in paths])
    is_vals = np.array([p.in_sample_metric for p in paths])
    mean_oos = float(oos_vals.mean())
    mean_is = float(is_vals.mean())

    return CPCVResult(
        metric=metric,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        n_paths=len(paths),
        paths=paths,
        mean_oos_metric=mean_oos,
        median_oos_metric=float(np.median(oos_vals)),
        std_oos_metric=float(np.std(oos_vals, ddof=1)) if len(oos_vals) > 1 else 0.0,
        pct_paths_oos_negative=float((oos_vals < 0).mean() * 100),
        pct_paths_oos_below_is=float((oos_vals < is_vals).mean() * 100),
        mean_is_metric=mean_is,
        degradation=mean_is - mean_oos,
        is_robust=(mean_oos >= robustness_threshold * mean_is) if mean_is > 0 else (mean_oos >= 0),
        robustness_threshold=robustness_threshold,
    )


# ---------------------------------------------------------------------------
# compute_pbo -- genuine multi-candidate Probability of Backtest Overfitting
# ---------------------------------------------------------------------------

@dataclass
class PBOResult:
    n_candidates: int
    n_groups: int
    n_test_groups: int
    n_paths: int
    metric: str
    pbo: float                       # fraction of paths where the IS-best candidate ranked <= median OOS
    logits: list                     # per-path logit of the IS-best candidate's relative OOS rank
    is_best_candidate_per_path: list  # candidate index chosen as IS-best, per path
    oos_rank_of_is_best_per_path: list  # 0..1, 0 = worst OOS, 1 = best OOS, of that same candidate
    overall_best_candidate_index: int  # candidate with the best mean IS metric across all paths
    mean_is_by_candidate: list
    mean_oos_by_candidate: list
    note: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compute_pbo(
    df: pd.DataFrame,
    candidate_specs: list[dict],
    risk: RiskConfig,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    metric: str = "eval_pass_probability",
    max_paths: int | None = 30,
    prop_rules=None,
    mc_cfg=None,
) -> PBOResult:
    """
    candidate_specs: the same uniform candidate-spec dicts used throughout
    the Search Lab (see app.search.strategy_space):
        {"source_type": "manual", "config": {...}}
        {"source_type": "python"/"pinescript"/"mql5", "code_text": "...", "code_extension": ".py"}
    Typically this is a Search Lab leaderboard slice, or the population of
    an Iterative Refinement run's final generation -- any pool of several
    independently-tried candidates evaluated on the SAME data.

    PBO is only meaningful for n_candidates > 1 (it is measuring whether
    SELECTING a winner among several candidates by in-sample performance
    generalizes) -- with a single candidate the function still runs but
    the result is degenerate by construction (pbo will be 0 or 1 and the
    note says so).
    """
    if metric == "eval_pass_probability" and prop_rules is None:
        metric = "sharpe_ratio"  # can't run per-candidate Monte Carlo without prop rules

    n_candidates = len(candidate_specs)
    if n_candidates < 1:
        raise CPCVError("compute_pbo requires at least one candidate.")

    n = len(df)
    if n_groups < 3 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise CPCVError("n_groups must be >= 3 and 1 <= n_test_groups < n_groups.")
    if n < n_groups * 20:
        raise CPCVError(
            f"Not enough bars ({n}) to split into {n_groups} groups of a meaningful size."
        )

    embargo = max(int(n * embargo_frac / n_groups), 0)
    bounds = _group_bounds(n, n_groups)
    all_combos = list(itertools.combinations(range(n_groups), n_test_groups))
    if max_paths is not None and len(all_combos) > max_paths:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), size=max_paths, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]

    with_tmp = any(spec.get("source_type") == "python" for spec in candidate_specs)
    tmp_dir = None
    if with_tmp:
        import shutil
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="t58_pbo_"))

    try:
        logits: list[float] = []
        is_best_idx_per_path: list[int] = []
        oos_rank_per_path: list[float] = []
        is_matrix = np.zeros((len(all_combos), n_candidates))
        oos_matrix = np.zeros((len(all_combos), n_candidates))
        valid_path_count = 0

        for path_idx, test_groups in enumerate(all_combos):
            test_group_set = set(test_groups)
            test_frames, train_frames = [], []
            for g, (lo, hi) in enumerate(bounds):
                if g in test_group_set:
                    test_frames.append(_slice_with_embargo(df, lo, hi, embargo, n))
                else:
                    train_frames.append(_slice_with_embargo(df, lo, hi, embargo, n))
            test_df = pd.concat(test_frames, ignore_index=True) if test_frames else df.iloc[0:0]
            train_df = pd.concat(train_frames, ignore_index=True) if train_frames else df.iloc[0:0]
            if len(test_df) < 5 or len(train_df) < 10:
                continue

            is_vals, oos_vals = [], []
            for c_idx, spec in enumerate(candidate_specs):
                try:
                    train_strategy = build_strategy_from_spec(spec, tmp_dir)
                    test_strategy = build_strategy_from_spec(spec, tmp_dir)
                except StrategySpaceError:
                    is_vals.append(0.0)
                    oos_vals.append(0.0)
                    continue
                train_bt = run_backtest(train_df, train_strategy, risk)
                test_bt = run_backtest(test_df, test_strategy, risk)
                is_vals.append(_metric_value(train_bt.statistics.to_dict(), metric, train_bt.trades, prop_rules, mc_cfg))
                oos_vals.append(_metric_value(test_bt.statistics.to_dict(), metric, test_bt.trades, prop_rules, mc_cfg))

            is_matrix[path_idx, :] = is_vals
            oos_matrix[path_idx, :] = oos_vals
            valid_path_count += 1

            is_best = int(np.argmax(is_vals))
            # OOS relative rank of that same candidate: fraction of candidates
            # it beats out-of-sample, in (0, 1). 0.5 boundary => coin-flip.
            oos_arr = np.array(oos_vals)
            rank = float((oos_arr < oos_arr[is_best]).sum()) / max(n_candidates - 1, 1) if n_candidates > 1 else 1.0
            # avoid exact 0/1 for a well-defined logit
            rank_clamped = min(max(rank, 1e-3), 1 - 1e-3)
            logit = math.log(rank_clamped / (1 - rank_clamped))

            is_best_idx_per_path.append(is_best)
            oos_rank_per_path.append(rank)
            logits.append(logit)

        if valid_path_count == 0:
            raise CPCVError("No CPCV path produced enough train/test bars to evaluate any candidate.")

        pbo = float(np.mean([r <= 0.5 for r in oos_rank_per_path])) if n_candidates > 1 else (0.0 if oos_rank_per_path and oos_rank_per_path[0] >= 0.5 else 1.0)

        mean_is_by_candidate = is_matrix[:valid_path_count].mean(axis=0).tolist() if valid_path_count else [0.0] * n_candidates
        mean_oos_by_candidate = oos_matrix[:valid_path_count].mean(axis=0).tolist() if valid_path_count else [0.0] * n_candidates
        overall_best = int(np.argmax(mean_is_by_candidate))

        note = (
            f"PBO estimated from {valid_path_count} combinatorial train/test paths "
            f"(C({n_groups},{n_test_groups}), capped at {max_paths or 'all'}). "
            "Interpretation: the probability that the candidate which looked "
            "best in-sample is, out-of-sample, no better than a coin flip "
            "against the rest of the pool. Values above ~0.5 mean the search "
            "process itself is more likely to be selecting noise than signal."
        )
        if n_candidates < 2:
            note += " (n_candidates == 1: this result is degenerate -- PBO measures SELECTION among candidates.)"

        return PBOResult(
            n_candidates=n_candidates,
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            n_paths=valid_path_count,
            metric=metric,
            pbo=pbo,
            logits=logits,
            is_best_candidate_per_path=is_best_idx_per_path,
            oos_rank_of_is_best_per_path=oos_rank_per_path,
            overall_best_candidate_index=overall_best,
            mean_is_by_candidate=mean_is_by_candidate,
            mean_oos_by_candidate=mean_oos_by_candidate,
            note=note,
        )
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
