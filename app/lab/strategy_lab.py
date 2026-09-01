"""
Strategy Lab -- Generate -> Test -> Validate -> Stress -> Rank -> Explain -> Approve.

Every other pipeline in this app starts from a strategy someone already
wrote: Run & Report backtests it, Iterative Refinement tunes it, Search
Lab (app.search.batch_runner) generates and validates a whole FAMILY of
related strategies. Strategy Lab is the product-level step above all of
that: point it at a market/timeframe's data and a goal ("maximize
probability of first payout"), and it runs the entire funnel end to end,
handing back a small, ranked, explained shortlist -- not a pile of raw
numbers to interpret by hand.

    Generate     100 candidate strategies across every named hypothesis
                 family (app.search.strategy_space) -- OR a single named
                 family if you already know which kind of edge you're
                 looking for.
    Test         basic viability filter -- one cheap backtest per
                 candidate. (Search Lab Stage 1)
    Validate     GA optimization, then the strict validation gate: full
                 Monte Carlo, multi-fold walk-forward, the lookahead-bias
                 detector, parameter-neighborhood robustness.
                 (Search Lab Stages 2-3 -- delegated to
                 app.search.batch_runner.run_search entirely; this module
                 does not reimplement any of that logic a second way.)
    Stress       two funnel stages that plain Search Lab does NOT have:
                   - Regime testing (app.validation.regime_testing) --
                     does this only work in one kind of market?
                   - Parameter stability (app.validation.
                     parameter_robustness) -- did the GA find a real edge
                     or a lucky exact value?
    Rank         an Untouched Test: every finalist is re-backtested on a
                 slice of the data reserved at the very start and never
                 touched by generation, optimization, validation, regime
                 testing, or parameter-stability testing -- then scored
                 into a single Prop Robustness Score using
                 app.prop.survival_engine's real survival numbers
                 (evaluation pass, 1st/2nd/3rd payout probability, risk
                 of ruin) computed on that untouched slice, not on the
                 data the GA already saw.
    Explain      every finalist carries a plain-language breakdown of
                 exactly why it scored the way it did (see
                 StrategyLabFinalist.explanation).
    Approve      left to the person: this module hands back a ranked,
                 fully-explained shortlist, not an auto-promoted winner --
                 promoting one into the Strategy Library is the same
                 app.search.batch_runner.promote_champion() step every
                 other Search Lab run already uses.

Funnel sizes default to the product spec's own example (100 -> 50 -> 25
-> 15 -> 10 -> 5 -> 3) but every cut is configurable on StrategyLabSpec.
"""
from __future__ import annotations

import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_cost_ladder
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, PropSurvivalResult, ResetEconomics, run_prop_survival_analysis
from app.search.batch_runner import SearchStageConfig, run_search
from app.search.results_db import ResultsDB
from app.search.strategy_space import build_strategy_from_spec, generate_search_space
from app.validation.parameter_robustness import ParameterRobustnessResult, compute_parameter_robustness
from app.validation.regime_testing import RegimeTestResult, run_regime_test

ProgressCallback = Callable[[str], None]

# The exact set of metric strings app.optimize.refinement.compute_fitness
# understands -- StrategyLabSpec.goal_metric must be one of these so it
# can be wired straight into both the search's own fitness_metric AND
# this module's own Monte-Carlo-stage ranking without translation.
KNOWN_GOAL_METRICS = {
    "net_profit", "profit_factor", "sharpe_ratio", "eval_pass_probability",
    "first_payout_probability", "expected_payout", "composite_prop_score", "prop_guide_score",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyLabSpec:
    # Descriptive metadata only -- echoed in logs/reports. The market's
    # actual data, timeframe, and trading-window filtering are properties
    # of the `df` you pass to run_strategy_lab (this app is single-
    # instrument/timeframe per run, same as every other pipeline here);
    # these fields exist so a Strategy Lab report can say what it was
    # actually run for.
    market: str = "Unknown"
    timeframe_label: str = "Unknown"
    trading_window_label: str = "Unknown"
    risk_label: str = "Unknown"        # e.g. "0.25-0.50% per trade" -- actual sizing lives on the RiskConfig you pass in

    # What to optimize for. Must be one of KNOWN_GOAL_METRICS -- this
    # flows straight into both the GA's fitness function and this
    # module's own Monte-Carlo-stage ranking.
    goal_metric: str = "first_payout_probability"

    family: str | None = None          # None = every named Manual family; or one app.search.strategy_space family name
    n_candidates: int = 100

    # Funnel cut sizes -- defaults mirror the product spec's own example:
    # 100 -> 50 -> 25 -> 15 -> 10 -> 5 -> 3.
    stage1_top_n: int = 50             # Generate -> basic viability filter
    stage2_top_n: int = 25             # -> optimization
    walk_forward_top_n: int = 15       # -> walk-forward
    monte_carlo_top_n: int = 10        # -> Monte Carlo
    regime_top_n: int = 5              # -> regime testing
    finalist_count: int = 3            # -> parameter stability -> finalists

    # The most-recent slice of `df`, reserved before Stage 1 even runs and
    # never touched again until the very last (Untouched Test) stage.
    untouched_holdout_frac: float = 0.15

    # Per-stage cost controls -- kept modest by default since this runs
    # the WHOLE funnel end to end; raise any of these for a slower, more
    # thorough lab run.
    ga_population: int = 10
    ga_generations: int = 4
    ga_search_mc_sims: int = 300
    full_mc_sims: int = 3_000
    walk_forward_folds: int = 4
    walk_forward_metric: str = "profit_factor"

    survival_mc_sims: int = 3_000
    survival_life_sims: int = 1_500
    reset_economics: ResetEconomics = field(default_factory=ResetEconomics)

    regime_count: int = 3
    regime_atr_period: int = 14
    regime_min_segment_bars: int = 100

    parameter_stability_max_params: int = 4
    parameter_stability_n_steps_1d: int = 5
    parameter_stability_n_steps_2d: int = 5
    parameter_stability_pass_threshold_pct: float = 50.0

    random_seed: int = 42
    workers: int | None = None

    def __post_init__(self):
        if self.goal_metric not in KNOWN_GOAL_METRICS:
            raise ValueError(
                f"Unknown goal_metric '{self.goal_metric}'. Must be one of: {sorted(KNOWN_GOAL_METRICS)}."
            )
        self.untouched_holdout_frac = min(max(float(self.untouched_holdout_frac), 0.0), 0.5)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StrategyLabFinalist:
    rank: int
    candidate_id: str
    family: str | None
    spec: dict                          # rebuildable candidate spec: {"source_type": ..., "config"/"code_text"+"code_extension": ...}

    # The headline numbers -- deliberately matching the product spec's own
    # example output field-for-field.
    prop_robustness_score: float
    evaluation_pass_pct: float
    first_payout_pct: float
    second_payout_pct: float
    third_payout_pct: float
    median_payout: float
    risk_of_ruin_pct: float
    oos_retention_pct: float
    parameter_stability_pct: float
    regime_stability_pct: float
    cost_stress_survival_pct: float
    worst_losing_streak: float
    median_days_to_payout: float | None

    development_statistics: dict        # app.backtest.statistics stats dict, from the Search Lab validation gate
    untouched_statistics: dict          # same, but re-run on the untouched holdout slice
    survival: dict                      # full app.prop.survival_engine.PropSurvivalResult.to_dict(), on the untouched slice
    regime_test: dict | None            # full app.validation.regime_testing.RegimeTestResult.to_dict()
    parameter_robustness: dict          # full app.validation.parameter_robustness.ParameterRobustnessResult.to_dict()
    explanation: list                   # plain-language reasons behind prop_robustness_score

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


@dataclass
class StrategyLabResult:
    spec: StrategyLabSpec
    total_candidates: int
    stage1_survivors: int
    stage2_survivors: int
    walk_forward_survivors: int
    monte_carlo_survivors: int
    regime_survivors: int
    finalists: list                     # list[StrategyLabFinalist], ranked best-first
    search_run_id: str
    search_db_path: str
    elapsed_seconds: float
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["spec"] = asdict(self.spec)
        d["finalists"] = [f.to_dict() for f in self.finalists]
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spec_from_leaderboard_record(r: dict) -> dict:
    """The uniform candidate spec dict (see app.search.strategy_space)
    for a leaderboard row -- every stage3 record already carries these
    exact fields (see app.search.batch_runner._record_fields_from_spec)."""
    source_type = r.get("source_type", "manual")
    if source_type == "manual":
        return {"source_type": "manual", "config": r.get("config")}
    return {"source_type": source_type, "code_text": r.get("code_text"), "code_extension": r.get("code_extension")}


def _finite(value, default=0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _walk_forward_efficiency(record: dict) -> float:
    wf = record.get("walk_forward") or {}
    return _finite(wf.get("walk_forward_efficiency"), float("-inf"))


def _walk_forward_is_stable_or_unproven(record: dict) -> bool:
    wf = record.get("walk_forward")
    return wf is None or bool(wf.get("is_stable", True))


def _monte_carlo_rank_score(record: dict, goal_metric: str) -> float:
    """The candidate's own goal-aligned fitness (already computed by the
    Search Lab validation gate using this exact goal_metric as its
    fitness_metric -- see run_strategy_lab), lightly penalized for risk of
    ruin so two candidates with similar fitness don't tie-break toward
    the riskier one."""
    fitness = _finite(record.get("fitness"), float("-inf"))
    mc = record.get("mc_summary") or {}
    return fitness - _finite(mc.get("risk_of_ruin_pct")) * 0.1


def _cost_stress_survival_pct(trades: list) -> float:
    """% of the cost-stress ladder's STRESSED rungs (i.e. every rung
    beyond the zero-extra-cost baseline) that stayed net profitable
    (profit factor >= 1.0). No trades / no stressed rungs -> 100.0 (there
    is nothing to have failed)."""
    if not trades:
        return 100.0
    ladder = compute_cost_ladder(trades)
    stressed = [rung for rung in ladder if rung.get("extra_cost_pct_per_trade", 0) > 0]
    if not stressed:
        return 100.0
    survived = sum(1 for rung in stressed if _finite(rung.get("profit_factor")) >= 1.0)
    return 100.0 * survived / len(stressed)


def _score_finalist(
    evaluation_pass_pct: float, first_payout_pct: float, second_payout_pct: float,
    oos_retention_pct: float, parameter_stability_pct: float, regime_stability_pct: float,
    cost_stress_survival_pct: float, risk_of_ruin_pct: float,
) -> tuple[float, list]:
    """
    The Prop Robustness Score: everything computed above, in one number.
    Weighted so that out-of-sample retention, parameter stability, and
    regime stability -- the three questions plain Search Lab does NOT
    already answer -- carry as much combined weight (0.45) as the
    evaluation/payout probabilities themselves (0.40), since a candidate
    that aced the validation gate but is fit to noise, to one exact
    parameter value, or to one kind of market is exactly the failure mode
    this whole module exists to catch.
    """
    weighted = {
        "evaluation_pass": (evaluation_pass_pct, 0.15),
        "first_payout": (first_payout_pct, 0.15),
        "second_payout": (second_payout_pct, 0.10),
        "oos_retention": (min(oos_retention_pct, 100.0), 0.15),
        "parameter_stability": (parameter_stability_pct, 0.15),
        "regime_stability": (regime_stability_pct, 0.15),
        "cost_stress_survival": (cost_stress_survival_pct, 0.10),
    }
    raw_score = sum(value * weight for value, weight in weighted.values()) - risk_of_ruin_pct * 0.05
    score = round(min(max(raw_score, 0.0), 100.0), 1)

    explanation = [
        f"Evaluation-pass probability on the untouched holdout: {evaluation_pass_pct:.1f}%.",
        f"First-payout probability on the untouched holdout: {first_payout_pct:.1f}% "
        f"(second payout: {second_payout_pct:.1f}%).",
        f"Out-of-sample retention (untouched vs. development-window edge): {oos_retention_pct:.1f}%.",
        f"Parameter Robustness Score: {parameter_stability_pct:.1f}/100.",
        f"Regime stability: profitable in {regime_stability_pct:.1f}% of the tested volatility regimes.",
        f"Cost-stress survival: stayed net profitable in {cost_stress_survival_pct:.1f}% of the "
        "stressed cost-ladder rungs.",
        f"Risk of hitting the max-drawdown floor on the untouched holdout: {risk_of_ruin_pct:.1f}%.",
    ]
    return score, explanation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strategy_lab(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    spec: StrategyLabSpec | None = None,
    output_dir: "str | Path | None" = None,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    progress_cb: ProgressCallback | None = None,
) -> StrategyLabResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    spec = spec or StrategyLabSpec()
    t0 = time.time()
    warnings: list[str] = []

    # -- Reserve the untouched holdout BEFORE anything else runs ---------
    n = len(df)
    split = max(1, int(n * (1 - spec.untouched_holdout_frac)))
    development_df = df.iloc[:split].reset_index(drop=True)
    untouched_df = df.iloc[split:].reset_index(drop=True)
    if len(untouched_df) < 50:
        warnings.append(
            "The untouched holdout slice is very small (<50 bars) -- its numbers in the final "
            "ranking should be treated as low-confidence."
        )

    log(
        f"Strategy Lab: {spec.market} / {spec.timeframe_label} / {spec.trading_window_label}, "
        f"goal={spec.goal_metric}. Development window: {len(development_df)} bars, "
        f"untouched holdout: {len(untouched_df)} bars (never touched until the final stage)."
    )

    # -- Generate ----------------------------------------------------------
    space = generate_search_space(
        mode="family", family=spec.family, max_candidates=spec.n_candidates, seed=spec.random_seed,
    )
    log(
        f"Generate: {len(space.candidates)} candidate strategy(ies)"
        + (f" (sampled from {space.total_generated})" if space.sampled else "") + "."
    )

    # -- Basic viability filter -> Optimization -> Validate ----------------
    # Delegated entirely to the app's existing Search Lab funnel -- see
    # this module's own docstring for why.
    stage_cfg = SearchStageConfig(
        stage1_top_n=spec.stage1_top_n,
        ga_population=spec.ga_population, ga_generations=spec.ga_generations, ga_search_sims=spec.ga_search_mc_sims,
        stage2_top_n=spec.stage2_top_n,
        full_mc_sims=spec.full_mc_sims,
        walk_forward_folds=spec.walk_forward_folds, walk_forward_metric=spec.walk_forward_metric,
        fitness_metric=spec.goal_metric,
        workers=spec.workers, random_seed=spec.random_seed,
    )

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="t58_strategy_lab_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "strategy_lab_search.db"

    summary = run_search(
        development_df, risk, prop_rules, space, stage_cfg, str(db_path),
        instrument=instrument, timeframe=timeframe, progress_cb=lambda m: log(f"  {m}"),
    )
    log(
        f"Basic viability filter: {summary.stage1_survivors} survivor(s). "
        f"Optimization: {summary.stage2_survivors} survivor(s). "
        f"Validation gate (Monte Carlo + walk-forward + lookahead + parameter-neighborhood robustness): "
        f"{summary.stage3_survivors} candidate(s) evaluated."
    )

    if not summary.stage3_survivors:
        return StrategyLabResult(
            spec=spec, total_candidates=len(space.candidates),
            stage1_survivors=summary.stage1_survivors, stage2_survivors=summary.stage2_survivors,
            walk_forward_survivors=0, monte_carlo_survivors=0, regime_survivors=0,
            finalists=[], search_run_id=summary.run_id, search_db_path=str(db_path),
            elapsed_seconds=time.time() - t0,
            warnings=warnings + [
                "No candidates survived far enough to reach the validation gate -- nothing to "
                "regime-test, parameter-stability-test, or rank."
            ],
        )

    # Query the DB directly rather than trusting summary.leaderboard's own
    # top_n (run_search's internal leaderboard() call is capped at 25) --
    # this guarantees every Stage 3 record is available for the funnel
    # cuts below regardless of how stage2_top_n is configured.
    with ResultsDB(str(db_path)) as db:
        records = db.leaderboard(summary.run_id, stage="stage3", top_n=max(spec.stage2_top_n, 25), only_passed=False)

    # -- Walk-forward cut --------------------------------------------------
    stable_records = [r for r in records if _walk_forward_is_stable_or_unproven(r)]
    pool = stable_records if stable_records else records
    wf_survivors = sorted(pool, key=_walk_forward_efficiency, reverse=True)[: spec.walk_forward_top_n]
    log(f"Walk-forward: {len(wf_survivors)} candidate(s) advance.")

    # -- Monte Carlo cut -----------------------------------------------------
    mc_survivors = sorted(
        wf_survivors, key=lambda r: _monte_carlo_rank_score(r, spec.goal_metric), reverse=True,
    )[: spec.monte_carlo_top_n]
    log(f"Monte Carlo: {len(mc_survivors)} candidate(s) advance.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="t58_strategy_lab_build_"))
    finalists: list[StrategyLabFinalist] = []
    regime_top: list[dict] = []
    try:
        # -- Regime testing (NEW -- not part of plain Search Lab) ---------
        regime_results: dict[str, "RegimeTestResult | None"] = {}
        scored_by_regime: list[tuple[float, dict]] = []
        for r in mc_survivors:
            candidate_spec = _spec_from_leaderboard_record(r)
            try:
                rt = run_regime_test(
                    development_df, lambda sd=candidate_spec: build_strategy_from_spec(sd, tmp_dir), risk,
                    n_regimes=spec.regime_count, atr_period=spec.regime_atr_period,
                    min_segment_bars=spec.regime_min_segment_bars,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad candidate must not stop the whole lab run
                rt = None
                warnings.append(f"Regime test failed for {r['candidate_id']}: {exc}")
            regime_results[r["candidate_id"]] = rt
            # None = not enough data to form regime buckets -- treated as
            # "unproven", not "unstable" (same convention as
            # app.search.robustness.run_walk_forward's own None case), so
            # it doesn't get unfairly ranked below candidates that simply
            # happened to have enough data to be tested.
            stability = rt.regime_stability_pct if rt is not None else 100.0
            scored_by_regime.append((stability, r))

        scored_by_regime.sort(key=lambda pair: pair[0], reverse=True)
        regime_top = [r for _, r in scored_by_regime[: spec.regime_top_n]]
        log(f"Regime testing: {len(regime_top)} candidate(s) advance.")

        # -- Parameter stability (NEW -- not part of plain Search Lab) ----
        param_results: dict[str, "ParameterRobustnessResult | None"] = {}
        scored_by_param: list[tuple[float, dict]] = []
        param_mc_cfg = MonteCarloConfig(n_simulations=max(200, spec.full_mc_sims // 10), random_seed=spec.random_seed)
        for r in regime_top:
            candidate_spec = _spec_from_leaderboard_record(r)
            strategy = build_strategy_from_spec(candidate_spec, tmp_dir)
            try:
                pr = compute_parameter_robustness(
                    development_df, strategy, risk, prop_rules, param_mc_cfg,
                    max_params=spec.parameter_stability_max_params,
                    n_steps_1d=spec.parameter_stability_n_steps_1d,
                    n_steps_2d=spec.parameter_stability_n_steps_2d,
                    pass_threshold_pct=spec.parameter_stability_pass_threshold_pct,
                    tmp_dir=tmp_dir,
                )
                score = pr.parameter_robustness_score
            except RefinementError:
                # No tunable numeric parameters left to destabilize -- there
                # is nothing for a cliff to hide in, so this is treated as
                # maximally stable rather than penalized for having no
                # knobs (a fixed, hand-tuned strategy isn't LESS robust for
                # having no parameters to sweep).
                pr = None
                score = 100.0
            param_results[r["candidate_id"]] = pr
            scored_by_param.append((score, r))

        scored_by_param.sort(key=lambda pair: pair[0], reverse=True)
        finalist_records = [r for _, r in scored_by_param[: spec.finalist_count]]
        log(f"Parameter stability: {len(finalist_records)} finalist(s) selected.")

        # -- Untouched Test + Final Ranking --------------------------------
        for r in finalist_records:
            candidate_spec = _spec_from_leaderboard_record(r)
            strategy = build_strategy_from_spec(candidate_spec, tmp_dir)
            try:
                bt_untouched = run_backtest(untouched_df, strategy, risk)
            except Exception as exc:  # noqa: BLE001 -- one bad finalist must not stop the whole lab run
                warnings.append(f"Untouched test failed for {r['candidate_id']}: {exc}")
                continue

            dev_stats = r.get("statistics") or {}
            untouched_stats = bt_untouched.statistics.to_dict()
            cost_stress_survival = _cost_stress_survival_pct(bt_untouched.trades)

            survival_result: PropSurvivalResult | None = None
            oos_retention = 0.0
            if not bt_untouched.trades:
                warnings.append(
                    f"{r['candidate_id']} produced zero trades on the untouched holdout slice -- "
                    "its funded-survival numbers below are all 0 and its OOS retention is undefined."
                )
            else:
                survival_cfg = PropSurvivalConfig(
                    n_simulations=spec.survival_mc_sims, life_simulations=spec.survival_life_sims,
                    random_seed=spec.random_seed, reset_economics=spec.reset_economics,
                )
                survival_result = run_prop_survival_analysis(bt_untouched.trades, prop_rules, survival_cfg)

                dev_pf = _finite(dev_stats.get("profit_factor"))
                dev_pf = 10.0 if dev_stats.get("profit_factor") == float("inf") else dev_pf
                unt_pf = _finite(untouched_stats.get("profit_factor"))
                unt_pf = 10.0 if untouched_stats.get("profit_factor") == float("inf") else unt_pf
                # Capped at 150% -- an untouched slice that looks BETTER
                # than development is good news, not something to reward
                # without bound (it's still a small, single out-of-sample
                # slice, not proof the edge got stronger).
                oos_retention = min(unt_pf / dev_pf, 1.5) * 100.0 if dev_pf > 0 else 0.0

            regime_result = regime_results.get(r["candidate_id"])
            param_result = param_results.get(r["candidate_id"])

            evaluation_pass_pct = survival_result.evaluation.probability_pass_evaluation if survival_result else 0.0
            first_payout_pct = survival_result.funded.probability_first_payout if survival_result else 0.0
            second_payout_pct = survival_result.funded.probability_second_payout if survival_result else 0.0
            third_payout_pct = survival_result.funded.probability_third_payout if survival_result else 0.0
            median_payout = survival_result.funded.median_payout_amount if survival_result else 0.0
            median_days_to_payout = survival_result.funded.median_days_to_first_payout if survival_result else None
            worst_losing_streak = survival_result.evaluation.worst_losing_streak_median if survival_result else 0.0
            risk_of_ruin_pct = survival_result.evaluation.probability_hit_max_drawdown if survival_result else 100.0
            parameter_stability_pct = param_result.parameter_robustness_score if param_result else 100.0
            regime_stability_pct = regime_result.regime_stability_pct if regime_result else 100.0

            prop_robustness_score, explanation = _score_finalist(
                evaluation_pass_pct, first_payout_pct, second_payout_pct, oos_retention,
                parameter_stability_pct, regime_stability_pct, cost_stress_survival, risk_of_ruin_pct,
            )

            finalists.append(StrategyLabFinalist(
                rank=0,  # assigned after sorting, below
                candidate_id=r["candidate_id"], family=r.get("family"), spec=candidate_spec,
                prop_robustness_score=prop_robustness_score,
                evaluation_pass_pct=evaluation_pass_pct, first_payout_pct=first_payout_pct,
                second_payout_pct=second_payout_pct, third_payout_pct=third_payout_pct,
                median_payout=median_payout, risk_of_ruin_pct=risk_of_ruin_pct,
                oos_retention_pct=oos_retention, parameter_stability_pct=parameter_stability_pct,
                regime_stability_pct=regime_stability_pct, cost_stress_survival_pct=cost_stress_survival,
                worst_losing_streak=worst_losing_streak, median_days_to_payout=median_days_to_payout,
                development_statistics=dev_stats, untouched_statistics=untouched_stats,
                survival=survival_result.to_dict() if survival_result else {},
                regime_test=regime_result.to_dict() if regime_result else None,
                parameter_robustness=param_result.to_dict() if param_result else {},
                explanation=explanation,
            ))

        finalists.sort(key=lambda f: f.prop_robustness_score, reverse=True)
        for i, finalist in enumerate(finalists, start=1):
            finalist.rank = i
    finally:
        from shutil import rmtree
        rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    log(f"Strategy Lab complete in {elapsed:.1f}s. {len(finalists)} finalist(s) ranked.")
    if finalists:
        best = finalists[0]
        log(
            f"#1: {best.candidate_id} -- Prop Robustness Score {best.prop_robustness_score:.1f}, "
            f"first payout {best.first_payout_pct:.1f}%, OOS retention {best.oos_retention_pct:.1f}%."
        )

    return StrategyLabResult(
        spec=spec, total_candidates=len(space.candidates),
        stage1_survivors=summary.stage1_survivors, stage2_survivors=summary.stage2_survivors,
        walk_forward_survivors=len(wf_survivors), monte_carlo_survivors=len(mc_survivors),
        regime_survivors=len(regime_top),
        finalists=finalists, search_run_id=summary.run_id, search_db_path=str(db_path),
        elapsed_seconds=elapsed, warnings=warnings,
    )
