"""
Full Pipeline -- one button that runs everything.

Every other tab in this app is a separate tool: Run & Report backtests
one fixed configuration, Iterative Refinement tunes it in-sample,
Walk-Forward-Aware GA tunes it against out-of-sample folds, Validation
Lab checks robustness after the fact. Getting from "here's a strategy
file" to "here's the best, validated version of it, ready for a prop
firm" means running several of those in the right order and carrying
the winner from one into the next by hand.

This module does that hand-off automatically:

    Step 1  Baseline           -- one backtest of the strategy exactly as
                                   given, plus a lookahead-bias check for
                                   code strategies. Fails fast (in under a
                                   second) if this produces zero trades,
                                   rather than wasting minutes discovering
                                   that fact three more times over.
    Step 2  Robust optimization -- app.optimize.walkforward_ga's GA, which
                                   scores every candidate ONLY on chained
                                   out-of-sample fold performance (never
                                   in-sample), specifically so the "best"
                                   configuration it finds is one that
                                   generalizes rather than one that just
                                   curve-fits the baseline harder. Skipped
                                   gracefully (not a failure) if the
                                   strategy has no tunable numeric
                                   parameters -- the baseline then IS the
                                   final configuration.
    Step 3  Final validation    -- the winning configuration is re-run
                                   through a full backtest, prop-firm
                                   simulation, and full-fidelity Monte
                                   Carlo on the WHOLE dataset (more
                                   simulations than the search phase used,
                                   since this is the one that counts).
    Step 4  Out-of-sample check -- app.search.robustness.run_walk_forward
                                   on the exact winning configuration, NO
                                   further re-tuning: does this exact
                                   strategy keep working across several
                                   distinct historical stretches?
    Step 5  Holdout check       -- the same chronological in-sample/
                                   holdout split every other pipeline run
                                   in this app already does.
    Step 6  Report + save       -- one full HTML/JSON report (the same
                                   generate_full_report every other run
                                   produces) for the FINAL strategy, a
                                   plain-language verdict, and -- for code
                                   strategies -- the winning source saved
                                   straight into the Strategy Library,
                                   tagged "validated", ready to hand to a
                                   prop firm or plug back into the app.

Every step is best-effort past Step 2: a step that can't run (e.g. not
enough bars for a walk-forward check) is recorded as skipped with a
reason, never allowed to take down a run that otherwise succeeded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.optimize.code_parameter_space import patched_source_for_strategy
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, preflight_signal_check
from app.optimize.walkforward_ga import WalkforwardGAResult, run_walkforward_aware_refinement
from app.prop.simulator import AccountSimResult, PropRules, simulate_account
from app.search.robustness import WalkForwardResult, run_walk_forward
from app.search.strategy_space import build_strategy_from_spec
from app.strategy.base import Strategy
from app.strategy.library import StrategyAlreadyExists, save_strategy_text, set_strategy_status, \
    record_backtest_result

ProgressCallback = Callable[[str], None]


@dataclass
class FullPipelineConfig:
    n_folds: int = 4
    window_mode: str = "rolling"           # for the GA's internal fold split
    ga_population: int = 12
    ga_generations: int = 6
    ga_search_mc_sims: int = 200
    fitness_metric: str = "composite_prop_score"
    final_mc_sims: int = 10_000
    holdout_frac: float = 0.2
    oos_check_folds: int = 4               # for the post-hoc run_walk_forward check
    oos_check_metric: str = "profit_factor"
    random_seed: int | None = 42
    save_to_library: bool = True           # code strategies only -- manual configs aren't files
    library_status: str = "validated"


@dataclass
class FullPipelineResult:
    strategy_source_type: str
    strategy_display_name: str

    baseline_bt: BacktestResult
    baseline_single_run: AccountSimResult
    baseline_mc: MonteCarloResult
    lookahead_summary: str | None

    refinement_ran: bool
    refinement_skip_reason: str | None
    ga_result: WalkforwardGAResult | None

    final_source_type: str
    final_config: dict | None
    final_code_text: str | None
    final_code_extension: str | None

    final_bt: BacktestResult
    final_single_run: AccountSimResult
    final_mc: MonteCarloResult
    final_holdout: dict | None

    oos_validation: WalkForwardResult | None
    oos_validation_skip_reason: str | None

    verdict: str                # "READY" | "MARGINAL" | "NOT READY"
    verdict_reasons: list[str]

    saved_library_path: Path | None
    saved_library_note: str | None

    report_paths: dict
    elapsed_seconds: float
    warnings: list = field(default_factory=list)


def _display_name(strategy: Strategy) -> str:
    if strategy.source_type == "manual":
        return strategy.config.get("name", "Manual Strategy")
    if strategy.source_type == "python":
        return Path(strategy.file_path).stem
    if strategy.source_type == "pinescript":
        return "PineScript Strategy"
    if strategy.source_type == "mql5":
        return "MQL5 Strategy"
    return "Strategy"


def _spec_for_manual(config: dict) -> dict:
    return {"source_type": "manual", "config": config}


def _spec_for_code(source_type: str, code_text: str, extension: str) -> dict:
    return {"source_type": source_type, "code_text": code_text, "code_extension": extension}


def _make_verdict(
    final_mc: MonteCarloResult, oos_validation: WalkForwardResult | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    eval_pass = final_mc.evaluation_pass_probability
    payout = final_mc.first_payout_probability
    ruin = final_mc.risk_of_ruin_pct

    score = 0
    if eval_pass >= 60:
        score += 1
        reasons.append(f"Monte Carlo evaluation-pass probability is {eval_pass:.1f}%.")
    else:
        reasons.append(f"Monte Carlo evaluation-pass probability is only {eval_pass:.1f}% (want 60%+).")
    if payout >= 40:
        score += 1
        reasons.append(f"Monte Carlo first-payout probability is {payout:.1f}%.")
    else:
        reasons.append(f"Monte Carlo first-payout probability is only {payout:.1f}% (want 40%+).")
    if ruin <= 15:
        score += 1
    else:
        reasons.append(f"Monte Carlo risk of ruin is {ruin:.1f}% (want under 15%).")

    if oos_validation is None:
        reasons.append("Out-of-sample fold check couldn't run (not enough data) -- treat this as UNPROVEN, not passing.")
    elif oos_validation.is_stable:
        score += 1
        reasons.append(
            f"Held up across {oos_validation.n_folds} out-of-sample fold(s) "
            f"(walk-forward efficiency {oos_validation.walk_forward_efficiency:.2f})."
        )
    else:
        reasons.append(
            f"Did NOT hold up consistently across {oos_validation.n_folds} out-of-sample fold(s) "
            f"(walk-forward efficiency {oos_validation.walk_forward_efficiency:.2f}, "
            f"below the {oos_validation.stability_threshold:.2f} stability threshold)."
        )

    if score >= 4:
        verdict = "READY"
    elif score >= 2:
        verdict = "MARGINAL"
    else:
        verdict = "NOT READY"
    return verdict, reasons


def run_full_pipeline(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: FullPipelineConfig | None = None,
    progress_cb: ProgressCallback | None = None,
    instrument: str = "unknown",
) -> FullPipelineResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or FullPipelineConfig()
    t0 = time.time()
    warnings: list[str] = []
    display_name = _display_name(strategy)

    # -- Step 1: baseline -----------------------------------------------
    log(f"Step 1/6: Baseline run for '{display_name}'...")
    preflight_signal_check(df, strategy, risk, "Full Pipeline")
    baseline_bt = run_backtest(df, strategy, risk)
    for w in baseline_bt.warnings:
        log(f"  WARNING: {w}")
        warnings.append(w)

    lookahead_summary = None
    if strategy.source_type in ("python", "pinescript", "mql5"):
        try:
            from app.strategy.lookahead_check import check_for_lookahead
            lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
            lookahead_summary = lookahead_result.summary()
            log(f"  Lookahead check: {lookahead_summary}")
        except Exception:
            log("  Lookahead check failed to run (skipped, best-effort only).")

    pnls = [t.pnl for t in baseline_bt.trades]
    dates = [t.entry_time for t in baseline_bt.trades]
    baseline_single_run = simulate_account(pnls, dates, prop_rules)
    baseline_mc = run_monte_carlo(baseline_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed))
    log(
        f"  Baseline: {len(baseline_bt.trades)} trades, net ${baseline_bt.statistics.net_profit:,.2f}, "
        f"eval pass {baseline_mc.evaluation_pass_probability:.1f}%, payout {baseline_mc.first_payout_probability:.1f}%."
    )

    # -- Step 2: robust (walk-forward-aware) optimization ----------------
    log("Step 2/6: Searching for a more robust configuration (walk-forward-aware GA)...")
    refinement_ran = False
    refinement_skip_reason = None
    ga_result: WalkforwardGAResult | None = None
    final_source_type = strategy.source_type
    final_config = strategy.config if strategy.source_type == "manual" else None
    final_code_text, final_code_ext = (None, None)
    if strategy.source_type != "manual":
        final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])

    try:
        refine_cfg = RefinementConfig(
            population_size=cfg.ga_population,
            generations=cfg.ga_generations,
            fitness_metric=cfg.fitness_metric,
            search_monte_carlo_sims=cfg.ga_search_mc_sims,
            random_seed=cfg.random_seed,
        )
        ga_result = run_walkforward_aware_refinement(
            df, strategy, risk, prop_rules,
            MonteCarloConfig(n_simulations=cfg.ga_search_mc_sims, random_seed=cfg.random_seed),
            refinement_config=refine_cfg,
            n_folds=cfg.n_folds, window_mode=cfg.window_mode,
            progress_cb=lambda m: log(f"  {m}"),
        )
        refinement_ran = True
        if ga_result.best.oos_trade_count > 0:
            final_config = ga_result.best.config
            final_code_text = ga_result.best.code_text
            final_code_ext = ga_result.best.code_extension
            log(
                f"  Winning configuration: chained-OOS fitness {ga_result.best.fitness:.3f} "
                f"({ga_result.best.oos_trade_count} OOS trades across {ga_result.n_folds} fold(s))."
            )
            if ga_result.overfitting_gap is not None and ga_result.overfitting_gap > 0:
                warnings.append(
                    f"Overfitting gap (in-sample fitness minus chained-OOS fitness): "
                    f"{ga_result.overfitting_gap:.3f}. A large positive gap means the winning "
                    f"configuration looks noticeably better in-sample than out-of-sample."
                )
            warnings.extend(ga_result.warnings)
        else:
            warnings.append(
                "The walk-forward-aware GA's best candidate still produced zero out-of-sample "
                "trades -- keeping the original baseline configuration as final instead of a "
                "'winner' that never actually traded out-of-sample."
            )
    except RefinementError as exc:
        refinement_skip_reason = str(exc)
        log(f"  Optimization skipped: {exc}")

    # -- Build the final strategy -----------------------------------------------
    if final_source_type == "manual":
        final_spec = _spec_for_manual(final_config)
    else:
        final_spec = _spec_for_code(final_source_type, final_code_text, final_code_ext)

    from tempfile import mkdtemp
    from shutil import rmtree
    final_tmp_dir = Path(mkdtemp(prefix="t58_fullpipeline_")) if final_source_type != "manual" else None
    try:
        final_strategy = build_strategy_from_spec(final_spec, final_tmp_dir)

        # -- Step 3: final validation ------------------------------------
        log("Step 3/6: Final validation (full backtest, prop simulation, Monte Carlo)...")
        final_bt = run_backtest(df, final_strategy, risk)
        for w in final_bt.warnings:
            log(f"  WARNING: {w}")
            warnings.append(w)
        if not final_bt.trades:
            # Should not happen (the GA never returns a worse-than-baseline
            # candidate, and baseline already passed preflight), but never
            # trust that blindly -- fall back to the baseline strategy/spec.
            warnings.append(
                "The selected final configuration unexpectedly produced zero trades on "
                "re-validation -- falling back to the original baseline configuration."
            )
            final_source_type = strategy.source_type
            final_config = strategy.config if strategy.source_type == "manual" else None
            if strategy.source_type != "manual":
                final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])
            final_spec = (
                _spec_for_manual(final_config) if final_source_type == "manual"
                else _spec_for_code(final_source_type, final_code_text, final_code_ext)
            )
            final_strategy = build_strategy_from_spec(final_spec, final_tmp_dir)
            final_bt = baseline_bt

        pnls = [t.pnl for t in final_bt.trades]
        dates = [t.entry_time for t in final_bt.trades]
        final_single_run = simulate_account(pnls, dates, prop_rules)
        final_mc = run_monte_carlo(final_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.final_mc_sims, random_seed=cfg.random_seed))
        log(
            f"  Final: {len(final_bt.trades)} trades, net ${final_bt.statistics.net_profit:,.2f}, "
            f"eval pass {final_mc.evaluation_pass_probability:.1f}%, payout {final_mc.first_payout_probability:.1f}%."
        )

        # -- Step 4: out-of-sample fold check (no re-tuning) --------------
        log("Step 4/6: Out-of-sample fold check (same configuration, no further tuning)...")
        oos_validation = None
        oos_skip_reason = None
        try:
            oos_validation = run_walk_forward(
                df, lambda: build_strategy_from_spec(final_spec, final_tmp_dir), risk,
                n_folds=cfg.oos_check_folds, metric=cfg.oos_check_metric,
            )
            if oos_validation is None:
                oos_skip_reason = "Not enough bars to build the requested number of out-of-sample folds."
            else:
                log(
                    f"  Walk-forward efficiency {oos_validation.walk_forward_efficiency:.2f} "
                    f"({'stable' if oos_validation.is_stable else 'NOT stable'})."
                )
        except Exception as exc:  # noqa: BLE001 -- best-effort validation step
            oos_skip_reason = f"Out-of-sample check failed to run: {exc}"
            log(f"  {oos_skip_reason}")

        # -- Step 5: holdout check ----------------------------------------
        log("Step 5/6: Out-of-sample holdout check...")
        try:
            final_holdout = run_holdout_comparison(df, final_strategy, risk, holdout_frac=cfg.holdout_frac)
        except Exception:
            final_holdout = None
            log("  Holdout check skipped (not enough data to split).")

        # -- Step 6: report + save -----------------------------------------
        log("Step 6/6: Generating final report...")
        verdict, verdict_reasons = _make_verdict(final_mc, oos_validation)

        elapsed = time.time() - t0
        return _finish(
            strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
            refinement_ran, refinement_skip_reason, ga_result,
            final_source_type, final_config, final_code_text, final_code_ext,
            final_bt, final_single_run, final_mc, final_holdout,
            oos_validation, oos_skip_reason, verdict, verdict_reasons,
            df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
            instrument,
        )
    finally:
        if final_tmp_dir is not None:
            rmtree(final_tmp_dir, ignore_errors=True)


def _finish(
    strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
    refinement_ran, refinement_skip_reason, ga_result,
    final_source_type, final_config, final_code_text, final_code_ext,
    final_bt, final_single_run, final_mc, final_holdout,
    oos_validation, oos_skip_reason, verdict, verdict_reasons,
    df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
    instrument="unknown",
) -> FullPipelineResult:
    """Writes the report + (for code strategies) saves the winner into the
    Strategy Library. Split out of run_full_pipeline only to keep that
    function's main try/finally block readable."""
    from app.reports.generator import generate_full_report

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    final_strategy_name = f"{display_name} (Full Pipeline)"
    report_paths = generate_full_report(
        output_dir=output_dir,
        strategy_name=final_strategy_name,
        strategy_source_type=final_source_type,
        instrument=instrument,
        timeframe="unknown",
        backtest_period=period,
        backtest_result=final_bt,
        prop_rules=prop_rules,
        prop_single_run=final_single_run,
        monte_carlo_result=final_mc,
        basename="full_pipeline_report",
        holdout_comparison=final_holdout,
        risk_config=risk,
        price_df=df,
    )

    saved_library_path = None
    saved_library_note = None
    if cfg.save_to_library and final_source_type in ("python", "pinescript", "mql5") and final_code_text:
        ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}[final_source_type]
        base_name = Path(display_name).stem.replace(" ", "_") or "full_pipeline_strategy"
        filename = f"{base_name}_pipeline{ext}"
        try:
            try:
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            except StrategyAlreadyExists:
                filename = f"{base_name}_pipeline_{int(time.time())}{ext}"
                saved_library_path = save_strategy_text(final_code_text, filename, final_source_type, overwrite=False)
            set_strategy_status(final_source_type, filename, cfg.library_status)
            record_backtest_result(final_source_type, filename, {
                "trades": len(final_bt.trades),
                "net_profit": round(final_bt.statistics.net_profit, 2),
                "win_rate": round(final_bt.statistics.win_rate, 1),
                "max_dd": round(final_bt.statistics.max_drawdown_pct, 2),
                "eval_pass_probability": round(final_mc.evaluation_pass_probability, 1),
                "first_payout_probability": round(final_mc.first_payout_probability, 1),
                "verdict": verdict,
                "report_html": str(report_paths["html"]),
            })
            saved_library_note = f"Saved to the Strategy Library as '{filename}' (status: {cfg.library_status})."
            log(f"  {saved_library_note}")
        except Exception as exc:  # noqa: BLE001 -- saving to the library is a convenience, not core output
            saved_library_note = f"Could not save to the Strategy Library: {exc}"
            log(f"  {saved_library_note}")
    elif final_source_type == "manual":
        saved_library_note = (
            "Manual Strategy Builder configurations aren't files, so there's nothing to save to "
            "the Strategy Library -- copy the winning settings from the report, or use "
            "'Apply Best Config to Strategy Tab' after a standalone Iterative Refinement run."
        )

    log(f"\nFull Pipeline complete in {elapsed:.1f}s. Verdict: {verdict}.")

    return FullPipelineResult(
        strategy_source_type=strategy.source_type,
        strategy_display_name=display_name,
        baseline_bt=baseline_bt,
        baseline_single_run=baseline_single_run,
        baseline_mc=baseline_mc,
        lookahead_summary=lookahead_summary,
        refinement_ran=refinement_ran,
        refinement_skip_reason=refinement_skip_reason,
        ga_result=ga_result,
        final_source_type=final_source_type,
        final_config=final_config,
        final_code_text=final_code_text,
        final_code_extension=final_code_ext,
        final_bt=final_bt,
        final_single_run=final_single_run,
        final_mc=final_mc,
        final_holdout=final_holdout,
        oos_validation=oos_validation,
        oos_validation_skip_reason=oos_skip_reason,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        saved_library_path=saved_library_path,
        saved_library_note=saved_library_note,
        report_paths=report_paths,
        elapsed_seconds=elapsed,
        warnings=warnings,
    )
