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
from app.validation.icir import ICIRGateResult, run_icir_gate_from_backtest

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
    # Step 1's baseline Monte Carlo run is diagnostic only -- it's logged
    # and reported alongside the final numbers, but never feeds the
    # verdict (see _make_verdict, which only reads final_mc) and gets
    # thrown away entirely whenever Step 2 finds a better configuration.
    # Running it at the same full_mc_sims fidelity as the run that
    # actually counts wasted a meaningful share of every pipeline run for
    # no quality benefit, so it defaults lower here -- this is a pure
    # speed change with no effect on the report's headline numbers.
    baseline_mc_sims: int = 2_000
    holdout_frac: float = 0.2
    oos_check_folds: int = 4               # for the post-hoc run_walk_forward check
    oos_check_metric: str = "profit_factor"
    random_seed: int | None = 42
    save_to_library: bool = True           # code strategies only -- manual configs aren't files
    library_status: str | None = None      # None = auto-pick from the READY/MARGINAL/NOT READY
                                            # verdict (see _VERDICT_TO_LIBRARY_STATUS below);
                                            # pass an explicit STRATEGY_STATUSES value to always
                                            # use that one regardless of verdict.
    # Passed straight through to Step 2's walk-forward-aware GA (by far
    # the most expensive step in a typical run) -- see
    # app.optimize.walkforward_ga.run_walkforward_aware_refinement for
    # what these control. parallel=True (the default) is a pure speed
    # change: it evaluates a generation's candidates across worker
    # processes instead of one at a time, with automatic fallback to a
    # single process if that can't be set up, so it never changes which
    # configuration wins.
    parallel_search: bool = True
    parallel_search_max_workers: int | None = None


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

    icir_gate: "ICIRGateResult | None"
    icir_gate_skip_reason: str | None

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
    icir_gate: "ICIRGateResult | None" = None,
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

    if icir_gate is None:
        reasons.append(
            "ICIR / signal-decay / Bonferroni-corrected significance gate couldn't run -- "
            "treat this as UNPROVEN, not passing."
        )
    elif icir_gate.ok:
        score += 1
        reasons.append("Passed the ICIR / signal-decay / Bonferroni-corrected significance gate: " +
                        " ".join(icir_gate.reasons))
    else:
        reasons.append("Did NOT pass the ICIR / signal-decay / Bonferroni-corrected significance gate: " +
                        " ".join(icir_gate.reasons))

    if score >= 5:
        verdict = "READY"
    elif score >= 3:
        verdict = "MARGINAL"
    else:
        verdict = "NOT READY"
    return verdict, reasons


# What each Full Pipeline verdict tags a newly-saved strategy with in the
# Strategy Library when FullPipelineConfig.library_status is left as None
# (the default) -- READY strategies still land on "validated" rather than
# jumping straight to "ready_for_demo"/"ready_for_live", since those last
# two stages are meant to reflect actual demo/live trading experience, not
# just a clean backtest -- promote it yourself once you've watched it run.
_VERDICT_TO_LIBRARY_STATUS = {
    "READY": "validated",
    "MARGINAL": "tested_passed",
    "NOT READY": "tested_failed",
}


def run_full_pipeline(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    cfg: FullPipelineConfig | None = None,
    progress_cb: ProgressCallback | None = None,
    instrument: str = "unknown",
    ollama_settings: "OllamaSettings | None" = None,
) -> FullPipelineResult:
    """
    ollama_settings: optional. When provided and `.is_usable` (enabled,
    with a host configured -- see app.ai.ollama_settings), Step 2's
    walk-forward-aware GA asks a local Ollama model for candidate
    parameter values once per generation and seeds them into that
    generation's population alongside the normal random/bred candidates
    (see app.optimize.walkforward_ga's ai_suggest_cb). Every suggestion
    still goes through the exact same backtest/prop-sim/Monte Carlo
    evaluation as any other candidate -- the model proposes numbers for
    already-existing tunable parameters, never code. None/disabled (the
    default) runs exactly as before this parameter existed; any failure
    to reach Ollama degrades to the same "AI assist is off" behavior
    without interrupting the pipeline.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = cfg or FullPipelineConfig()
    t0 = time.time()
    warnings: list[str] = []
    display_name = _display_name(strategy)

    # -- Step 1: baseline -----------------------------------------------
    log(f"Step 1/7: Baseline run for '{display_name}'...")
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
    baseline_mc = run_monte_carlo(baseline_bt.trades, prop_rules, MonteCarloConfig(n_simulations=cfg.baseline_mc_sims, random_seed=cfg.random_seed))
    log(
        f"  Baseline: {len(baseline_bt.trades)} trades, net ${baseline_bt.statistics.net_profit:,.2f}, "
        f"eval pass {baseline_mc.evaluation_pass_probability:.1f}%, payout {baseline_mc.first_payout_probability:.1f}%."
    )

    # -- Step 2: robust (walk-forward-aware) optimization ----------------
    log("Step 2/7: Searching for a more robust configuration (walk-forward-aware GA)...")
    refinement_ran = False
    refinement_skip_reason = None
    ga_result: WalkforwardGAResult | None = None
    final_source_type = strategy.source_type
    final_config = strategy.config if strategy.source_type == "manual" else None
    final_code_text, final_code_ext = (None, None)
    if strategy.source_type != "manual":
        final_code_text, final_code_ext = patched_source_for_strategy(strategy, [], [])

    ai_suggest_cb = None
    if ollama_settings is not None and ollama_settings.is_usable:
        from app.ai.ollama_client import OllamaClient
        from app.ai.research_library import find_relevant_excerpts
        from app.optimize.gene_fitness_analysis import analyze_gene_fitness_correlation

        ollama_client = OllamaClient(ollama_settings)
        # Captures Step 1's baseline stats once -- good enough context for
        # every generation's request without re-running anything extra.
        # A live "how's the search going" readout would need re-summarizing
        # the current best each generation; left for a future pass.
        baseline_stats_summary = {
            k: v for k, v in baseline_bt.statistics.to_dict().items()
            if k in ("net_profit", "win_rate", "profit_factor", "max_drawdown_pct", "total_trades")
        }
        prop_rules_summary = {
            "account_size": prop_rules.account_size,
            "max_drawdown_pct": prop_rules.max_drawdown_pct,
            "daily_loss_limit_pct": prop_rules.daily_loss_limit_pct,
            "evaluation_profit_target_pct": prop_rules.evaluation_profit_target_pct,
        }

        # Circuit breaker: a genuinely slow/unreachable/misconfigured
        # Ollama would otherwise pay its full timeout on EVERY generation
        # (confirmed via a real run: 7 straight "didn't respond in time"
        # messages, one per generation, ~90s+ each wasted for nothing).
        # After 2 consecutive failures, stop trying for the rest of this
        # run and say so once -- the search still proceeds exactly as if
        # AI assist were off.
        consecutive_failures = 0
        gave_up = False

        def ai_suggest_cb(genes: list, population: list) -> list:
            nonlocal consecutive_failures, gave_up
            if gave_up:
                return []
            # Stage 4 of the quant loop framework ("analyze why the losers
            # failed, feed that back into generation") computed as plain
            # statistics over the population the GA already evaluated --
            # no extra backtests, no AI call. Only the resulting few lines
            # of text are spent as prompt tokens below, which is what
            # keeps this "systematic first, AI only where it must be."
            analysis = analyze_gene_fitness_correlation(genes, population)
            feedback_lines = analysis.summary_lines(top_n=3)

            # Same "retrieval is free, AI is only for the last step"
            # principle as the gene-fitness analysis above: a plain
            # keyword search over whatever's in the research/ folder,
            # queried on the strategy's name and its genes' own labels
            # (e.g. "EMA period", "session filter") since those are
            # exactly the terms a relevant paper would use. Costs nothing
            # extra beyond folder-mtime bookkeeping and returns [] with
            # an empty research/ folder -- identical to this feature not
            # existing at all.
            research_query = f"{display_name} {strategy.source_type} " + " ".join(g.label for g in genes)
            research_excerpts = find_relevant_excerpts(research_query, max_excerpts=2)

            result = ollama_client.suggest_parameter_adjustments(
                strategy_name=display_name,
                source_type=strategy.source_type,
                genes=genes,
                baseline_stats=baseline_stats_summary,
                prop_rules_summary=prop_rules_summary,
                failure_analysis_lines=feedback_lines,
                research_excerpts=research_excerpts,
            )
            if result.error:
                log(f"  AI assist: {result.error}")
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    gave_up = True
                    log("  AI assist: giving up after 2 consecutive failures -- "
                        "continuing the search without it for the rest of this run.")
                return []
            consecutive_failures = 0
            if feedback_lines and not analysis.note:
                log(f"  AI assist: seeded with {len(feedback_lines)} observed parameter pattern(s) from this search.")
            return result.genomes

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
            ai_suggest_cb=ai_suggest_cb,
            parallel=cfg.parallel_search,
            max_workers=cfg.parallel_search_max_workers,
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
        log("Step 3/7: Final validation (full backtest, prop simulation, Monte Carlo)...")
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
        log("Step 4/7: Out-of-sample fold check (same configuration, no further tuning)...")
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
        log("Step 5/7: Out-of-sample holdout check...")
        try:
            final_holdout = run_holdout_comparison(df, final_strategy, risk, holdout_frac=cfg.holdout_frac)
        except Exception:
            final_holdout = None
            log("  Holdout check skipped (not enough data to split).")

        # -- Step 6: ICIR / signal-decay / Bonferroni-corrected gate ------
        # The quant loop framework's "out-of-sample gate": scores the
        # strategy's directional signal with the standard IC/ICIR metric,
        # checks whether its predictive power decays too fast to trade,
        # and requires the result to still be statistically significant
        # after correcting for how many candidates the GA actually tried
        # (Bonferroni). All pure arithmetic over trades already produced
        # above -- no AI, no extra network calls, no extra backtests
        # beyond the same in-sample/holdout split run_holdout_comparison
        # just used. See app.validation.icir.
        log("Step 6/7: ICIR / signal-decay / Bonferroni-corrected significance gate...")
        icir_gate = None
        icir_gate_skip_reason = None
        try:
            n_candidates_tested = 1  # the baseline itself always counts as one candidate tried
            if refinement_ran and ga_result is not None:
                n_candidates_tested = cfg.ga_population * (cfg.ga_generations + 1)
            icir_gate = run_icir_gate_from_backtest(
                df, final_strategy, risk, n_tests=n_candidates_tested, holdout_frac=cfg.holdout_frac,
            )
            log(f"  {'PASSED' if icir_gate.ok else 'DID NOT PASS'} "
                f"(Bonferroni-corrected for {n_candidates_tested} candidate(s) tried).")
            for reason in icir_gate.reasons:
                log(f"    {reason}")
        except Exception as exc:  # noqa: BLE001 -- best-effort validation step
            icir_gate_skip_reason = f"ICIR gate failed to run: {exc}"
            log(f"  {icir_gate_skip_reason}")

        # -- Step 7: report + save -----------------------------------------
        log("Step 7/7: Generating final report...")
        verdict, verdict_reasons = _make_verdict(final_mc, oos_validation, icir_gate)

        elapsed = time.time() - t0
        return _finish(
            strategy, display_name, baseline_bt, baseline_single_run, baseline_mc, lookahead_summary,
            refinement_ran, refinement_skip_reason, ga_result,
            final_source_type, final_config, final_code_text, final_code_ext,
            final_bt, final_single_run, final_mc, final_holdout,
            oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
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
    oos_validation, oos_skip_reason, icir_gate, icir_gate_skip_reason, verdict, verdict_reasons,
    df, prop_rules, risk, cfg, elapsed, warnings, log, output_dir,
    instrument="unknown",
) -> FullPipelineResult:
    """Writes the report + (for code strategies) saves the winner into the
    Strategy Library. Split out of run_full_pipeline only to keep that
    function's main try/finally block readable."""
    from app.reports.generator import generate_full_report

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    final_strategy_name = f"{display_name} (Full Pipeline)"

    final_parameters = None
    if ga_result is not None and ga_result.genes:
        final_parameters = {
            gene.label: (str(int(round(value))) if gene.is_int else f"{value:.4f}".rstrip("0").rstrip("."))
            for gene, value in zip(ga_result.genes, ga_result.best.genome)
        }

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
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        final_parameters=final_parameters,
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
            set_strategy_status(final_source_type, filename, cfg.library_status or _VERDICT_TO_LIBRARY_STATUS.get(verdict, "tested_passed"))
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
        icir_gate=icir_gate,
        icir_gate_skip_reason=icir_gate_skip_reason,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        saved_library_path=saved_library_path,
        saved_library_note=saved_library_note,
        report_paths=report_paths,
        elapsed_seconds=elapsed,
        warnings=warnings,
    )
