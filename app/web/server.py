"""
T58 Quant Algo Backtester — Mobile Web App.

A thin Flask front end over the exact same engine used by the desktop GUI
and the --cli entry point (app/backtest, app/prop, app/monte_carlo,
app/strategy, app/reports) -- no logic is duplicated here.

This is what makes the tool usable "as an app" on a phone: any phone
browser on the same network as the machine running this server can open
it, and "Add to Home Screen" installs it as an installable PWA (manifest +
service worker included) with its own icon and a standalone window, no App
Store / Play Store submission required.

Run:
    python -m app.web.server
    -> serves on http://0.0.0.0:5000
    -> on your phone (same Wi-Fi), open http://<your-computer's-LAN-IP>:5000

To make it reachable from anywhere (not just local Wi-Fi), deploy this
Flask app to any small host (Render, Railway, Fly.io, a VPS, etc.) -- see
README.md for notes. Running a persistent Python server directly on a
phone (as opposed to browsing to one) is out of scope for this MVP.
"""
from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for,
)

from app.ai.ollama_settings import OllamaSettings
from app.web.network_info import lan_url, print_startup_banner, qr_code_data_uri, qr_code_file
from app.ai.ollama_settings import load_settings as load_ollama_settings
from app.ai.ollama_settings import save_settings as save_ollama_settings
from app.ai.research_agent import ResearchAgentContext, ResearchAgent
from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.data import alpaca_credentials
from app.data.alpaca_source import (
    ASSET_CLASSES, ADJUSTMENT_CHOICES, FEED_CHOICES, TIMEFRAME_LABELS,
    AlpacaFetchError, AlpacaImportError, fetch_bars, save_bars_as_csv,
)
from app.data.importer import import_csv, import_csv_bytes
from app.data.storage import get_app_base_dir, get_raw_data_dir, list_datasets_by_instrument, list_stored_datasets, store_csv_bytes
from app.ensemble.ensemble import EnsembleError, EnsembleVoteConfig, run_ensemble_blend, run_ensemble_vote
from app.evolution import checkpoint as evo_checkpoint
from app.evolution.engine import EvolutionConfig, EvolutionRunner
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.risk_sweep import DEFAULT_RISK_VALUES, run_risk_sweep
from app.optimize.multi_objective import (    DEFAULT_OBJECTIVES, MultiObjectiveConfig, OBJECTIVE_DIRECTIONS, run_multi_objective_refinement,
)
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig, RefinementError, run_iterative_refinement
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_FULL_PIPELINE, JOB_SEARCH_LAB, JOB_SPEED_RUN,
    JOB_WFO, JOB_WFGA, JOB_CPCV, JOB_SENSITIVITY, JOB_MULTI_OBJECTIVE, JOB_REGIME_MATRIX,
)
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, run_speed_run
from app.orchestration.speed_run import _rank_key as _speedrun_rank_key
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioError, run_portfolio_backtest
from app.prop.simulator import PropRules, simulate_account
from app.prop.survival_engine import PropSurvivalConfig, ResetEconomics, run_prop_survival_analysis
from app.reports.generator import generate_full_report
from app.reports.crash_log import install_thread_excepthook, log_crash
from app.reports.refinement_report import generate_refinement_report
from app.reports.survival_report import generate_survival_report
from app.reports.validation_reports import (
    generate_cpcv_report, generate_multi_objective_report, generate_portfolio_report,
    generate_sensitivity_report, generate_walk_forward_report, generate_walkforward_ga_report,
)
from app.reports import run_history
from app.reports import strategy_state
from app.scoring.t58_scorecard import score_from_results
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.family_diversity import render_family_report, summarize_family_performance
from app.search.search_report import generate_search_report
from app.search.strategy_space import (
    StrategySpaceError, family_description, generate_search_space, list_families,
)
from app.search.results_db import ResultsDB
from app.strategy.base import StrategyError
from app.validation.cpcv import CPCVError, run_cpcv
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity
from app.validation.walk_forward_opt import run_walk_forward_optimization
from app.strategy.library import (
    STRATEGY_STATUSES, STRATEGY_TYPES, StrategyAlreadyExists, delete_many,
    delete_saved_strategy, export_library_zip_bytes, list_all_markets, list_all_tags,
    list_saved_strategies, load_strategy_text, record_backtest_result, record_lookahead_result,
    record_search_result, rename_saved_strategy, save_strategy_bytes, save_strategy_metadata,
    save_strategy_text, set_strategy_status, set_strategy_tags,
)
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy
from app.web import live_market

# get_app_base_dir() already knows how to find a persistent, writable
# folder next to the running .exe when frozen (see app/data/storage.py),
# vs. the repo root during normal development. Reusing it here (instead
# of the old `Path(__file__).resolve().parent.parent.parent`) matters
# specifically for the packaged web-app exe: PyInstaller --onefile runs
# code from a temporary extraction folder that's deleted on exit, so the
# old path would silently drop every report the moment the app closed.
BASE_DIR = get_app_base_dir()
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIR = BASE_DIR / "reports" / "search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)
REFINEMENT_DIR = BASE_DIR / "reports" / "refinement"
REFINEMENT_DIR.mkdir(parents=True, exist_ok=True)
FULL_PIPELINE_DIR = BASE_DIR / "reports" / "full_pipeline"
FULL_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
WFO_DIR = BASE_DIR / "reports" / "walk_forward_opt"
WFO_DIR.mkdir(parents=True, exist_ok=True)
MULTI_OBJ_DIR = BASE_DIR / "reports" / "multi_objective"
MULTI_OBJ_DIR.mkdir(parents=True, exist_ok=True)
WFGA_DIR = BASE_DIR / "reports" / "walkforward_ga"
WFGA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_DIR = BASE_DIR / "reports" / "portfolio"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR = BASE_DIR / "reports" / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
CPCV_DIR = BASE_DIR / "reports" / "cpcv"
CPCV_DIR.mkdir(parents=True, exist_ok=True)
SENSITIVITY_DIR = BASE_DIR / "reports" / "sensitivity"
SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
QUICK_OPT_DIR = BASE_DIR / "reports" / "quick_optimize"
QUICK_OPT_DIR.mkdir(parents=True, exist_ok=True)
PAYOUT_DIR = BASE_DIR / "reports" / "payout_probability"
PAYOUT_DIR.mkdir(parents=True, exist_ok=True)
REGIME_DIR = BASE_DIR / "reports" / "regime_matrix"
REGIME_DIR.mkdir(parents=True, exist_ok=True)
SPEEDRUN_DIR = BASE_DIR / "reports" / "speed_run"
SPEEDRUN_DIR.mkdir(parents=True, exist_ok=True)
# run_speed_run() itself writes each validated candidate's Full Pipeline
# report into output_dir / "speed_run" -- see app.orchestration.speed_run.
SPEEDRUN_REPORTS_DIR = SPEEDRUN_DIR / "speed_run"
SPEEDRUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
# Belt-and-suspenders alongside run_web.py's own call (this module can also
# be run directly via `python -m app.web.server`, which never goes through
# run_web.py) -- idempotent either way. See app.reports.crash_log.
install_thread_excepthook()


def resolve_strategy_source(mode: str, form, files) -> tuple[str, str]:
    """Single source of truth for "where did this strategy's code come
    from" -- an uploaded file, a saved-library pick, or pasted text.
    Returns (code, suggested_save_name). Any future entry point that needs
    a strategy's source text (not just /run and /search/start) should call
    this instead of re-implementing the precedence rules.

    Precedence: a freshly uploaded file wins over pasted text (the page's
    JS also mirrors an upload into the textarea, so this only matters if
    JS didn't run), and pasted/uploaded text wins over a saved-library pick
    -- matches _resolve_dataset's "most recent upload wins" pattern."""
    code = (form.get("strategy_code") or "").strip()
    existing_choice = (form.get(f"existing_strategy_{mode}") or "").strip()
    uploaded = files.get("strategy_file")
    save_name = (form.get("strategy_save_name") or "").strip()

    if uploaded and uploaded.filename:
        code = uploaded.read().decode("utf-8", errors="replace")
        if not save_name:
            save_name = uploaded.filename

    library_ref = None
    if not code and existing_choice:
        code = load_strategy_text(mode, existing_choice)
        library_ref = existing_choice

    if not code:
        raise StrategyError(
            f"Paste your {mode} strategy code, upload a file, or choose a saved strategy from the library."
        )
    return code, (save_name or library_ref or "strategy")


def maybe_save_strategy_to_library(mode: str, code: str, save_name: str, form) -> str | None:
    """Single source of truth for "save this strategy's code to the
    library", including the overwrite-vs-duplicate prompt and optional
    description/market/tags. Returns the saved filename, or None if the
    "save to library" checkbox wasn't checked. Any future entry point that
    wants to offer a save-to-library option should call this instead of
    re-implementing it."""
    if not form.get("strategy_save_to_library"):
        return None

    overwrite = bool(form.get("strategy_overwrite"))
    try:
        saved_path = save_strategy_text(code, save_name, mode, overwrite=overwrite)
    except StrategyAlreadyExists as exc:
        raise StrategyError(
            f"'{exc.filename}' is already in the {mode} strategy library. "
            'Check "Overwrite existing" and run again to replace it, or change "Save as" to a new name.'
        ) from exc

    description = (form.get("strategy_save_description") or "").strip()
    market = (form.get("strategy_save_market") or "").strip()
    tags_raw = (form.get("strategy_save_tags") or "").strip()
    meta_update = {}
    if description:
        meta_update["description"] = description
    if market:
        meta_update["market"] = market
    if meta_update:
        save_strategy_metadata(mode, saved_path.name, meta_update)
    if tags_raw:
        set_strategy_tags(mode, saved_path.name, [t.strip() for t in tags_raw.split(",")])
    return saved_path.name


def build_strategy_from_code(mode: str, code: str):
    """Single source of truth for turning resolved source text into a
    Strategy object, once code is known (see resolve_strategy_source)."""
    if mode == "python":
        tmp = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
        tmp.write_text(code, encoding="utf-8")
        return PythonStrategy(tmp)
    if mode == "pinescript":
        return PineScriptStrategy(code)
    if mode == "mql5":
        return MQL5Strategy(code)
    raise StrategyError(f"Unknown strategy mode: {mode}")


def _try_acquire_heavy_job(job_name: str, template_name: str, **template_ctx):
    """Shared guard check for every long-running background tool (not just
    the four ProcessPoolExecutor-based ones) -- see resource_guard.py's
    module docstring for why WFO/WFGA/CPCV/Sensitivity/Multi-Objective/
    Regime Matrix are included here too. Returns None if the job may
    proceed; otherwise a ready-to-return (response, 409) tuple refusing it,
    rendered with the same template/context the caller's own error paths
    already use, so the person sees an ordinary in-page error rather than a
    generic 409 page."""
    if HEAVY_JOB_GUARD.try_acquire(job_name):
        return None
    msg = (
        f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than one "
        f"long, heavy job at the same time can exhaust available memory or CPU and is a common "
        f"cause of the app becoming unresponsive or crashing. Wait for it to finish (or stop it) "
        f"before starting {job_name}."
    )
    return render_template(template_name, error=msg, **template_ctx), 409


def _build_strategy(mode: str, form, files):
    """Thin orchestrator over resolve_strategy_source / build_strategy_from_code
    / maybe_save_strategy_to_library -- kept as the one call /run and
    /search/start both use, but each piece is independently reusable (see
    their own docstrings) so a future entry point isn't stuck copy-pasting
    this. Returns (strategy, library_ref) where library_ref is
    (mode, filename) if the strategy is tied to a saved library entry
    (loaded from it, or just saved to it) -- used to stamp lookahead/search
    results back onto that entry's metadata -- or None for manual/one-off
    pasted strategies with no library tie."""
    if mode == "manual":
        cfg = {
            "name": "Manual Strategy (web)",
            "indicators": [
                {"type": "sma", "period": int(form.get("sma_fast", 20)), "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": int(form.get("sma_slow", 50)), "column": "close", "as": "sma_slow"},
            ],
            "long_entry": "sma_fast > sma_slow",
            "long_exit": "sma_fast < sma_slow",
            "short_entry": "sma_fast < sma_slow",
            "short_exit": "sma_fast > sma_slow",
            "stop_loss_pips": float(form.get("sl_pips", 20)),
            "take_profit_pips": float(form.get("tp_pips", 40)),
        }
        return ManualStrategy(cfg), None

    code, save_name = resolve_strategy_source(mode, form, files)
    existing_choice = (form.get(f"existing_strategy_{mode}") or "").strip()

    saved_name = maybe_save_strategy_to_library(mode, code, save_name, form)
    library_ref = (mode, saved_name) if saved_name else ((mode, existing_choice) if existing_choice else None)

    return build_strategy_from_code(mode, code), library_ref


def _resolve_dataset(form, files):
    """
    Shared by /run and /search/start: picks the active DataFrame from
    either newly-uploaded CSV file(s) or a previously-stored dataset,
    exactly the same precedence /run has always used (most recently
    uploaded valid file wins over a selected stored dataset). Returns
    (df, label, import_note, error_message) -- df is None and
    error_message is set if nothing usable was provided.
    """
    uploaded_files = [f for f in files.getlist("csv_file") if f and f.filename]
    existing_choice = (form.get("existing_dataset") or "").strip()

    imported_names, failed = [], []
    active_df = None
    active_label = None

    for f in uploaded_files:
        content = f.read()
        result = import_csv_bytes(content)
        if not result.is_valid:
            failed.append((f.filename, "; ".join(result.errors)))
            continue
        store_csv_bytes(content, f.filename)
        imported_names.append(f.filename)
        active_df = result.dataframe  # most recently imported valid file becomes active
        active_label = f.filename

    if active_df is None and existing_choice:
        candidate = get_raw_data_dir() / existing_choice
        if candidate.exists():
            result = import_csv(candidate)
            if result.is_valid:
                active_df = result.dataframe
                active_label = existing_choice

    if active_df is None:
        msg = "Please upload at least one valid CSV, or choose a previously stored dataset."
        if failed:
            msg += " Failed upload(s): " + "; ".join(f"{n} ({e})" for n, e in failed)
        return None, None, None, msg

    import_note = None
    if imported_names:
        import_note = f"Stored {len(imported_names)} file(s) in data/raw/: {', '.join(imported_names)}."
        if failed:
            import_note += f" {len(failed)} file(s) failed and were skipped: " + \
                "; ".join(f"{n} ({e})" for n, e in failed)

    return active_df, active_label, import_note, None




def _saved_strategies_json() -> str:
    """{"python": [{"name", "description", "market", "tags", "status",
    "last_run", "lookahead", "last_search"}, ...], "pinescript": [...],
    "mql5": [...]} for the page's JS to build the "load from library"
    dropdowns, the client-side search/tag/market/status filters, and to
    prefill fields when a saved strategy is picked."""
    return json.dumps({
        t: [
            {
                "name": s.name,
                "description": s.metadata.get("description", ""),
                "market": s.metadata.get("market", ""),
                "tags": s.tags,
                "status": s.status,
                "last_run": s.metadata.get("last_run"),
                "lookahead": s.metadata.get("lookahead"),
                "last_search": s.metadata.get("last_search"),
            }
            for s in list_saved_strategies(t)
        ]
        for t in STRATEGY_TYPES
    })


def _alpaca_template_context() -> dict:
    """Shared context injected wherever the Market Data card is rendered
    (index.html today) -- the dropdown/option lists plus whether keys are
    already saved, so the form can pre-check "save keys" and (for privacy)
    never echo a saved secret back into the page source."""
    return {
        "alpaca_asset_classes": ASSET_CLASSES,
        "alpaca_timeframes": TIMEFRAME_LABELS,
        "alpaca_feeds": FEED_CHOICES,
        "alpaca_adjustments": ADJUSTMENT_CHOICES,
        "alpaca_has_saved_keys": alpaca_credentials.has_saved_credentials(),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
        alpaca_notice=request.args.get("alpaca_notice"),
        alpaca_notice_kind=request.args.get("alpaca_notice_kind", "info"),
        strategy_statuses=STRATEGY_STATUSES,
        **_alpaca_template_context(),
    )


@app.route("/data/alpaca/fetch", methods=["POST"])
def data_alpaca_fetch():
    """Fetches bars from Alpaca and saves them into data/raw/<SYMBOL>/,
    same as the desktop app's FETCH & SAVE button. A plain form POST (not
    AJAX) to stay consistent with the rest of this page and to keep
    working with JS disabled; redirects back to '/' with a short notice."""
    form = request.form
    api_key = (form.get("alpaca_api_key") or "").strip()
    secret_key = (form.get("alpaca_secret_key") or "").strip()
    save_keys = form.get("alpaca_save_keys") == "on"

    # A blank, masked secret field means "keep using the saved one" -- the
    # page never echoes a real saved secret back into the HTML.
    if not api_key or not secret_key:
        saved = alpaca_credentials.load_credentials()
        if saved:
            api_key = api_key or saved.api_key
            secret_key = secret_key or saved.secret_key

    symbols = [s.strip() for s in (form.get("alpaca_symbols") or "").split(",") if s.strip()]
    asset_class = form.get("alpaca_asset_class") or ASSET_CLASSES[0]
    timeframe_label = form.get("alpaca_timeframe") or TIMEFRAME_LABELS[0]
    start = (form.get("alpaca_start") or "").strip()
    end = (form.get("alpaca_end") or "").strip()
    feed = form.get("alpaca_feed") or FEED_CHOICES[0]
    adjustment = form.get("alpaca_adjustment") or ADJUSTMENT_CHOICES[0]

    if not api_key or not secret_key:
        return redirect(url_for("index", alpaca_notice="Enter both an API key and a secret key.", alpaca_notice_kind="error"))
    if not symbols:
        return redirect(url_for("index", alpaca_notice="Enter at least one symbol.", alpaca_notice_kind="error"))

    if save_keys:
        alpaca_credentials.save_credentials(api_key, secret_key)

    saved_names, errors = [], []
    for symbol in symbols:
        try:
            df = fetch_bars(
                api_key, secret_key, symbol, asset_class, timeframe_label, start, end,
                feed=feed, adjustment=adjustment,
            )
            dest = save_bars_as_csv(df, symbol, timeframe_label)
            saved_names.append(dest.relative_to(get_raw_data_dir()).as_posix())
        except (AlpacaImportError, AlpacaFetchError) as exc:
            errors.append(f"{symbol}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{symbol}: unexpected error ({exc})")

    if saved_names and not errors:
        notice, kind = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}.", "success"
    elif saved_names and errors:
        notice = f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}. Failed: {'; '.join(errors)}"
        kind = "warning"
    else:
        notice, kind = f"Fetch failed: {'; '.join(errors)}", "error"

    return redirect(url_for("index", alpaca_notice=notice, alpaca_notice_kind=kind))


@app.route("/data/alpaca/forget", methods=["POST"])
def data_alpaca_forget():
    alpaca_credentials.clear_credentials()
    return redirect(url_for("index", alpaca_notice="Saved Alpaca keys removed from this computer.", alpaca_notice_kind="success"))


@app.route("/mobile-access")
def mobile_access():
    """Shows the same LAN address + QR code the console banner prints
    (see app.web.network_info), but IN the running app itself -- so it's
    reachable from a browser tab on the PC (e.g. after starting via
    `python -m app.web.server`, which used to show none of this), not
    only from the separate run_web.py launcher's popped-open image."""
    url = lan_url()
    qr_data_uri = qr_code_data_uri(url)
    return render_template(
        "mobile_access.html", active_page="mobile_access", url=url, qr_data_uri=qr_data_uri,
    )


@app.route("/dashboard")
def dashboard():
    current = strategy_state.get_current_strategy()
    checklist = None
    score = None
    if current:
        checklist = strategy_state.get_checklist(current["strategy_name"], current["instrument"])
        score = strategy_state.robustness_score(current["strategy_name"], current["instrument"])
    return render_template(
        "dashboard.html",
        data=run_history.dashboard_data(),
        dataset_groups=list_datasets_by_instrument(),
        current_strategy=current,
        checklist=checklist,
        robustness=score,
        validation_labels=strategy_state.VALIDATION_LABELS,
        validation_hrefs=strategy_state.VALIDATION_HREFS,
        validation_kinds=strategy_state.VALIDATION_KINDS,
    )


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """JSON feed the dashboard page polls to refresh live, without a full
    page reload, whenever a run finishes (desktop, web, or CLI)."""
    return jsonify(run_history.dashboard_data())


@app.route("/current-strategy/set", methods=["POST"])
def set_current_strategy():
    """Explicit only -- the person picks a strategy off the Dashboard
    scorecard (or the Champion card) and marks it current. Nothing here
    infers a current strategy from just running a backtest, so a stray
    one-off test never silently hijacks the checklist."""
    form = request.form
    name = (form.get("strategy_name") or "").strip()
    instrument = (form.get("instrument") or "").strip()
    timeframe = (form.get("timeframe") or "").strip()
    if name and instrument:
        strategy_state.set_current_strategy(name, instrument, timeframe)
    return redirect(url_for("dashboard"))


@app.route("/current-strategy/clear", methods=["POST"])
def clear_current_strategy():
    strategy_state.clear_current_strategy()
    return redirect(url_for("dashboard"))


@app.route("/optimize")
def optimize_hub():
    """Guided picker for the OPTIMIZE stage -- every method underneath is
    still its own full page (nothing lost), this just answers "which one
    should I use" before sending the person on to it."""
    current = strategy_state.get_current_strategy()
    return render_template("optimize_hub.html", active_page="optimize_hub", current_strategy=current)


@app.route("/validate")
def validate_hub():
    """Guided checklist for the VALIDATE stage, built entirely from data
    already recorded by CPCV / WFO / WFGA / Sensitivity / Regime Matrix --
    nothing here re-runs or duplicates those tools, it just aggregates what
    they've already found for whichever strategy is marked current."""
    current = strategy_state.get_current_strategy()
    checklist = None
    score = None
    if current:
        checklist = strategy_state.get_checklist(current["strategy_name"], current["instrument"])
        score = strategy_state.robustness_score(current["strategy_name"], current["instrument"])
    return render_template(
        "validate_hub.html", active_page="validate_hub", current_strategy=current,
        checklist=checklist, robustness=score,
        validation_labels=strategy_state.VALIDATION_LABELS,
        validation_hrefs=strategy_state.VALIDATION_HREFS,
        validation_kinds=strategy_state.VALIDATION_KINDS,
    )


@app.route("/strategies/<strategy_type>/<path:filename>")
def get_saved_strategy(strategy_type, filename):
    """Raw source text of one saved strategy, for the page's JS to pull into
    the strategy_code textarea when the person picks it from the library."""
    try:
        text = load_strategy_text(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(text, mimetype="text/plain")


def _redirect_target(form) -> str:
    return "search_form" if form.get("return_to") == "search" else "index"


@app.route("/strategies/delete", methods=["POST"])
def delete_saved_strategy_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    notice = f"Deleted '{filename}' from the {strategy_type} library."
    try:
        delete_saved_strategy(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        notice = str(exc)
    return redirect(url_for(_redirect_target(request.form), strategy_notice=notice))


@app.route("/strategies/bulk-delete", methods=["POST"])
def bulk_delete_saved_strategies_route():
    """Delete every "type::filename" item checked in the library's bulk-
    manage panel in one request, instead of one delete round-trip each."""
    items = _parse_bulk_items(request.form.getlist("items"))
    target = _redirect_target(request.form)
    if not items:
        return redirect(url_for(target, strategy_notice="No strategies were selected to delete."))
    deleted, failed = delete_many(items)
    notice = f"Deleted {len(deleted)} strategy(ies)."
    if failed:
        notice += f" {len(failed)} failed: " + "; ".join(failed)
    return redirect(url_for(target, strategy_notice=notice))


def _parse_bulk_items(raw_items: list[str]) -> list[tuple[str, str]]:
    """Bulk checkboxes post as "type::filename" strings (see the
    bulk-manage checkboxes in index.html/search.html) -- parse and drop
    anything malformed rather than letting one bad value 500 the request."""
    items = []
    for raw in raw_items:
        parts = raw.split("::", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            items.append((parts[0], parts[1]))
    return items


@app.route("/strategies/bulk-export")
def bulk_export_saved_strategies_route():
    """Download only the checked "type::filename" items as a zip (bulk
    export a subset), via query string (?items=python::a.py&items=...) so
    it can be a plain link like the full-library export."""
    items = _parse_bulk_items(request.args.getlist("items"))
    if not items:
        return Response("No strategies were selected to export.", status=400, mimetype="text/plain")
    try:
        data = export_library_zip_bytes(selection=items)
    except FileNotFoundError as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(
        data,
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=t58_strategy_library_selection.zip"},
    )


@app.route("/strategies/view-code")
def view_saved_strategy_code_route():
    """Plain-text view of one saved strategy's source -- the web
    equivalent of the desktop app's VIEW CODE / CONFIG button. Opens in a
    new tab from the library panel rather than a modal, since a phone
    browser handles a plain page far better than in-page JS overlay text."""
    strategy_type = request.args.get("strategy_type", "")
    filename = request.args.get("filename", "")
    try:
        text = load_strategy_text(strategy_type, filename)
    except (ValueError, FileNotFoundError) as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(text, mimetype="text/plain")


@app.route("/strategies/batch-test", methods=["POST"])
def batch_test_saved_strategies_route():
    """Runs every checked "type::filename" library item through the same
    backtest -> prop-sim -> Monte Carlo -> report pipeline /run uses, one
    after another -- the web equivalent of the desktop app's TEST
    SELECTED (BATCH) button. Reuses the SAME market-data / risk / prop
    rule fields the main Run & Report form already carries (the page's JS
    clones them into this request), with one exception: a freshly
    uploaded CSV can't be cloned this way by a browser for security
    reasons, so batch-testing from the web currently requires picking an
    already-stored dataset rather than uploading a brand new file in the
    same click -- upload it once via Run & Report first, then batch-test
    against it."""
    form = request.form
    items = _parse_bulk_items(form.getlist("batch_items"))
    if not items:
        return redirect(url_for("index", strategy_notice="No strategies were checked to batch-test."))

    df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
    if dataset_error:
        return redirect(url_for("index", strategy_notice=dataset_error))

    risk = RiskConfig(
        initial_balance=float(form.get("initial_balance", 100000)),
        risk_mode=form.get("risk_mode", "percent"),
        risk_value=float(form.get("risk_value", 1.0)),
        max_trades_per_day=int(form.get("max_trades_day", 10)),
        commission_per_trade=float(form.get("commission", 0)),
        slippage_pips=float(form.get("slippage_pips", 0.5)),
        spread_pips=float(form.get("spread_pips", 1.0)),
        pip_size=float(form.get("pip_size", 0.0001)),
    )
    payout_cap = form.get("payout_cap", "").strip()
    rules = PropRules(
        account_size=float(form.get("account_size", 100000)),
        evaluation_profit_target_pct=float(form.get("profit_target", 8)),
        daily_loss_limit_pct=float(form.get("daily_loss", 5)),
        max_drawdown_pct=float(form.get("max_dd", 10)),
        drawdown_type=form.get("dd_type", "trailing"),
        drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
        consistency_rule_pct=float(form.get("consistency", 30)) if form.get("consistency") else None,
        min_trading_days=int(form.get("min_days", 5)),
        payout_threshold_pct=float(form.get("payout_threshold", 0)),
        payout_cap_pct=float(payout_cap) if payout_cap else None,
        payout_frequency_days=int(form.get("payout_freq", 14)),
        required_buffer_pct=float(form.get("buffer", 0)),
    )
    n_sims = int(form.get("n_sims", 5000))
    mc_method = form.get("mc_method", "bootstrap")

    batch_items = []
    load_errors = []
    for strategy_type, filename in items:
        try:
            code = load_strategy_text(strategy_type, filename)
            strategy = build_strategy_from_code(strategy_type, code)
        except Exception as exc:  # noqa: BLE001 -- one bad strategy must not stop the batch
            load_errors.append(f"{filename}: {exc}")
            continue
        batch_items.append(BatchTestItem(label=filename, strategy=strategy, library_ref=(strategy_type, filename)))

    if not batch_items:
        return redirect(url_for("index", strategy_notice="Every checked strategy failed to load: " + "; ".join(load_errors)))

    run_id = uuid.uuid4().hex[:8]
    summary = run_batch_test(
        df, batch_items, risk, rules, REPORTS_DIR,
        instrument=active_label, mc_sims=min(n_sims, 50_000), mc_method=mc_method,
        basename_prefix=f"webbatch_{run_id}",
    )

    return render_template(
        "index.html",
        stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
        **_alpaca_template_context(),
        batch_result={
            "active_dataset": active_label,
            "import_note": import_note,
            "outcomes": [
                {
                    "label": o.label, "ok": o.ok, "reason": o.reason, "trades": o.trades,
                    "net_profit": o.net_profit, "eval_pass_probability": o.eval_pass_probability,
                    "report_html": f"/reports/{o.report_html.name}" if o.report_html else None,
                }
                for o in summary.outcomes
            ],
            "load_errors": load_errors,
        },
    )


@app.route("/strategies/rename", methods=["POST"])
def rename_saved_strategy_route():
    strategy_type = request.form.get("strategy_type", "")
    old_filename = request.form.get("old_filename", "")
    new_filename = (request.form.get("new_filename") or "").strip()
    overwrite = bool(request.form.get("overwrite"))
    target = _redirect_target(request.form)

    if not new_filename:
        return redirect(url_for(target, strategy_notice="Enter a new filename to rename to."))
    try:
        new_path = rename_saved_strategy(strategy_type, old_filename, new_filename, overwrite=overwrite)
        notice = f"Renamed '{old_filename}' to '{new_path.name}'."
    except StrategyAlreadyExists as exc:
        notice = (
            f"'{exc.filename}' already exists in the {strategy_type} library. "
            'Check "Overwrite" and rename again to replace it.'
        )
    except (ValueError, FileNotFoundError) as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/metadata", methods=["POST"])
def save_strategy_metadata_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    description = (request.form.get("description") or "").strip()
    market = (request.form.get("market") or "").strip()
    tags_raw = (request.form.get("tags") or "").strip()
    target = _redirect_target(request.form)
    try:
        save_strategy_metadata(strategy_type, filename, {"description": description, "market": market})
        set_strategy_tags(strategy_type, filename, [t.strip() for t in tags_raw.split(",")] if tags_raw else [])
        notice = f"Saved info for '{filename}'."
    except ValueError as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/status", methods=["POST"])
def set_strategy_status_route():
    strategy_type = request.form.get("strategy_type", "")
    filename = request.form.get("filename", "")
    status = request.form.get("status", "")
    target = _redirect_target(request.form)
    try:
        set_strategy_status(strategy_type, filename, status)
        notice = f"'{filename}' marked {status}."
    except ValueError as exc:
        notice = str(exc)
    return redirect(url_for(target, strategy_notice=notice))


@app.route("/strategies/export")
def export_strategy_library_route():
    """Download every saved strategy (all three languages, plus their
    metadata sidecars) as one zip -- the backup button, and also how a
    packaged .exe's library (which lives next to the .exe, not in the git
    repo) gets synced back into the repo: download, unzip into
    strategies/, commit."""
    data = export_library_zip_bytes()
    return Response(
        data,
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=t58_strategy_library_backup.zip"},
    )


@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json", mimetype="application/manifest+json")


@app.route("/run", methods=["POST"])
def run_pipeline():
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(request.form, request.files)
        if dataset_error:
            return render_template("index.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        form = request.form
        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        payout_cap = form.get("payout_cap", "").strip()
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
            drawdown_type=form.get("dd_type", "trailing"),
            drawdown_check_mode=form.get("dd_check_mode", "intrabar"),
            consistency_rule_pct=float(form.get("consistency", 30)) if form.get("consistency") else None,
            min_trading_days=int(form.get("min_days", 5)),
            payout_threshold_pct=float(form.get("payout_threshold", 0)),
            payout_cap_pct=float(payout_cap) if payout_cap else None,
            payout_frequency_days=int(form.get("payout_freq", 14)),
            required_buffer_pct=float(form.get("buffer", 0)),
        )

        bt_result = run_backtest(df, strategy, risk)

        lookahead_warning = None
        if strategy.source_type == "python":
            try:
                lookahead_result = check_for_lookahead(strategy, df, max_signal_checkpoints=8)
                if lookahead_result.bug_detected:
                    lookahead_warning = lookahead_result.summary()
                if library_ref:
                    record_lookahead_result(*library_ref, {
                        "clean": not lookahead_result.bug_detected,
                        "summary": lookahead_result.summary(),
                    })
            except Exception:
                # Best-effort audit -- never let it block a run that would
                # otherwise succeed.
                pass

        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, rules)

        if not bt_result.trades:
            msg = (
                "No trades were generated by this strategy over the given "
                "data -- there is nothing to run a prop-firm simulation or "
                "Monte Carlo simulation on, and no report was produced. "
                "This usually means the strategy's entry conditions never "
                "fired for this data/date range rather than an app problem."
            )
            return render_template("index.html", error=msg, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400

        n_sims = int(form.get("n_sims", 5000))
        mc_cfg = MonteCarloConfig(n_simulations=min(n_sims, 50_000), method=form.get("mc_method", "bootstrap"))
        mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)

        try:
            holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
        except Exception:
            holdout_comparison = None

        run_id = uuid.uuid4().hex[:10]
        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_full_report(
            output_dir=REPORTS_DIR,
            strategy_name=bt_result.strategy_name,
            strategy_source_type=strategy.source_type,
            instrument=active_label,
            timeframe="unknown",
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            basename=f"report_{run_id}",
            holdout_comparison=holdout_comparison,
            risk_config=risk,
            price_df=df,
        )

        if library_ref:
            try:
                record_backtest_result(*library_ref, {
                    "trades": len(bt_result.trades),
                    "net_profit": round(bt_result.statistics.net_profit, 2),
                    "win_rate": round(bt_result.statistics.win_rate, 1),
                    "max_dd": round(bt_result.statistics.max_drawdown_pct, 2),
                    "passed_evaluation": single_run.passed_evaluation,
                    "report_html": f"/reports/{paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass  # strategy was renamed/deleted mid-run -- not worth failing the response over

        return render_template(
            "index.html",
            stored_datasets=list_stored_datasets(),
            saved_strategies_json=_saved_strategies_json(),
            **_alpaca_template_context(),
            result={
                "active_dataset": active_label,
                "import_note": import_note,
                "lookahead_warning": lookahead_warning,
                "trades": len(bt_result.trades),
                "net_profit": bt_result.statistics.net_profit,
                "win_rate": bt_result.statistics.win_rate,
                "max_dd": bt_result.statistics.max_drawdown_pct,
                "passed_eval": single_run.passed_evaluation,
                "reached_payout": single_run.reached_first_payout,
                "eval_pass_prob": mc_result.evaluation_pass_probability,
                "first_payout_prob": mc_result.first_payout_probability,
                "risk_of_ruin": mc_result.risk_of_ruin_pct,
                "expected_payout": mc_result.expected_payout,
                "n_sims": mc_result.n_simulations,
                "report_html": f"/reports/{paths['html'].name}",
                "report_json": f"/reports/{paths['json'].name}",
                "report_csv": f"/reports/{paths['summary_csv'].name}",
            },
        )
    except StrategyError as exc:
        return render_template("index.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **_alpaca_template_context()), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Live Market -- a real-time (or best-available) candlestick chart backed by
# TradingView's Lightweight Charts library, fed from whichever data source
# is actually available: a connected MT5 demo terminal, saved Alpaca keys,
# or (with neither) a steady bar-by-bar replay of an already-imported CSV so
# the page always has something real to draw.
# ---------------------------------------------------------------------------

@app.route("/live-market")
def live_market_page():
    try:
        mt5 = live_market.mt5_status()
        has_alpaca = bool(alpaca_credentials.load_credentials())
        replay_datasets = live_market.list_replay_datasets()
        return render_template(
            "live_market.html",
            mt5_status=mt5,
            has_alpaca=has_alpaca,
            replay_datasets=replay_datasets,
            timeframe_choices=live_market.TIMEFRAME_CHOICES_MINUTES,
            theme=request.args.get("theme", "dark"),
            initial_symbol=request.args.get("symbol") or mt5["default_symbol"] or "XAUUSD",
            initial_timeframe=request.args.get("timeframe", type=int) or mt5["default_timeframe_minutes"] or 15,
            initial_source=request.args.get("source") or ("mt5" if mt5["configured"] else ("alpaca" if has_alpaca else "replay")),
        )
    except Exception as exc:  # noqa: BLE001 -- a broken data source must never take the whole page down
        return render_template(
            "live_market.html",
            mt5_status={"available": False, "configured": False, "connected": False,
                        "default_symbol": "XAUUSD", "default_timeframe_minutes": 15},
            has_alpaca=False, replay_datasets=[], timeframe_choices=live_market.TIMEFRAME_CHOICES_MINUTES,
            theme=request.args.get("theme", "dark"), initial_symbol="XAUUSD", initial_timeframe=15,
            initial_source="replay", load_error=str(exc),
        )


@app.route("/api/live-market/status")
def api_live_market_status():
    try:
        return jsonify({
            "mt5": live_market.mt5_status(),
            "alpaca_configured": bool(alpaca_credentials.load_credentials()),
            "replay_datasets": live_market.list_replay_datasets(),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"mt5": {"available": False, "configured": False, "connected": False},
                         "alpaca_configured": False, "replay_datasets": [], "error": str(exc)}), 200


@app.route("/api/live-market/bars")
def api_live_market_bars():
    source = request.args.get("source", "replay")
    symbol = request.args.get("symbol", "")
    timeframe = request.args.get("timeframe", 15, type=int) or 15
    seed = request.args.get("seed", "0") == "1"  # first request for this symbol/timeframe this page-load

    try:
        if source == "mt5":
            bars = live_market.fetch_mt5_bars(symbol, timeframe)
            status = "live" if bars else ("connecting" if live_market.mt5_status()["configured"] else "unavailable")
        elif source == "alpaca":
            asset_class = request.args.get("asset_class", "Stock")
            bars = live_market.fetch_alpaca_bars(symbol, asset_class, timeframe)
            status = "delayed" if bars else "unavailable"
        else:
            bars, finished = live_market.fetch_replay_bars(symbol, advance=not seed)
            status = "replay-finished" if finished else "replay"
        return jsonify({"bars": bars, "status": status})
    except Exception as exc:  # noqa: BLE001 -- any data-source hiccup must surface as UNAVAILABLE, never a 500
        return jsonify({"bars": [], "status": "unavailable", "error": str(exc)}), 200


@app.route("/api/live-market/trades")
def api_live_market_trades():
    symbol = request.args.get("symbol", "")
    try:
        return jsonify({"markers": live_market.recent_trade_markers(symbol)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"markers": [], "error": str(exc)}), 200


# ---------------------------------------------------------------------------
# Search Lab (Stages 1-5) -- same engine as the desktop app's Search Lab tab.
#
# A search run can take anywhere from ~10 seconds to several minutes
# (hundreds of candidates x GA refinement x full Monte Carlo x walk-forward
# x robustness), which is too long to hold open a single HTTP request/response
# for. So this runs the search in a background thread and hands the browser
# a job_id immediately; the job status page polls a small JSON endpoint
# every couple of seconds and renders the live log + leaderboard once done --
# the same "kick off a background job, poll for status" shape any long job
# on the web needs, just backed by a plain in-memory dict since this app is
# a single-user local/LAN tool (same trust model as the rest of this server).
# ---------------------------------------------------------------------------

_SEARCH_JOBS: dict[str, dict] = {}
_SEARCH_JOBS_LOCK = threading.Lock()


def _job_log(job_id: str, msg: str) -> None:
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_search_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, space, stage_cfg: SearchStageConfig,
    instrument: str, db_path: str, library_ref: tuple[str, str] | None = None,
) -> None:
    try:
        summary = run_search(
            df, risk, rules, space, stage_cfg, db_path=db_path,
            instrument=instrument, timeframe="unknown",
            progress_cb=lambda msg: _job_log(job_id, msg),
        )
        report_paths = generate_search_report(
            output_dir=str(SEARCH_DIR), summary=summary, space=space,
            instrument=instrument, timeframe="unknown",
        )
        if library_ref and summary.leaderboard:
            try:
                record_search_result(*library_ref, {
                    "candidates_tested": summary.total_candidates,
                    "best_fitness": round(summary.leaderboard[0].get("fitness", 0), 4),
                    "fitness_metric": stage_cfg.fitness_metric,
                    "report_html": f"/search_reports/{report_paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass  # base strategy was renamed/deleted mid-search -- don't fail the job over it
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["summary"] = summary
            job["db_path"] = db_path
            job["df"] = df
            job["risk"] = risk
            job["rules"] = rules
            job["report_html"] = f"/search_reports/{report_paths['html'].name}"
            job["report_json"] = f"/search_reports/{report_paths['json'].name}"
    except Exception as exc:  # noqa: BLE001 -- a search job must fail visibly on the status page, not crash a thread silently
        log_crash("Search Lab (web)", exc=exc)
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    finally:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)


# ---------------------------------------------------------------------------
# Iterative Refinement (Step 6 on desktop) -- same "background job, poll for
# status" shape as Search Lab above, since a multi-generation GA run over a
# few hundred backtests is too slow for a single request/response cycle.
# ---------------------------------------------------------------------------

_REFINEMENT_JOBS: dict[str, dict] = {}
_REFINEMENT_JOBS_LOCK = threading.Lock()


def _refinement_job_log(job_id: str, msg: str) -> None:
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_refinement_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules,
    mc_cfg: MonteCarloConfig, cfg: RefinementConfig, active_label: str,
    library_ref: tuple[str, str] | None = None,
) -> None:
    try:
        result = run_iterative_refinement(
            df, strategy, risk, rules, mc_cfg, cfg,
            progress_cb=lambda msg: _refinement_job_log(job_id, msg),
        )
        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_refinement_report(
            output_dir=REFINEMENT_DIR, result=result,
            strategy_name=getattr(strategy, "name", "Strategy"),
            instrument=active_label, timeframe="unknown", backtest_period=period,
            basename=f"refinement_{job_id}", price_df=df,
        )
        if library_ref:
            try:
                record_backtest_result(*library_ref, {
                    "note": "iterative refinement run",
                    "best_fitness": round(result.best.fitness, 4),
                    "generations": cfg.generations,
                    "report_html": f"/refinement_reports/{paths['html'].name}",
                })
            except (FileNotFoundError, ValueError):
                pass
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/refinement_reports/{paths['html'].name}"
            job["report_json"] = f"/refinement_reports/{paths['json'].name}"
            best_file_key = "best_config_json" if "best_config_json" in paths else "best_strategy_file"
            job["best_file"] = f"/refinement_reports/{paths[best_file_key].name}"
    except RefinementError as exc:
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        with _REFINEMENT_JOBS_LOCK:
            job = _REFINEMENT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/refine")
def refine_form():
    return render_template(
        "refine.html",
        stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES,
    )


@app.route("/refine/start", methods=["POST"])
def refine_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("refine.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 2000) or 2000))

        cfg = RefinementConfig(
            enabled=True,
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            population_size=int(form.get("population_size", 10) or 10),
            generations=int(form.get("generations", 5) or 5),
            elite_count=int(form.get("elite_count", 2) or 2),
            mutation_rate=float(form.get("mutation_rate", 0.35) or 0.35),
            mutation_strength=float(form.get("mutation_strength", 0.25) or 0.25),
            random_immigrants_frac=float(form.get("random_immigrants_frac", 0.15) or 0.15),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 500) or 500),
            cost_stress_enabled=form.get("cost_stress_enabled") == "on",
            cost_stress_multiplier=float(form.get("cost_stress_multiplier", 2.0) or 2.0),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _REFINEMENT_JOBS_LOCK:
            _REFINEMENT_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_refinement_job,
            args=(job_id, df, strategy, risk, rules, mc_cfg, cfg, active_label, library_ref),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("refine_job", job_id=job_id))

    except (StrategyError, RefinementError) as exc:
        return render_template("refine.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("refine.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 500


@app.route("/refine/job/<job_id>")
def refine_job(job_id):
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
    if job is None:
        return render_template("refine_job.html", job_id=job_id, not_found=True), 404
    return render_template("refine_job.html", job_id=job_id, not_found=False)


@app.route("/refine/job/<job_id>/status.json")
def refine_job_status(job_id):
    with _REFINEMENT_JOBS_LOCK:
        job = _REFINEMENT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    generations = None
    if result is not None:
        generations = [
            {
                "generation": g.generation, "best_fitness": g.best_fitness,
                "mean_fitness": g.mean_fitness, "diversity": g.diversity,
            }
            for g in (result.generation_history or [])
        ]

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": None if result is None else {
            "baseline_fitness": result.baseline.fitness,
            "best_fitness": result.best.fitness,
            "best_generation": result.best.generation,
            "improvement_pct": (
                None if not result.baseline.fitness else
                round(100 * (result.best.fitness - result.baseline.fitness) / abs(result.baseline.fitness), 1)
            ),
            "generations_run": len(generations or []),
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            "best_file": job.get("best_file"),
        },
        "generations": generations,
    })


@app.route("/refinement_reports/<path:filename>")
def serve_refinement_report(filename):
    return send_from_directory(REFINEMENT_DIR, filename)


# ---------------------------------------------------------------------------
# Full Pipeline (Step 15 on desktop) -- the "run everything" button:
# baseline -> walk-forward-aware GA search -> re-validated final Monte Carlo
# -> OOS fold check -> holdout check -> READY/MARGINAL/NOT READY verdict.
# Same background-job/poll shape as Search Lab and Iterative Refinement --
# this is the single slowest thing the app can run.
# ---------------------------------------------------------------------------

_FULLPIPELINE_JOBS: dict[str, dict] = {}
_FULLPIPELINE_JOBS_LOCK = threading.Lock()


def _fullpipeline_job_log(job_id: str, msg: str) -> None:
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_fullpipeline_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules,
    cfg: FullPipelineConfig, active_label: str, ollama_settings: OllamaSettings | None,
) -> None:
    try:
        result = run_full_pipeline(
            df, strategy, risk, rules, FULL_PIPELINE_DIR, cfg,
            progress_cb=lambda msg: _fullpipeline_job_log(job_id, msg),
            instrument=active_label, ollama_settings=ollama_settings,
            report_basename=f"full_pipeline_{job_id}",
        )
        with _FULLPIPELINE_JOBS_LOCK:
            job = _FULLPIPELINE_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/full_pipeline_reports/{Path(result.report_paths['html']).name}"
            job["report_json"] = f"/full_pipeline_reports/{Path(result.report_paths['json']).name}"
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Full Pipeline (web)", exc=exc)
        with _FULLPIPELINE_JOBS_LOCK:
            job = _FULLPIPELINE_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


@app.route("/full-pipeline")
def full_pipeline_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "full_pipeline.html",
        stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES,
        fitness_metrics=FITNESS_METRICS,
        ai_enabled=saved_ai.enabled,
        ai_host=saved_ai.host,
        ai_model=saved_ai.model,
    )


@app.route("/full-pipeline/start", methods=["POST"])
def full_pipeline_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_FULL_PIPELINE):
        return render_template(
            "full_pipeline.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory. Wait for it to finish before starting Full Pipeline."
            ),
            stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
            fitness_metrics=FITNESS_METRICS,
        ), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
            return render_template("full_pipeline.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            max_trades_per_day=int(form.get("max_trades_day", 10)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )

        library_status_raw = (form.get("library_status") or "").strip()
        cfg = FullPipelineConfig(
            n_folds=int(form.get("n_folds", 4) or 4),
            window_mode=form.get("window_mode", "rolling"),
            ga_population=int(form.get("ga_population", 12) or 12),
            ga_generations=int(form.get("ga_generations", 6) or 6),
            ga_search_mc_sims=int(form.get("ga_search_mc_sims", 200) or 200),
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            final_mc_sims=int(form.get("final_mc_sims", 10000) or 10000),
            baseline_mc_sims=int(form.get("baseline_mc_sims", 2000) or 2000),
            holdout_frac=float(form.get("holdout_frac", 0.2) or 0.2),
            oos_check_folds=int(form.get("oos_check_folds", 4) or 4),
            random_seed=int(form.get("random_seed", 42) or 42),
            save_to_library=form.get("save_to_library") == "on",
            library_status=library_status_raw or None,
            parallel_search=form.get("parallel_search", "on") == "on",
        )

        ollama_settings = None
        if form.get("ai_enabled") == "on":
            ollama_settings = OllamaSettings(
                enabled=True,
                host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
                model=form.get("ai_model", "llama3.1") or "llama3.1",
            )
            try:
                save_ollama_settings(ollama_settings)  # persists, same as the desktop tab's own checkbox
            except Exception:
                pass  # best-effort -- a save failure shouldn't block the run itself

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _FULLPIPELINE_JOBS_LOCK:
            _FULLPIPELINE_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_fullpipeline_job,
            args=(job_id, df, strategy, risk, rules, cfg, active_label, ollama_settings),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("full_pipeline_job", job_id=job_id))

    except StrategyError as exc:
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        return render_template("full_pipeline.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)
        log_crash("Full Pipeline (web, start)", exc=exc)
        return render_template("full_pipeline.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 500


@app.route("/full-pipeline/job/<job_id>")
def full_pipeline_job(job_id):
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
    if job is None:
        return render_template("full_pipeline_job.html", job_id=job_id, not_found=True), 404
    return render_template("full_pipeline_job.html", job_id=job_id, not_found=False)


@app.route("/full-pipeline/job/<job_id>/status.json")
def full_pipeline_job_status(job_id):
    with _FULLPIPELINE_JOBS_LOCK:
        job = _FULLPIPELINE_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "verdict": result.verdict,
            "verdict_reasons": result.verdict_reasons,
            "baseline_trades": len(result.baseline_bt.trades),
            "baseline_net_profit": result.baseline_bt.statistics.net_profit,
            "final_trades": len(result.final_bt.trades),
            "final_net_profit": result.final_bt.statistics.net_profit,
            "final_win_rate": result.final_bt.statistics.win_rate,
            "final_max_dd": result.final_bt.statistics.max_drawdown_pct,
            "eval_pass_probability": result.final_mc.evaluation_pass_probability,
            "first_payout_probability": result.final_mc.first_payout_probability,
            "risk_of_ruin_pct": result.final_mc.risk_of_ruin_pct,
            "refinement_ran": result.refinement_ran,
            "refinement_skip_reason": result.refinement_skip_reason,
            "oos_skip_reason": result.oos_validation_skip_reason,
            "saved_library_note": result.saved_library_note,
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
            "warnings": result.warnings,
        }

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": summary,
    })


@app.route("/full_pipeline_reports/<path:filename>")
def serve_fullpipeline_report(filename):
    return send_from_directory(FULL_PIPELINE_DIR, filename)


# ---------------------------------------------------------------------------
# Step 08: Walk-Forward Optimization -- re-optimizes on each fold's train
# window, applies the winner UNCHANGED to that fold's held-out test window,
# and chains every fold's OOS trades into one continuous result. Same
# background-job/poll shape as the other slow tabs above.
# ---------------------------------------------------------------------------

_WFO_JOBS: dict[str, dict] = {}
_WFO_JOBS_LOCK = threading.Lock()


def _wfo_job_log(job_id: str, msg: str) -> None:
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_wfo_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    n_folds: int, window_mode: str, train_frac: float, embargo_bars: int, refine_cfg: RefinementConfig,
    strategy_name: str = "", instrument: str = "",
) -> None:
    try:
        result = run_walk_forward_optimization(
            df, strategy, risk, rules, mc_cfg, n_folds=n_folds, window_mode=window_mode,
            train_frac=train_frac, embargo_bars=embargo_bars, refine_cfg=refine_cfg,
            progress_cb=lambda msg: _wfo_job_log(job_id, msg),
        )
        paths = generate_walk_forward_report(WFO_DIR, result, basename=f"walk_forward_opt_{job_id}")
        report_html = f"/wfo_reports/{Path(paths['html']).name}"
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/wfo_reports/{Path(paths['json']).name}"
        eff = getattr(result, "out_of_sample_efficiency", None)
        summary = f"OOS efficiency {eff:.2f}" if eff is not None else f"{n_folds} folds completed"
        # No strict pass/fail verdict is computed by this tool -- passed=None
        # records that it *ran*, without inventing a threshold it doesn't set.
        strategy_state.record_validation(strategy_name, instrument, "wfo", passed=None, summary=summary, report_html=report_html)
    except RefinementError as exc:
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _WFO_JOBS_LOCK:
            job = _WFO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_WFO)


@app.route("/walk-forward-opt")
def wfo_form():
    return render_template(
        "wfo.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS,
    )


@app.route("/walk-forward-opt/start", methods=["POST"])
def wfo_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_WFO, "wfo.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        fitness_metrics=FITNESS_METRICS,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_WFO)
            return render_template("wfo.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000))
        refine_cfg = RefinementConfig(
            population_size=int(form.get("population_size", 8) or 8),
            generations=int(form.get("generations", 3) or 3),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 200) or 200),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _WFO_JOBS_LOCK:
            _WFO_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_wfo_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg,
                int(form.get("n_folds", 5) or 5), form.get("window_mode", "rolling"),
                float(form.get("train_frac", 0.6) or 0.6), int(form.get("embargo_bars", 0) or 0), refine_cfg,
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("wfo_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_WFO)
        return render_template("wfo.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_WFO)
        return render_template("wfo.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 500


@app.route("/walk-forward-opt/job/<job_id>")
def wfo_job(job_id):
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
    if job is None:
        return render_template("wfo_job.html", job_id=job_id, not_found=True), 404
    return render_template("wfo_job.html", job_id=job_id, not_found=False)


@app.route("/walk-forward-opt/job/<job_id>/status.json")
def wfo_job_status(job_id):
    with _WFO_JOBS_LOCK:
        job = _WFO_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        d = result.to_summary_dict()
        d["report_html"] = job.get("report_html")
        d["report_json"] = job.get("report_json")
        summary = d
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/wfo_reports/<path:filename>")
def serve_wfo_report(filename):
    return send_from_directory(WFO_DIR, filename)


# ---------------------------------------------------------------------------
# Step 09: Multi-Objective (NSGA-II) Optimization -- searches for a Pareto
# front across 2+ objectives at once (e.g. Sharpe vs. max drawdown vs. eval
# pass probability) instead of collapsing everything into one fitness
# number. Same background-job/poll shape as the other slow tabs.
# ---------------------------------------------------------------------------

_MO_JOBS: dict[str, dict] = {}
_MO_JOBS_LOCK = threading.Lock()


def _mo_job_log(job_id: str, msg: str) -> None:
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_mo_job(job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig, mo_cfg: MultiObjectiveConfig) -> None:
    try:
        result = run_multi_objective_refinement(
            df, strategy, risk, rules, mc_cfg, mo_cfg,
            progress_cb=lambda msg: _mo_job_log(job_id, msg),
        )
        paths = generate_multi_objective_report(MULTI_OBJ_DIR, result, basename=f"multi_objective_{job_id}")
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = f"/mo_reports/{Path(paths['html']).name}"
            job["report_json"] = f"/mo_reports/{Path(paths['json']).name}"
    except RefinementError as exc:
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _MO_JOBS_LOCK:
            job = _MO_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)


@app.route("/multi-objective")
def mo_form():
    return render_template(
        "multi_objective.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES,
    )


@app.route("/multi-objective/start", methods=["POST"])
def mo_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_MULTI_OBJECTIVE, "multi_objective.html", stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS),
        default_objectives=DEFAULT_OBJECTIVES,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
            return render_template("multi_objective.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000))

        objectives = form.getlist("objectives") or list(DEFAULT_OBJECTIVES)
        mo_cfg = MultiObjectiveConfig(
            objectives=objectives,
            population_size=int(form.get("population_size", 20) or 20),
            generations=int(form.get("generations", 8) or 8),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 300) or 300),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _MO_JOBS_LOCK:
            _MO_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(target=_run_mo_job, args=(job_id, df, strategy, risk, rules, mc_cfg, mo_cfg), daemon=True)
        thread.start()
        return redirect(url_for("mo_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
        return render_template("multi_objective.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_MULTI_OBJECTIVE)
        return render_template("multi_objective.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), all_objectives=sorted(OBJECTIVE_DIRECTIONS), default_objectives=DEFAULT_OBJECTIVES), 500


@app.route("/multi-objective/job/<job_id>")
def mo_job(job_id):
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
    if job is None:
        return render_template("multi_objective_job.html", job_id=job_id, not_found=True), 404
    return render_template("multi_objective_job.html", job_id=job_id, not_found=False)


@app.route("/multi-objective/job/<job_id>/status.json")
def mo_job_status(job_id):
    with _MO_JOBS_LOCK:
        job = _MO_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "objectives": result.config.objectives,
            "pareto_front": [
                {"objective_values": dict(zip(result.config.objectives, c.objective_values)), "feasible": c.feasible}
                for c in result.pareto_front
            ],
            "generations_run": len(result.generation_history or []),
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/mo_reports/<path:filename>")
def serve_mo_report(filename):
    return send_from_directory(MULTI_OBJ_DIR, filename)


# ---------------------------------------------------------------------------
# Step 10: Walk-Forward-Aware GA -- the same GA engine as Iterative
# Refinement (Step 06), but every candidate's fitness is scored ONLY on
# chained out-of-sample fold performance instead of a single in-sample run.
# Same background-job/poll shape as the other slow tabs.
# ---------------------------------------------------------------------------

_WFGA_JOBS: dict[str, dict] = {}
_WFGA_JOBS_LOCK = threading.Lock()


def _wfga_job_log(job_id: str, msg: str) -> None:
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_wfga_job(
    job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    refine_cfg: RefinementConfig, n_folds: int, window_mode: str, train_frac: float,
    strategy_name: str = "", instrument: str = "",
) -> None:
    try:
        result = run_walkforward_aware_refinement(
            df, strategy, risk, rules, mc_cfg, refine_cfg, n_folds=n_folds, window_mode=window_mode,
            train_frac=train_frac, progress_cb=lambda msg: _wfga_job_log(job_id, msg),
        )
        paths = generate_walkforward_ga_report(WFGA_DIR, result, basename=f"walkforward_ga_{job_id}")
        report_html = f"/wfga_reports/{Path(paths['html']).name}"
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/wfga_reports/{Path(paths['json']).name}"
        gap = getattr(result, "overfitting_gap", None)
        summary = f"overfitting gap {gap:.2f}" if gap is not None else f"{n_folds} folds, walk-forward-aware GA"
        strategy_state.record_validation(strategy_name, instrument, "wfga", passed=None, summary=summary, report_html=report_html)
    except RefinementError as exc:
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _WFGA_JOBS_LOCK:
            job = _WFGA_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_WFGA)


@app.route("/walk-forward-ga")
def wfga_form():
    return render_template(
        "wfga.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS,
    )


@app.route("/walk-forward-ga/start", methods=["POST"])
def wfga_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_WFGA, "wfga.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        fitness_metrics=FITNESS_METRICS,
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_WFGA)
            return render_template("wfga.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400

        strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            commission_per_trade=float(form.get("commission", 0)),
            slippage_pips=float(form.get("slippage_pips", 0.5)),
            spread_pips=float(form.get("spread_pips", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("n_sims", 1000) or 1000))
        refine_cfg = RefinementConfig(
            population_size=int(form.get("population_size", 10) or 10),
            generations=int(form.get("generations", 5) or 5),
            search_monte_carlo_sims=int(form.get("search_mc_sims", 200) or 200),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _WFGA_JOBS_LOCK:
            _WFGA_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_wfga_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg, refine_cfg,
                int(form.get("n_folds", 4) or 4), form.get("window_mode", "rolling"), float(form.get("train_frac", 0.6) or 0.6),
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("wfga_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_WFGA)
        return render_template("wfga.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_WFGA)
        return render_template("wfga.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 500


@app.route("/walk-forward-ga/job/<job_id>")
def wfga_job(job_id):
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
    if job is None:
        return render_template("wfga_job.html", job_id=job_id, not_found=True), 404
    return render_template("wfga_job.html", job_id=job_id, not_found=False)


@app.route("/walk-forward-ga/job/<job_id>/status.json")
def wfga_job_status(job_id):
    with _WFGA_JOBS_LOCK:
        job = _WFGA_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "best_fitness": result.best.fitness,
            "best_in_sample_fitness": result.best.in_sample_fitness,
            "overfitting_gap": result.overfitting_gap,
            "oos_trade_count": result.best.oos_trade_count,
            "n_folds": result.n_folds,
            "window_mode": result.window_mode,
            "generations_run": len(result.generation_history or []),
            "elapsed_seconds": result.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/wfga_reports/<path:filename>")
def serve_wfga_report(filename):
    return send_from_directory(WFGA_DIR, filename)


def _resolve_leg_dataset(form, files, prefix: str):
    """Resolves ONE instrument leg's dataset for Portfolio -- unlike
    _resolve_dataset (which picks a single active df for the whole page),
    Portfolio needs N independent DataFrames at once, one per leg, so each
    leg gets its own uploaded-file/stored-dataset pair under a distinct
    form-field prefix (leg1_csv/leg1_existing_dataset, leg2_..., etc.).
    Returns (df, label) or (None, None) if this leg wasn't filled in."""
    uploaded = files.get(f"{prefix}_csv")
    if uploaded and uploaded.filename:
        content = uploaded.read()
        result = import_csv_bytes(content)
        if result.is_valid:
            store_csv_bytes(content, uploaded.filename)
            return result.dataframe, uploaded.filename
        raise StrategyError(f"'{uploaded.filename}': {'; '.join(result.errors)}")
    existing_choice = (form.get(f"{prefix}_existing_dataset") or "").strip()
    if existing_choice:
        candidate = get_raw_data_dir() / existing_choice
        if candidate.exists():
            result = import_csv_bytes(candidate.read_bytes())
            if result.is_valid:
                return result.dataframe, existing_choice
    return None, None


def _mode_from_filename(filename: str) -> str | None:
    """Extension-based strategy-type detection for Ensemble's multi-file
    leg upload, where (unlike every other form on this site) the mode
    isn't already known from which tab/tab-button the person is on."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in (".pine", ".pinescript", ".txt"):
        return "pinescript"
    if suffix in (".mq5", ".mqh"):
        return "mql5"
    return None


# ---------------------------------------------------------------------------
# Step 11: Multi-Asset Portfolio -- the SAME strategy config + SAME base
# risk settings applied to N DIFFERENT instruments at once, correlation-
# aware re-weighted and chained into one combined equity curve. Fast
# enough (a handful of plain backtests, no GA) to run synchronously like
# /run does, rather than the background-job/poll pattern the slower tabs
# above need.
# ---------------------------------------------------------------------------

@app.route("/portfolio")
def portfolio_form():
    return render_template(
        "portfolio.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES,
    )


@app.route("/portfolio/run", methods=["POST"])
def portfolio_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **kw)
    try:
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        legs: list[InstrumentLeg] = []
        leg_labels = []
        for i in range(1, 5):
            prefix = f"leg{i}"
            df, label = _resolve_leg_dataset(form, request.files, prefix)
            if df is None:
                continue
            weight = float(form.get(f"{prefix}_weight", 1.0) or 1.0)

            # Library-based leg picker: each leg can override the shared
            # Step-1 strategy with a specific saved strategy from the
            # library (mirrors the desktop app's "ADD LEG FROM LIBRARY"
            # button), so a portfolio can combine genuinely DIFFERENT
            # strategies -- not just one strategy across instruments.
            leg_mode = (form.get(f"{prefix}_library_mode") or "").strip()
            leg_name = (form.get(f"{prefix}_library_name") or "").strip()
            if leg_mode and leg_name:
                try:
                    leg_code = load_strategy_text(leg_mode, leg_name)
                    leg_strategy = build_strategy_from_code(leg_mode, leg_code)
                except (StrategyError, FileNotFoundError, OSError) as exc:
                    return render_template("portfolio.html", **ctx(
                        error=f"Could not load library strategy '{leg_name}' for leg {i}: {exc}"
                    )), 400
                leg_label = f"{label} ({leg_name})"
            else:
                leg_strategy = strategy
                leg_label = label

            legs.append(InstrumentLeg(name=leg_label, df=df, strategy=leg_strategy, risk=risk, weight=weight))
            leg_labels.append(leg_label)

        if len(legs) < 2:
            return render_template("portfolio.html", **ctx(error="A portfolio needs at least 2 instrument legs -- fill in a market data file/dataset for at least 2 of the leg slots below.")), 400

        config = PortfolioConfig(
            initial_balance=risk.initial_balance,
            correlation_penalty_strength=float(form.get("correlation_penalty_strength", 0.6) or 0.6),
            max_instrument_weight_frac=float(form.get("max_instrument_weight_frac", 0.5) or 0.5),
            min_weight_frac=float(form.get("min_weight_frac", 0.15) or 0.15),
        )
        result = run_portfolio_backtest(legs, config)
        run_id = uuid.uuid4().hex[:10]
        paths = generate_portfolio_report(PORTFOLIO_DIR, result, basename=f"portfolio_{run_id}")

        return render_template("portfolio.html", **ctx(result={
            "legs": leg_labels,
            "trades": result.combined_statistics.total_trades,
            "net_profit": result.combined_statistics.net_profit,
            "max_dd": result.combined_statistics.max_drawdown_pct,
            "sharpe": result.combined_statistics.sharpe_ratio,
            "diversification_ratio": result.diversification_ratio,
            "warnings": result.warnings,
            "report_html": f"/portfolio_reports/{Path(paths['html']).name}",
            "report_json": f"/portfolio_reports/{Path(paths['json']).name}",
        }))
    except (StrategyError, PortfolioError) as exc:
        return render_template("portfolio.html", **ctx(error=str(exc))), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("portfolio.html", **ctx(error=f"Unexpected error: {exc}")), 500


@app.route("/portfolio_reports/<path:filename>")
def serve_portfolio_report(filename):
    return send_from_directory(PORTFOLIO_DIR, filename)


# ---------------------------------------------------------------------------
# Regime Survival Matrix
# ---------------------------------------------------------------------------

_REGIME_DIMENSIONS = ("trend", "volatility", "session", "environment")


@app.route("/regime-matrix")
def regime_matrix_form():
    return render_template(
        "regime_matrix.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
    )


@app.route("/regime-matrix/run", methods=["POST"])
def regime_matrix_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **kw)
    guard_resp = _try_acquire_heavy_job(JOB_REGIME_MATRIX, "regime_matrix.html", **ctx())
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("regime_matrix.html", **ctx(error=dataset_error)), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )

        dim_a = form.get("dimension_a", "volatility")
        dim_b = form.get("dimension_b", "environment")
        if dim_a == dim_b:
            return render_template("regime_matrix.html", **ctx(
                error="Pick two DIFFERENT dimensions to cross for the primary matrix.",
            )), 400

        result = run_regime_matrix(df, strategy, risk, dimensions=(dim_a, dim_b))
        if result is None:
            return render_template("regime_matrix.html", **ctx(
                error="Not enough bars in this dataset to classify regimes reliably -- try a longer history.",
            )), 400

        strategy_state.record_validation(
            getattr(strategy, "name", "Strategy"), active_label, "regime_matrix",
            passed=None, summary=f"{len(result.cells)} regime cell(s) analyzed",
        )

        return render_template("regime_matrix.html", **ctx(result={
            "dataset": active_label,
            "dimensions": list(result.primary_dimensions),
            "cells": sorted([c.to_dict() for c in result.cells], key=lambda c: c["net_profit"], reverse=True),
            "single_dimension": {k: [c.to_dict() for c in v] for k, v in result.single_dimension.items()},
            "disable_regimes": [c.to_dict() for c in result.disable_regimes()],
            "notes": result.notes,
            "table_text": result.render_table(),
        }))
    except StrategyError as exc:
        return render_template("regime_matrix.html", **ctx(error=str(exc))), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("regime_matrix.html", **ctx(error=f"Unexpected error: {exc}")), 500
    finally:
        HEAVY_JOB_GUARD.release(JOB_REGIME_MATRIX)


# ---------------------------------------------------------------------------
# Strategy Family Diversity -- reads an already-completed Search Lab run
# (any search_*.db under SEARCH_DIR) and reports per-family performance.
# ---------------------------------------------------------------------------

def _available_search_runs() -> list[dict]:
    """Scans every search_*.db under SEARCH_DIR (one per Search Lab job --
    see run_search's db_path convention) and lists their runs, most
    recent first, for the family-diversity page's run picker."""
    runs = []
    for db_file in sorted(SEARCH_DIR.glob("search_*.db"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with ResultsDB(db_file) as db:
                for row in db.list_runs(limit=10):
                    runs.append({
                        "db_path": str(db_file), "run_id": row.get("run_id"),
                        "mode": row.get("mode"), "family": row.get("family"),
                        "instrument": row.get("instrument"), "created_at": row.get("created_at"),
                    })
        except Exception:  # noqa: BLE001 -- a corrupt/partial db file must not break the whole picker
            continue
    return runs


@app.route("/family-diversity")
def family_diversity_form():
    db_path = request.args.get("db_path", "")
    run_id = request.args.get("run_id", "")
    stage = request.args.get("stage", "stage1")
    runs = _available_search_runs()

    result_ctx = None
    error = None
    if db_path and run_id:
        try:
            with ResultsDB(Path(db_path)) as db:
                records = db.leaderboard(run_id, stage=stage, top_n=5000)
            if not records:
                error = f"No '{stage}' candidates found for that run -- try a different stage."
            else:
                summaries = summarize_family_performance(records)
                result_ctx = {
                    "run_id": run_id, "stage": stage, "n_records": len(records),
                    "summaries": [s.to_dict() for s in summaries],
                    "report_text": render_family_report(summaries),
                }
        except Exception as exc:  # noqa: BLE001
            error = f"Could not load that run: {exc}"

    return render_template(
        "family_diversity.html", runs=runs, selected_db_path=db_path, selected_run_id=run_id,
        selected_stage=stage, result=result_ctx, error=error,
    )


@app.route("/payout-probability")
def payout_probability_form():
    return render_template(
        "payout_probability.html", stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
    )


@app.route("/payout-probability/run", methods=["POST"])
def payout_probability_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **kw)
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("payout_probability.html", **ctx(error=dataset_error)), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )

        bt_result = run_backtest(df, strategy, risk)
        if not bt_result.trades:
            return render_template("payout_probability.html", **ctx(
                error="No trades were generated by this strategy over the given data -- there is "
                      "nothing to run a lifecycle simulation on."
            )), 400

        reset_fee_raw = (form.get("reset_fee") or "").strip()
        econ = ResetEconomics(
            evaluation_fee=float(form.get("evaluation_fee", 0) or 0),
            reset_fee=(float(reset_fee_raw) if reset_fee_raw else None),
            profit_split_pct=float(form.get("profit_split", 80) or 80),
            max_attempts=int(form.get("max_attempts", 3) or 3),
        )
        cfg = PropSurvivalConfig(
            n_simulations=min(int(form.get("n_sims", 5000) or 5000), 50_000),
            max_payouts_tracked=int(form.get("max_payouts_tracked", 5) or 5),
            funding_approval_probability=float(form.get("funding_approval_pct", 100) or 100),
            reset_economics=econ,
        )
        result = run_prop_survival_analysis(bt_result.trades, rules, cfg)

        run_id = uuid.uuid4().hex[:10]
        paths = generate_survival_report(
            PAYOUT_DIR, result, bt_result.strategy_name, active_label, rules, cfg,
            basename=f"payout_{run_id}",
        )

        return render_template("payout_probability.html", **ctx(result={
            "strategy_name": bt_result.strategy_name,
            "instrument": active_label,
            "score": result.prop_survival_score,
            "funnel": result.funnel.to_dict(),
            "net_positive_after_resets": result.reset_economics.probability_net_positive_after_resets,
            "expected_net_profit_after_resets": result.reset_economics.expected_net_profit_after_resets,
            "notes": result.notes,
            "report_html": f"/payout_reports/{Path(paths['html']).name}",
            "report_json": f"/payout_reports/{Path(paths['json']).name}",
        }))
    except StrategyError as exc:
        return render_template("payout_probability.html", **ctx(error=str(exc))), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("payout_probability.html", **ctx(error=f"Unexpected error: {exc}")), 500


@app.route("/payout_reports/<path:filename>")
def serve_payout_report(filename):
    return send_from_directory(PAYOUT_DIR, filename)


# ---------------------------------------------------------------------------
# Step 12: Multi-Strategy Ensemble -- the mirror case of Portfolio: N
# DIFFERENT strategies combined on the SAME instrument. "Blend" mode
# reuses run_portfolio_backtest under the hood (fast, synchronous, same
# report template). "Vote" mode is a single combined backtest + Monte
# Carlo, same shape as /run.
# ---------------------------------------------------------------------------

@app.route("/ensemble")
def ensemble_form():
    return render_template(
        "ensemble.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES,
    )


@app.route("/ensemble/run", methods=["POST"])
def ensemble_run():
    form = request.form
    ctx = lambda **kw: dict(stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), **kw)
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("ensemble.html", **ctx(error=dataset_error)), 400

        strategies, names = [], []
        for i in range(1, 5):
            f = request.files.get(f"leg{i}_file")
            if not f or not f.filename:
                continue
            mode = _mode_from_filename(f.filename)
            if mode is None:
                return render_template("ensemble.html", **ctx(error=f"'{f.filename}': unrecognized strategy file type (expected .py, .pine, or .mq5).")), 400
            code = f.read().decode("utf-8", errors="replace")
            try:
                strategies.append(build_strategy_from_code(mode, code))
            except StrategyError as exc:
                return render_template("ensemble.html", **ctx(error=f"'{f.filename}': {exc}")), 400
            names.append(Path(f.filename).stem)

        if len(strategies) < 2:
            return render_template("ensemble.html", **ctx(error="An ensemble needs at least 2 strategy legs -- upload at least 2 strategy files below (Python/PineScript/MQL5, mixing types is fine).")), 400

        balance = float(form.get("initial_balance", 100000) or 100000)
        risk = RiskConfig(initial_balance=balance)
        mode = form.get("ensemble_mode", "blend")

        if mode == "vote":
            min_agreement = int(form.get("min_agreement", 2) or 2)
            bt_result = run_ensemble_vote(df, strategies, risk, names=names, vote_config=EnsembleVoteConfig(min_agreement=min_agreement))
            if not bt_result.trades:
                return render_template("ensemble.html", **ctx(error="This vote ensemble produced zero trades on the given data -- nothing to report.")), 400
            rules = PropRules(account_size=balance)
            period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
            pnls = [t.pnl for t in bt_result.trades]
            dates = [t.entry_time for t in bt_result.trades]
            single_run = simulate_account(pnls, dates, rules)
            mc_result = run_monte_carlo(bt_result.trades, rules, MonteCarloConfig(n_simulations=int(form.get("n_sims", 2000) or 2000)))
            run_id = uuid.uuid4().hex[:10]
            paths = generate_full_report(
                output_dir=ENSEMBLE_DIR, strategy_name=bt_result.strategy_name, strategy_source_type="ensemble_vote",
                instrument=active_label, timeframe="unknown", backtest_period=period, backtest_result=bt_result,
                prop_rules=rules, prop_single_run=single_run, monte_carlo_result=mc_result, basename=f"ensemble_vote_{run_id}",
                risk_config=risk, price_df=df,
            )
            return render_template("ensemble.html", **ctx(result={
                "mode": "vote", "legs": names, "trades": len(bt_result.trades),
                "net_profit": bt_result.statistics.net_profit, "max_dd": bt_result.statistics.max_drawdown_pct,
                "eval_pass_probability": mc_result.evaluation_pass_probability,
                "report_html": f"/ensemble_reports/{Path(paths['html']).name}",
                "report_json": f"/ensemble_reports/{Path(paths['json']).name}",
            }))

        config = PortfolioConfig(initial_balance=balance, correlation_penalty_strength=float(form.get("correlation_penalty_strength", 0.6) or 0.6))
        result = run_ensemble_blend(df, strategies, risk, names=names, config=config)
        run_id = uuid.uuid4().hex[:10]
        paths = generate_portfolio_report(ENSEMBLE_DIR, result, basename=f"ensemble_blend_{run_id}")
        return render_template("ensemble.html", **ctx(result={
            "mode": "blend", "legs": names, "trades": result.combined_statistics.total_trades,
            "net_profit": result.combined_statistics.net_profit, "max_dd": result.combined_statistics.max_drawdown_pct,
            "sharpe": result.combined_statistics.sharpe_ratio, "diversification_ratio": result.diversification_ratio,
            "warnings": result.warnings,
            "report_html": f"/ensemble_reports/{Path(paths['html']).name}",
            "report_json": f"/ensemble_reports/{Path(paths['json']).name}",
        }))
    except (StrategyError, EnsembleError, PortfolioError) as exc:
        return render_template("ensemble.html", **ctx(error=str(exc))), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("ensemble.html", **ctx(error=f"Unexpected error: {exc}")), 500


@app.route("/ensemble_reports/<path:filename>")
def serve_ensemble_report(filename):
    return send_from_directory(ENSEMBLE_DIR, filename)


# ---------------------------------------------------------------------------
# Step 13: CPCV (Combinatorial Purged Cross-Validation) -- re-backtests one
# strategy across every combinatorial train/test partition of purged,
# embargoed groups, instead of trusting a single train/test split. Genuine
# multi-candidate PBO (Probability of Backtest Overfitting) needs a POOL of
# already-tried candidates (e.g. a Search Lab leaderboard or a Refinement
# run's final generation) as input rather than a single strategy config, so
# it isn't wired up here yet -- see WEB_PARITY_ROADMAP.md.
# ---------------------------------------------------------------------------

_CPCV_JOBS: dict[str, dict] = {}
_CPCV_JOBS_LOCK = threading.Lock()


def _cpcv_job_log(job_id: str, msg: str) -> None:
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_cpcv_job(job_id: str, df, strategy, risk: RiskConfig, n_groups: int, n_test_groups: int, embargo_frac: float, metric: str, robustness_threshold: float, max_paths: int, prop_rules=None, strategy_name: str = "", instrument: str = "") -> None:
    try:
        _cpcv_job_log(job_id, f"Running CPCV: {n_groups} groups, {n_test_groups} held out per path, metric={metric}...")
        result = run_cpcv(
            df, lambda: strategy, risk, n_groups=n_groups, n_test_groups=n_test_groups,
            embargo_frac=embargo_frac, metric=metric, robustness_threshold=robustness_threshold, max_paths=max_paths,
            prop_rules=prop_rules,
        )
        _cpcv_job_log(job_id, f"Done: {result.n_paths} paths evaluated.")
        paths = generate_cpcv_report(CPCV_DIR, result, basename=f"cpcv_{job_id}")
        report_html = f"/cpcv_reports/{Path(paths['html']).name}"
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["result"] = result
            job["report_html"] = report_html
            job["report_json"] = f"/cpcv_reports/{Path(paths['json']).name}"
        strategy_state.record_validation(
            strategy_name, instrument, "cpcv",
            passed=bool(result.is_robust), summary=f"{result.n_paths} paths evaluated", report_html=report_html,
        )
    except CPCVError as exc:
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _CPCV_JOBS_LOCK:
            job = _CPCV_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_CPCV)


@app.route("/cpcv")
def cpcv_form():
    return render_template("cpcv.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES)


@app.route("/cpcv/start", methods=["POST"])
def cpcv_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_CPCV, "cpcv.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_CPCV)
            return render_template("cpcv.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        prop_rules = PropRules(account_size=float(form.get("initial_balance", 100000)))
        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _CPCV_JOBS_LOCK:
            _CPCV_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_cpcv_job,
            args=(
                job_id, df, strategy, risk,
                int(form.get("n_groups", 6) or 6), int(form.get("n_test_groups", 2) or 2),
                float(form.get("embargo_frac", 0.01) or 0.01), form.get("metric", "eval_pass_probability"),
                float(form.get("robustness_threshold", 0.5) or 0.5), int(form.get("max_paths", 30) or 30),
                prop_rules,
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("cpcv_job", job_id=job_id))
    except (StrategyError, CPCVError) as exc:
        HEAVY_JOB_GUARD.release(JOB_CPCV)
        return render_template("cpcv.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_CPCV)
        return render_template("cpcv.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 500


@app.route("/cpcv/job/<job_id>")
def cpcv_job(job_id):
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
    if job is None:
        return render_template("cpcv_job.html", job_id=job_id, not_found=True), 404
    return render_template("cpcv_job.html", job_id=job_id, not_found=False)


@app.route("/cpcv/job/<job_id>/status.json")
def cpcv_job_status(job_id):
    with _CPCV_JOBS_LOCK:
        job = _CPCV_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = result.to_dict()
        summary["report_html"] = job.get("report_html")
        summary["report_json"] = job.get("report_json")
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/cpcv_reports/<path:filename>")
def serve_cpcv_report(filename):
    return send_from_directory(CPCV_DIR, filename)


# ---------------------------------------------------------------------------
# Step 14: Parameter Sensitivity -- sweeps every tunable numeric parameter
# independently across +/- a percent range, holding others fixed, and flags
# any "cliff" (a narrow-edge parameter rather than a stable plateau). 1D
# sweeps only for now; the 2D heatmap needs a two-step UI (discover tunable
# parameter names, then pick 2) not yet built -- see WEB_PARITY_ROADMAP.md.
# ---------------------------------------------------------------------------

_SENS_JOBS: dict[str, dict] = {}
_SENS_JOBS_LOCK = threading.Lock()


def _sens_job_log(job_id: str, msg: str) -> None:
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_sensitivity_job(job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig, metric: str, pct_range: float, n_steps: int, max_params: int, strategy_name: str = "", instrument: str = "") -> None:
    try:
        _sens_job_log(job_id, f"Sweeping up to {max_params} tunable parameter(s), {n_steps} steps each, metric={metric}...")
        results = compute_1d_sensitivity(df, strategy, risk, rules, mc_cfg, metric=metric, pct_range=pct_range, n_steps=n_steps, max_params=max_params)
        _sens_job_log(job_id, f"Done: swept {len(results)} parameter(s).")
        paths = generate_sensitivity_report(SENSITIVITY_DIR, results, basename=f"sensitivity_{job_id}")
        report_html = f"/sensitivity_reports/{Path(paths['html']).name}"
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["results"] = results
            job["report_html"] = report_html
            job["report_json"] = f"/sensitivity_reports/{Path(paths['json']).name}"
        # This tool is diagnostic (flags cliffs vs. stable plateaus per
        # parameter) rather than pass/fail -- passed=None records that it ran.
        strategy_state.record_validation(
            strategy_name, instrument, "sensitivity",
            passed=None, summary=f"{len(results)} parameter(s) swept", report_html=report_html,
        )
    except RefinementError as exc:
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _SENS_JOBS_LOCK:
            job = _SENS_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)


@app.route("/sensitivity")
def sensitivity_form():
    return render_template("sensitivity.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES)


@app.route("/sensitivity/start", methods=["POST"])
def sensitivity_start():
    form = request.form
    guard_resp = _try_acquire_heavy_job(
        JOB_SENSITIVITY, "sensitivity.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
    )
    if guard_resp:
        return guard_resp
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
            return render_template("sensitivity.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000)), pip_size=float(form.get("pip_size", 0.0001)))
        rules = PropRules(account_size=float(form.get("account_size", 100000)))
        mc_cfg = MonteCarloConfig(n_simulations=int(form.get("mc_sims", 500) or 500))

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _SENS_JOBS_LOCK:
            _SENS_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "results": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(
            target=_run_sensitivity_job,
            args=(
                job_id, df, strategy, risk, rules, mc_cfg, form.get("metric", "profit_factor"),
                float(form.get("pct_range", 0.5) or 0.5), int(form.get("n_steps", 9) or 9), int(form.get("max_params", 8) or 8),
            ),
            kwargs={"strategy_name": getattr(strategy, "name", "Strategy"), "instrument": active_label},
            daemon=True,
        )
        thread.start()
        return redirect(url_for("sensitivity_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
        return render_template("sensitivity.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SENSITIVITY)
        return render_template("sensitivity.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 500


@app.route("/sensitivity/job/<job_id>")
def sensitivity_job(job_id):
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
    if job is None:
        return render_template("sensitivity_job.html", job_id=job_id, not_found=True), 404
    return render_template("sensitivity_job.html", job_id=job_id, not_found=False)


@app.route("/sensitivity/job/<job_id>/status.json")
def sensitivity_job_status(job_id):
    with _SENS_JOBS_LOCK:
        job = _SENS_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    results = job.get("results")
    summary = None
    if results is not None:
        summary = {
            "sweeps": [r.to_dict() for r in results],
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


@app.route("/sensitivity_reports/<path:filename>")
def serve_sensitivity_report(filename):
    return send_from_directory(SENSITIVITY_DIR, filename)


# ---------------------------------------------------------------------------
# Quick Optimize -- one-click single-strategy GA tune from the Strategy
# Library (reuses the same walk-forward-aware GA as Full Pipeline/Step 06,
# saves the winner back into the library tagged "draft"). Same background-
# job/poll pattern as the other GA-driven tabs.
# ---------------------------------------------------------------------------

_QUICKOPT_JOBS: dict[str, dict] = {}
_QUICKOPT_JOBS_LOCK = threading.Lock()


def _quickopt_job_log(job_id: str, msg: str) -> None:
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_quickopt_job(job_id: str, df, strategy, risk: RiskConfig, rules: PropRules, cfg: QuickOptimizeConfig) -> None:
    try:
        result = run_quick_optimize(df, strategy, risk, rules, cfg, progress_cb=lambda msg: _quickopt_job_log(job_id, msg))
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except RefinementError as exc:
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        with _QUICKOPT_JOBS_LOCK:
            job = _QUICKOPT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/quick-optimize")
def quickopt_form():
    return render_template("quick_optimize.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), strategy_statuses=STRATEGY_STATUSES, fitness_metrics=FITNESS_METRICS)


@app.route("/quick-optimize/start", methods=["POST"])
def quickopt_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("quick_optimize.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400
        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000)), pip_size=float(form.get("pip_size", 0.0001)))
        rules = PropRules(account_size=float(form.get("account_size", 100000)))
        cfg = QuickOptimizeConfig(
            ga_population=int(form.get("ga_population", 16) or 16),
            ga_generations=int(form.get("ga_generations", 8) or 8),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            n_folds=int(form.get("n_folds", 4) or 4),
            save_to_library=form.get("save_to_library", "on") == "on",
            adaptive_risk_enabled=form.get("adaptive_risk_enabled") == "on",
        )
        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _QUICKOPT_JOBS_LOCK:
            _QUICKOPT_JOBS[job_id] = {"log": initial_log, "done": False, "error": None, "result": None, "started_at": time.time(), "instrument": active_label}
        thread = threading.Thread(target=_run_quickopt_job, args=(job_id, df, strategy, risk, rules, cfg), daemon=True)
        thread.start()
        return redirect(url_for("quickopt_job", job_id=job_id))
    except (StrategyError, RefinementError) as exc:
        return render_template("quick_optimize.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("quick_optimize.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), fitness_metrics=FITNESS_METRICS), 500


@app.route("/quick-optimize/job/<job_id>")
def quickopt_job(job_id):
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
    if job is None:
        return render_template("quick_optimize_job.html", job_id=job_id, not_found=True), 404
    return render_template("quick_optimize_job.html", job_id=job_id, not_found=False)


@app.route("/quick-optimize/job/<job_id>/status.json")
def quickopt_job_status(job_id):
    with _QUICKOPT_JOBS_LOCK:
        job = _QUICKOPT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "strategy_display_name": result.strategy_display_name,
            "baseline_trades": result.baseline_trades, "baseline_net_profit": result.baseline_net_profit,
            "baseline_win_rate": result.baseline_win_rate, "baseline_eval_pass_probability": result.baseline_eval_pass_probability,
            "optimized_trades": result.optimized_trades, "optimized_net_profit": result.optimized_net_profit,
            "optimized_win_rate": result.optimized_win_rate, "optimized_eval_pass_probability": result.optimized_eval_pass_probability,
            "improved": result.improved,
            "saved_library_note": result.saved_library_note,
            "elapsed_seconds": result.elapsed_seconds,
            "warnings": result.warnings,
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "instrument": job.get("instrument"), "summary": summary})


# ---------------------------------------------------------------------------
# Evolution Lab -- an open-ended, resumable multi-family GA search that
# runs generation after generation until STOP is clicked (or the process
# restarts, in which case its own on-disk checkpoint resumes it). Unlike
# every other tab above, this isn't a "start a job, wait for it to finish"
# shape -- app.evolution.engine.EvolutionRunner already owns its own
# background thread and start()/stop()/status() control surface, so the
# web app holds ONE global runner instance (matches this app's existing
# single-user/LAN trust model) and drives it directly rather than
# reimplementing job management around it.
# ---------------------------------------------------------------------------

_EVOLUTION_RUNNER: EvolutionRunner | None = None
_EVOLUTION_LOCK = threading.Lock()
_EVOLUTION_LOG: list[str] = []
_EVOLUTION_LOG_MAX = 500


def _evolution_log(msg: str) -> None:
    _EVOLUTION_LOG.append(msg)
    del _EVOLUTION_LOG[:-_EVOLUTION_LOG_MAX]


@app.route("/evolution")
def evolution_form():
    return render_template(
        "evolution.html", stored_datasets=list_stored_datasets(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        running=(_EVOLUTION_RUNNER is not None and _EVOLUTION_RUNNER.is_running),
    )


@app.route("/evolution/start", methods=["POST"])
def evolution_start():
    global _EVOLUTION_RUNNER
    form = request.form
    with _EVOLUTION_LOCK:
        if _EVOLUTION_RUNNER is not None and _EVOLUTION_RUNNER.is_running:
            return redirect(url_for("evolution_form"))
    if not HEAVY_JOB_GUARD.try_acquire(JOB_EVOLUTION_LAB):
        return render_template(
            "evolution.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one of Search Lab / Evolution Lab / Full Pipeline / Speed Run at the same time can "
                f"exhaust available memory (each spawns its own worker processes, each holding a full "
                f"copy of the loaded data). Wait for it to finish, or stop it, before starting Evolution Lab."
            ),
            stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            running=False,
        ), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
            return render_template("evolution.html", error=dataset_error, stored_datasets=list_stored_datasets(), families=[{"name": n, "description": family_description(n)} for n in list_families()], running=False), 400

        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000) or 100000))
        rules = PropRules(account_size=float(form.get("initial_balance", 100000) or 100000))
        families_selected = form.getlist("families") or None
        cfg = EvolutionConfig(
            population_size=int(form.get("population_size", 60) or 60),
            elite_keep=int(form.get("elite_keep", 10) or 10),
            families=families_selected,
            mc_sims=int(form.get("mc_sims", 1000) or 1000),
            max_generations=(int(form["max_generations"]) if form.get("max_generations") else None),
            save_to_library=form.get("save_to_library", "on") == "on",
            resume_from_checkpoint=form.get("resume_from_checkpoint", "on") == "on",
        )
        _EVOLUTION_LOG.clear()
        _EVOLUTION_LOG.append(f"Loaded {len(df)} bars from {active_label}.")
        with _EVOLUTION_LOCK:
            _EVOLUTION_RUNNER = EvolutionRunner(df, risk, rules, cfg, progress_cb=_evolution_log)
            _EVOLUTION_RUNNER.start()
        return redirect(url_for("evolution_form"))
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
        log_crash("Evolution Lab (web)", exc=exc)
        return render_template("evolution.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), families=[{"name": n, "description": family_description(n)} for n in list_families()], running=False), 500


@app.route("/evolution/stop", methods=["POST"])
def evolution_stop():
    with _EVOLUTION_LOCK:
        if _EVOLUTION_RUNNER is not None:
            _EVOLUTION_RUNNER.stop()
    return redirect(url_for("evolution_form"))


@app.route("/evolution/reset", methods=["POST"])
def evolution_reset():
    with _EVOLUTION_LOCK:
        if _EVOLUTION_RUNNER is not None and not _EVOLUTION_RUNNER.is_running:
            _EVOLUTION_RUNNER.reset()
            _EVOLUTION_LOG.clear()
    return redirect(url_for("evolution_form"))


@app.route("/evolution/promote", methods=["POST"])
def evolution_promote():
    """Web equivalent of the desktop app's PROMOTE TO STRATEGY LIBRARY
    button: saves one leaderboard candidate's manual-builder config into
    the Strategy Library (as a "manual" type -- see
    app.strategy.library.STRATEGY_TYPES, which didn't recognize "manual"
    at all until this fix, so this exact action always failed with
    "Unknown strategy type 'manual'" on both desktop and web). Looks in
    the live runner first, then falls back to the on-disk checkpoint so
    this also works after a server restart, matching the desktop app's
    _load_evolution_leaderboard_from_disk fallback.
    """
    candidate_id = (request.form.get("candidate_id") or "").strip()
    if not candidate_id:
        return jsonify({"ok": False, "error": "No candidate_id given."}), 400

    record = None
    with _EVOLUTION_LOCK:
        runner = _EVOLUTION_RUNNER
    if runner is not None:
        for r in runner.leaderboard:
            d = r.to_checkpoint_dict()
            if d.get("candidate_id") == candidate_id:
                record = d
                break
    if record is None:
        try:
            checkpoint = evo_checkpoint.load_checkpoint()
            if checkpoint is not None:
                for d in checkpoint.leaderboard:
                    if d.get("candidate_id") == candidate_id:
                        record = d
                        break
        except Exception:
            pass
    if record is None:
        return jsonify({"ok": False, "error": f"Candidate '{candidate_id}' not found on the leaderboard."}), 404

    config = (record.get("spec") or {}).get("config")
    if not config:
        return jsonify({"ok": False, "error": "This candidate has no manual-builder config to promote."}), 400

    family = (record.get("meta") or {}).get("family", "strategy")
    filename = f"evolab_promoted_{family}_{candidate_id[-8:]}.json"
    text = json.dumps(config, indent=2)
    try:
        try:
            save_strategy_text(text, filename, "manual", overwrite=False)
        except StrategyAlreadyExists:
            save_strategy_text(text, filename, "manual", overwrite=True)
        set_strategy_status("manual", filename, "validated")  # see main_window.py's matching note
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "filename": filename})


@app.route("/evolution/status.json")
def evolution_status():
    with _EVOLUTION_LOCK:
        runner = _EVOLUTION_RUNNER
    if runner is None:
        return jsonify({"running": False, "started": False, "log": [], "leaderboard": [], "journal": []})
    status = runner.status()
    if not status["running"]:
        # Covers both a deliberate stop and the runner's own thread exiting
        # on its own (finished, or crashed) -- either way the heavy-job
        # slot must free up so another tab/route can start. Harmless if
        # some other job already holds/released it (release() is a no-op
        # unless this name is the current holder).
        HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    leaderboard = [r.to_checkpoint_dict() for r in runner.leaderboard]
    return jsonify({
        "started": True,
        "running": status["running"],
        "generation": status["generation"],
        "leaderboard_size": status["leaderboard_size"],
        "resumed": status["resumed"],
        "log": list(_EVOLUTION_LOG),
        "leaderboard": leaderboard,
        "journal": runner.journal[-30:],
    })


# ---------------------------------------------------------------------------
# 18. Research Agent -- a ReAct-style tool-calling agent whose tools are
# 100% read-only calls into the real backtest/prop-sim/Monte Carlo/walk-
# forward/regime/sensitivity/cost-stress engine (see app.ai.research_agent
# -- no code-editing tool exists, so it can only recommend, never apply).
# Needs a local Ollama reachable from wherever this server runs -- see the
# AI Assist note on the Full Pipeline page for what that means (point it
# at whatever machine on your LAN is running Ollama).
# ---------------------------------------------------------------------------

_AGENT_JOBS: dict[str, dict] = {}
_AGENT_JOBS_LOCK = threading.Lock()


def _agent_job_log(job_id: str, msg: str) -> None:
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_agent_job(job_id: str, question: str, ctx: ResearchAgentContext, settings: OllamaSettings) -> None:
    try:
        agent = ResearchAgent(settings)
        result = agent.run(question, ctx, progress_cb=lambda msg: _agent_job_log(job_id, msg))
        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except Exception as exc:  # noqa: BLE001
        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/research-agent")
def research_agent_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "research_agent.html", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(),
        strategy_statuses=STRATEGY_STATUSES, ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
    )


@app.route("/research-agent/start", methods=["POST"])
def research_agent_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template("research_agent.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model=""), 400

        strategy, _library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
        risk = RiskConfig(initial_balance=float(form.get("initial_balance", 100000) or 100000), pip_size=float(form.get("pip_size", 0.0001) or 0.0001))
        rules = PropRules(account_size=float(form.get("account_size", 100000) or 100000))
        question = (form.get("question") or "").strip()
        if not question:
            return render_template("research_agent.html", error="Enter a question for the agent to investigate.", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model=""), 400

        settings = OllamaSettings(enabled=True, host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434", model=form.get("ai_model", "llama3.1") or "llama3.1")
        try:
            save_ollama_settings(settings)
        except Exception:
            pass

        ctx = ResearchAgentContext(
            df=df, strategy_builder=(lambda s=strategy: s), strategy_name=getattr(strategy, "name", "Strategy"),
            source_type=strategy.source_type, risk=risk, prop_rules=rules, instrument=active_label,
        )

        job_id = uuid.uuid4().hex[:12]
        with _AGENT_JOBS_LOCK:
            _AGENT_JOBS[job_id] = {"log": [f"Loaded {len(df)} bars from {active_label}.", f"Question: {question}"], "done": False, "error": None, "result": None, "started_at": time.time()}
        thread = threading.Thread(target=_run_agent_job, args=(job_id, question, ctx, settings), daemon=True)
        thread.start()
        return redirect(url_for("research_agent_job", job_id=job_id))
    except StrategyError as exc:
        return render_template("research_agent.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model=""), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("research_agent.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json(), ai_enabled=False, ai_host="", ai_model=""), 500


@app.route("/research-agent/job/<job_id>")
def research_agent_job(job_id):
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
    if job is None:
        return render_template("research_agent_job.html", job_id=job_id, not_found=True), 404
    return render_template("research_agent_job.html", job_id=job_id, not_found=False)


@app.route("/research-agent/job/<job_id>/status.json")
def research_agent_job_status(job_id):
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    result = job.get("result")
    summary = None
    if result is not None:
        summary = {
            "final_answer": result.final_answer,
            "error": result.error,
            "stopped_reason": result.stopped_reason,
            "steps": [
                {"step_index": s.step_index, "thought": s.thought, "action": s.action, "action_input": s.action_input, "observation": s.observation, "note": s.note}
                for s in result.steps
            ],
        }
    return jsonify({"found": True, "done": job["done"], "error": job["error"], "log": job["log"], "summary": summary})


@app.route("/forward-test")
def forward_test_info():
    return render_template("forward_test.html")


@app.route("/deploy-live")
def deploy_live_info():
    return render_template("deploy_live.html")


@app.route("/search")
def search_form():
    return render_template(
        "search.html",
        stored_datasets=list_stored_datasets(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
        strategy_statuses=STRATEGY_STATUSES,
    )


@app.route("/search/start", methods=["POST"])
def search_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_SEARCH_LAB):
        return render_template(
            "search.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory. Wait for it to finish before starting Search Lab."
            ),
            stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
            return render_template(
                "search.html", error=dataset_error, stored_datasets=list_stored_datasets(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                saved_strategies_json=_saved_strategies_json(),
            ), 400

        mode_key = form.get("search_mode", "family_named")
        seed = int(form.get("seed", 42) or 42)
        max_candidates = int(form.get("max_candidates", 200) or 200)
        library_ref = None

        if mode_key == "single":
            strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
            space = generate_search_space(mode="single", strategy=strategy)
        elif mode_key == "family_grid":
            strategy, library_ref = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
            space = generate_search_space(
                mode="family", strategy=strategy,
                grid_points_per_gene=int(form.get("grid_points", 3) or 3),
                max_candidates=max_candidates, seed=seed,
            )
        else:
            family_key = form.get("family", "all") or "all"
            space = generate_search_space(mode="family", family=family_key, max_candidates=max_candidates, seed=seed)

        workers_raw = (form.get("workers") or "").strip()
        stage_cfg = SearchStageConfig(
            min_trades=int(form.get("min_trades", 20) or 20),
            min_profit_factor=float(form.get("min_profit_factor", 1.05) or 1.05),
            stage1_top_n=int(form.get("stage1_top_n", 40) or 40),
            ga_population=int(form.get("ga_population", 10) or 10),
            ga_generations=int(form.get("ga_generations", 4) or 4),
            stage2_top_n=int(form.get("stage2_top_n", 10) or 10),
            full_mc_sims=int(form.get("full_mc_sims", 3000) or 3000),
            walk_forward_folds=int(form.get("walk_forward_folds", 4) or 4),
            robustness_neighbors=int(form.get("robustness_neighbors", 6) or 6),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            workers=int(workers_raw) if workers_raw else None,
            random_seed=seed,
        )
        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000) or 100000),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0) or 1.0),
            pip_size=float(form.get("pip_size", 0.0001) or 0.0001),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000) or 100000),
            evaluation_profit_target_pct=float(form.get("profit_target", 8) or 8),
            daily_loss_limit_pct=float(form.get("daily_loss", 5) or 5),
            max_drawdown_pct=float(form.get("max_dd", 10) or 10),
        )

        job_id = uuid.uuid4().hex[:12]
        db_path = str(SEARCH_DIR / f"search_{job_id}.db")
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _SEARCH_JOBS_LOCK:
            _SEARCH_JOBS[job_id] = {
                "log": initial_log,
                "done": False, "error": None, "summary": None,
                "started_at": time.time(), "instrument": active_label, "mode": mode_key,
            }
        thread = threading.Thread(
            target=_run_search_job,
            args=(job_id, df, risk, rules, space, stage_cfg, active_label, db_path, library_ref),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("search_job", job_id=job_id))

    except StrategySpaceError as exc:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 400
    except StrategyError as exc:
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 400
    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
        log_crash("Search Lab (web, start)", exc=exc)
        return render_template(
            "search.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 500


@app.route("/search/job/<job_id>")
def search_job(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None:
        return render_template("search_job.html", job_id=job_id, not_found=True), 404
    return render_template("search_job.html", job_id=job_id, not_found=False)


@app.route("/search/job/<job_id>/status.json")
def search_job_status(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    summary = job.get("summary")
    leaderboard = None
    if summary is not None:
        leaderboard = [
            {
                "candidate_id": row.get("candidate_id"),
                "source_type": row.get("source_type", "manual"),
                "family": row.get("family"),
                "composite_score": row.get("composite_score"),
                "psr": (row.get("deflated_sharpe") or {}).get("probabilistic_sharpe"),
                "net_profit": (row.get("statistics") or {}).get("net_profit"),
                "profit_factor": (row.get("statistics") or {}).get("profit_factor"),
                "win_rate": (row.get("statistics") or {}).get("win_rate"),
                "total_trades": (row.get("statistics") or {}).get("total_trades"),
                "eval_pass_pct": (row.get("mc_summary") or {}).get("evaluation_pass_probability"),
                "passed_gate": bool(row.get("passed_stage3_gate")),
                "gate_notes": row.get("gate_notes") or "",
            }
            for row in (summary.leaderboard or [])
        ]

    return jsonify({
        "found": True,
        "done": job["done"],
        "error": job["error"],
        "log": job["log"],
        "instrument": job.get("instrument"),
        "summary": None if summary is None else {
            "mode": summary.mode, "family": summary.family,
            "total_candidates": summary.total_candidates,
            "stage1_survivors": summary.stage1_survivors,
            "stage2_survivors": summary.stage2_survivors,
            "stage3_survivors": summary.stage3_survivors,
            "champion_candidate_id": summary.champion_candidate_id,
            "elapsed_seconds": summary.elapsed_seconds,
            "report_html": job.get("report_html"),
            "report_json": job.get("report_json"),
        },
        "leaderboard": leaderboard,
    })


@app.route("/search/job/<job_id>/promote", methods=["POST"])
def search_job_promote(job_id):
    with _SEARCH_JOBS_LOCK:
        job = _SEARCH_JOBS.get(job_id)
    if job is None or not job.get("done") or job.get("summary") is None:
        return jsonify({"ok": False, "error": "Job not found, not finished, or produced no results."}), 400

    candidate_id = request.form.get("candidate_id")
    if not candidate_id and request.is_json:
        candidate_id = (request.get_json(silent=True) or {}).get("candidate_id")
    if not candidate_id:
        return jsonify({"ok": False, "error": "candidate_id is required."}), 400

    try:
        result = promote_champion(
            job["db_path"], job["summary"].run_id, candidate_id,
            job["df"], job["risk"], job["rules"],
            output_dir=str(SEARCH_DIR / "champion" / job_id),
        )
        with _SEARCH_JOBS_LOCK:
            job.setdefault("promoted", {})[candidate_id] = {
                "html": f"/search_reports_champion/{job_id}/{result['report_paths']['html'].name}",
                "json": f"/search_reports_champion/{job_id}/{result['report_paths']['json'].name}",
            }
        return jsonify({"ok": True, "report_html": job["promoted"][candidate_id]["html"]})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/search_reports/<path:filename>")
def serve_search_report(filename):
    return send_from_directory(SEARCH_DIR, filename)


@app.route("/search_reports_champion/<job_id>/<path:filename>")
def serve_search_champion_report(job_id, filename):
    return send_from_directory(SEARCH_DIR / "champion" / job_id, filename)


# ---------------------------------------------------------------------------
# Speed Run -- "I have almost no time left, find me anything that works."
# One button, no strategy to bring in: chains a wide multi-family discovery
# search straight into concurrent Full Pipeline validation of the top
# survivors, then reports whichever one is best. Same background-job/poll
# shape as Full Pipeline above -- see app.orchestration.speed_run for the
# actual chained-phases logic, which this route just wires up to the web UI
# the same way the desktop Speed Run tab wires it up to Tkinter.
# ---------------------------------------------------------------------------

_SPEEDRUN_JOBS: dict[str, dict] = {}
_SPEEDRUN_JOBS_LOCK = threading.Lock()


def _speedrun_job_log(job_id: str, msg: str) -> None:
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
        if job is not None:
            job["log"].append(msg)


def _run_speedrun_job(
    job_id: str, df, risk: RiskConfig, rules: PropRules, cfg: SpeedRunConfig, active_label: str,
) -> None:
    try:
        result = run_speed_run(
            df, risk, rules, SPEEDRUN_DIR, cfg,
            progress_cb=lambda msg: _speedrun_job_log(job_id, msg),
            instrument=active_label,
        )
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["result"] = result
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Speed Run (web)", exc=exc)
        with _SPEEDRUN_JOBS_LOCK:
            job = _SPEEDRUN_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"
    finally:
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)


@app.route("/speed-run")
def speed_run_form():
    return render_template(
        "speed_run.html", stored_datasets=list_stored_datasets(), fitness_metrics=FITNESS_METRICS,
    )


@app.route("/speed-run/start", methods=["POST"])
def speed_run_start():
    form = request.form
    if not HEAVY_JOB_GUARD.try_acquire(JOB_SPEED_RUN):
        return render_template(
            "speed_run.html",
            error=(
                f"{HEAVY_JOB_GUARD.active_name} is already running on this server. Running more than "
                f"one heavy job (Search Lab / Evolution Lab / Full Pipeline / Speed Run) at the same "
                f"time can exhaust available memory -- this is the same failure mode that can freeze "
                f"or crash the desktop app. Wait for it to finish before starting Speed Run."
            ),
            stored_datasets=list_stored_datasets(), fitness_metrics=FITNESS_METRICS,
        ), 409
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
            return render_template(
                "speed_run.html", error=dataset_error, stored_datasets=list_stored_datasets(),
                fitness_metrics=FITNESS_METRICS,
            ), 400

        risk = RiskConfig(
            initial_balance=float(form.get("initial_balance", 100000)),
            risk_mode=form.get("risk_mode", "percent"),
            risk_value=float(form.get("risk_value", 1.0)),
            pip_size=float(form.get("pip_size", 0.0001)),
        )
        rules = PropRules(
            account_size=float(form.get("account_size", 100000)),
            evaluation_profit_target_pct=float(form.get("profit_target", 8)),
            daily_loss_limit_pct=float(form.get("daily_loss", 5)),
            max_drawdown_pct=float(form.get("max_dd", 10)),
        )
        cfg = SpeedRunConfig(
            max_candidates=int(form.get("max_candidates", 1200) or 1200),
            stage1_top_n=int(form.get("stage1_top_n", 24) or 24),
            ga_population=int(form.get("ga_population", 8) or 8),
            ga_generations=int(form.get("ga_generations", 3) or 3),
            top_k_to_validate=int(form.get("top_k_to_validate", 3) or 3),
            max_concurrent_validations=int(form.get("max_concurrent_validations", 2) or 2),
            validation_folds=int(form.get("validation_folds", 3) or 3),
            validation_final_mc_sims=int(form.get("validation_final_mc_sims", 3000) or 3000),
            fitness_metric=form.get("fitness_metric", "eval_pass_probability"),
            save_winner_to_library=form.get("save_to_library") == "on",
            random_seed=int(form.get("random_seed", 42) or 42),
        )

        job_id = uuid.uuid4().hex[:12]
        initial_log = [f"Loaded {len(df)} bars from {active_label}."]
        if import_note:
            initial_log.append(import_note)
        with _SPEEDRUN_JOBS_LOCK:
            _SPEEDRUN_JOBS[job_id] = {
                "log": initial_log, "done": False, "error": None, "result": None,
                "started_at": time.time(), "instrument": active_label,
            }
        thread = threading.Thread(
            target=_run_speedrun_job, args=(job_id, df, risk, rules, cfg, active_label), daemon=True,
        )
        thread.start()
        return redirect(url_for("speed_run_job", job_id=job_id))

    except Exception as exc:  # noqa: BLE001
        HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)
        log_crash("Speed Run (web, start)", exc=exc)
        return render_template(
            "speed_run.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(),
            fitness_metrics=FITNESS_METRICS,
        ), 500


@app.route("/speed-run/job/<job_id>")
def speed_run_job(job_id):
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return render_template("speed_run_job.html", job_id=job_id, not_found=True), 404
    return render_template("speed_run_job.html", job_id=job_id, not_found=False)


@app.route("/speed-run/job/<job_id>/status.json")
def speed_run_job_status(job_id):
    with _SPEEDRUN_JOBS_LOCK:
        job = _SPEEDRUN_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404

    result: SpeedRunResult | None = job.get("result")
    summary = None
    if result is not None:
        winner = None
        if result.winner is not None and result.winner.pipeline_result is not None:
            pr = result.winner.pipeline_result
            winner = {
                "candidate_id": result.winner.candidate_id,
                "family": result.winner.family,
                "verdict": pr.verdict,
                "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                "first_payout_probability": pr.final_mc.first_payout_probability,
                "saved_library_note": pr.saved_library_note,
                "report_html": f"/speed_run_reports/{Path(pr.report_paths['html']).name}" if pr.report_paths.get("html") else None,
            }
        candidates = []
        for r in sorted(result.candidates, key=_speedrun_rank_key):
            if r.pipeline_result is not None:
                pr = r.pipeline_result
                candidates.append({
                    "candidate_id": r.candidate_id, "family": r.family, "verdict": pr.verdict,
                    "eval_pass_probability": pr.final_mc.evaluation_pass_probability,
                    "first_payout_probability": pr.final_mc.first_payout_probability,
                    "report_html": f"/speed_run_reports/{Path(pr.report_paths['html']).name}" if pr.report_paths.get("html") else None,
                })
            else:
                candidates.append({
                    "candidate_id": r.candidate_id, "family": r.family, "verdict": None,
                    "eval_pass_probability": None, "first_payout_probability": None, "report_html": None,
                })
        summary = {
            "winner": winner,
            "winner_reason": result.winner_reason,
            "elapsed_seconds": result.elapsed_seconds,
            "candidates": candidates,
            "guidance": result.guidance,
        }

    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "log": job["log"],
        "instrument": job.get("instrument"), "summary": summary,
    })


@app.route("/speed_run_reports/<path:filename>")
def serve_speedrun_report(filename):
    return send_from_directory(SPEEDRUN_REPORTS_DIR, filename)


# ---------------------------------------------------------------------------
# Generate Strategies (AI) -- drafts a NEW strategy's source code from a
# plain-language idea via a local Ollama model, grounded in research/ papers
# and your own best existing strategies. See app.ai.strategy_generator's
# module docstring for the full safety rationale: every result is saved
# tagged DRAFT and nothing here ever runs generated code automatically.
# Needs no market data at all, unlike everything else in this file -- it
# only drafts source code, it doesn't backtest it.
# ---------------------------------------------------------------------------

_GENSTRAT_JOBS: dict[str, dict] = {}
_GENSTRAT_JOBS_LOCK = threading.Lock()


def _genstrat_job_progress(job_id: str, tokens: int, elapsed: float) -> None:
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
        if job is not None:
            job["tokens"] = tokens
            job["elapsed"] = elapsed


def _run_genstrat_job(
    job_id: str, settings: OllamaSettings, language: str, idea: str,
    num_ctx: int, num_predict: int, stall_timeout: int, max_total: int,
) -> None:
    from app.ai.strategy_generator import generate_strategy

    try:
        result = generate_strategy(
            settings, language, idea,
            timeout=stall_timeout, max_total_seconds=max_total,
            num_ctx=num_ctx, num_predict=num_predict,
            progress_cb=lambda tokens, elapsed: _genstrat_job_progress(job_id, tokens, elapsed),
        )
        with _GENSTRAT_JOBS_LOCK:
            job = _GENSTRAT_JOBS[job_id]
            job["done"] = True
            if result.code is None:
                job["error"] = result.error or "Generation failed."
            else:
                job["code"] = result.code
                job["filename_hint"] = result.filename_hint
                job["language"] = language
                job["idea"] = idea
    except Exception as exc:  # noqa: BLE001 -- must surface on the status page, not crash the thread silently
        log_crash("Generate Strategies (web)", exc=exc)
        with _GENSTRAT_JOBS_LOCK:
            job = _GENSTRAT_JOBS[job_id]
            job["done"] = True
            job["error"] = f"Unexpected error: {exc}"


@app.route("/generate-strategies")
def generate_strategies_form():
    saved_ai = load_ollama_settings()
    return render_template(
        "generate_strategies.html", ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
    )


@app.route("/generate-strategies/start", methods=["POST"])
def generate_strategies_start():
    form = request.form
    idea = (form.get("idea") or "").strip()
    if not idea:
        saved_ai = load_ollama_settings()
        return render_template(
            "generate_strategies.html", error="Describe the strategy idea first.",
            ai_enabled=saved_ai.enabled, ai_host=saved_ai.host, ai_model=saved_ai.model,
        ), 400

    language = form.get("language", "python")
    settings = OllamaSettings(
        enabled=True, host=form.get("ai_host", "http://localhost:11434") or "http://localhost:11434",
        model=form.get("ai_model", "llama3.1") or "llama3.1",
    )
    try:
        save_ollama_settings(settings)  # persists, same as the desktop tab's own checkbox
    except Exception:
        pass  # best-effort -- a save failure shouldn't block the run itself

    from app.ai.strategy_generator import DEFAULT_MAX_TOTAL_SECONDS, DEFAULT_NUM_CTX, DEFAULT_NUM_PREDICT, DEFAULT_TIMEOUT_SECONDS

    num_ctx = int(form.get("num_ctx", DEFAULT_NUM_CTX) or DEFAULT_NUM_CTX)
    num_predict = int(form.get("num_predict", DEFAULT_NUM_PREDICT) or DEFAULT_NUM_PREDICT)
    stall_timeout = int(form.get("stall_timeout", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
    max_total = int(form.get("max_total", DEFAULT_MAX_TOTAL_SECONDS) or DEFAULT_MAX_TOTAL_SECONDS)

    job_id = uuid.uuid4().hex[:12]
    with _GENSTRAT_JOBS_LOCK:
        _GENSTRAT_JOBS[job_id] = {
            "done": False, "error": None, "code": None, "filename_hint": None, "language": language,
            "idea": idea, "tokens": 0, "elapsed": 0.0, "started_at": time.time(),
        }
    thread = threading.Thread(
        target=_run_genstrat_job,
        args=(job_id, settings, language, idea, num_ctx, num_predict, stall_timeout, max_total),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("generate_strategies_job", job_id=job_id))


@app.route("/generate-strategies/job/<job_id>")
def generate_strategies_job(job_id):
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
    if job is None:
        return render_template("generate_strategies_job.html", job_id=job_id, not_found=True), 404
    return render_template("generate_strategies_job.html", job_id=job_id, not_found=False)


@app.route("/generate-strategies/job/<job_id>/status.json")
def generate_strategies_job_status(job_id):
    with _GENSTRAT_JOBS_LOCK:
        job = _GENSTRAT_JOBS.get(job_id)
    if job is None:
        return jsonify({"found": False}), 404
    return jsonify({
        "found": True, "done": job["done"], "error": job["error"], "tokens": job["tokens"], "elapsed": job["elapsed"],
        "code": job["code"], "filename_hint": job["filename_hint"], "language": job["language"], "idea": job["idea"],
    })


@app.route("/generate-strategies/save", methods=["POST"])
def generate_strategies_save():
    """AJAX save -- mirrors the desktop tab's SAVE TO LIBRARY AS DRAFT
    button. Takes whatever's currently in the code editor on the page
    (lets you hand-edit the draft before saving, same as desktop), not
    whatever's stored in the job dict, so edits made after generation
    finished are respected."""
    payload = request.get_json(force=True, silent=True) or {}
    language = payload.get("language", "python")
    code_text = (payload.get("code") or "").rstrip("\n")
    idea = payload.get("idea", "")
    filename_stem = (payload.get("filename") or "").strip() or "generated_strategy"
    filename_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", filename_stem).strip("_") or "generated_strategy"
    ext = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}.get(language, ".py")
    filename = f"{filename_stem}{ext}"
    if not code_text:
        return jsonify({"ok": False, "error": "No code to save."}), 400
    try:
        try:
            save_strategy_text(code_text, filename, language, overwrite=False)
        except StrategyAlreadyExists:
            return jsonify({"ok": False, "error": f"'{filename}' already exists in the library. Rename it and try again."}), 409
        # Always DRAFT, regardless of anything else -- see
        # app.ai.strategy_generator's module docstring for why an
        # AI-generated strategy is never allowed to start higher than this.
        set_strategy_status(language, filename, "draft")
        save_strategy_metadata(language, filename, {"description": f"AI-drafted from idea: {idea[:200]}"})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "filename": filename})


def main():
    # UPGRADE (Sep 2026, QR-code/phone-reachability fix): this used to be
    # a bare `app.run(host="0.0.0.0", port=5000, ...)` with no banner at
    # all -- see app.web.network_info's module docstring for why that was
    # the actual root cause of "the QR code still doesn't generate" (there
    # was never a QR code generated on THIS entry point to begin with, only
    # on the separate `run_web.py` launcher). Now both entry points print
    # the identical LAN-address-and-QR-code banner.
    url = lan_url()
    qr_path = qr_code_file(url)
    print_startup_banner(url, qr_path)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
