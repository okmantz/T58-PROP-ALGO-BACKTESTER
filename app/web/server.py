"""
T58 Prop Algo Backtester — Mobile Web App.

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
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for,
)

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv, import_csv_bytes
from app.data.storage import get_app_base_dir, get_raw_data_dir, list_stored_datasets, store_csv_bytes
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.search.batch_runner import SearchStageConfig, promote_champion, run_search
from app.search.search_report import generate_search_report
from app.search.strategy_space import (
    StrategySpaceError, family_description, generate_search_space, list_families,
)
from app.strategy.base import StrategyError
from app.strategy.library import (
    STRATEGY_TYPES, StrategyAlreadyExists, delete_saved_strategy, export_library_zip_bytes,
    list_saved_strategies, load_strategy_text, rename_saved_strategy, save_strategy_bytes,
    save_strategy_metadata, save_strategy_text,
)
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

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

app = Flask(__name__, static_folder="static", template_folder="templates")


def _build_strategy(mode: str, form, files):
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
        return ManualStrategy(cfg)

    code = (form.get("strategy_code") or "").strip()
    existing_choice = (form.get(f"existing_strategy_{mode}") or "").strip()
    uploaded = files.get("strategy_file")
    save_name = (form.get("strategy_save_name") or "").strip()

    # Precedence: a freshly uploaded file wins over pasted text (the page's
    # JS also mirrors an upload into the textarea, so this only matters if
    # JS didn't run), and pasted/uploaded text wins over a saved-library
    # pick -- matches _resolve_dataset's "most recent upload wins" pattern.
    if uploaded and uploaded.filename:
        code = uploaded.read().decode("utf-8", errors="replace")
        if not save_name:
            save_name = uploaded.filename

    if not code and existing_choice:
        code = load_strategy_text(mode, existing_choice)

    if not code:
        raise StrategyError(
            f"Paste your {mode} strategy code, upload a file, or choose a saved strategy from the library."
        )

    if form.get("strategy_save_to_library"):
        overwrite = bool(form.get("strategy_overwrite"))
        try:
            saved_path = save_strategy_text(code, save_name or "strategy", mode, overwrite=overwrite)
        except StrategyAlreadyExists as exc:
            raise StrategyError(
                f"'{exc.filename}' is already in the {mode} strategy library. "
                'Check "Overwrite existing" and run again to replace it, or change "Save as" to a new name.'
            ) from exc
        description = (form.get("strategy_save_description") or "").strip()
        market = (form.get("strategy_save_market") or "").strip()
        if description or market:
            save_strategy_metadata(mode, saved_path.name, {
                "description": description, "market": market,
            })

    if mode == "python":
        tmp = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
        tmp.write_text(code, encoding="utf-8")
        return PythonStrategy(tmp)
    if mode == "pinescript":
        return PineScriptStrategy(code)
    if mode == "mql5":
        return MQL5Strategy(code)
    raise StrategyError(f"Unknown strategy mode: {mode}")


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
    """{"python": [{"name", "description", "market"}, ...], "pinescript": [...],
    "mql5": [...]} for the page's JS to build the "load from library"
    dropdowns, the client-side search filter, and to prefill the
    description/market fields when a saved strategy is picked."""
    return json.dumps({
        t: [
            {
                "name": s.name,
                "description": s.metadata.get("description", ""),
                "market": s.metadata.get("market", ""),
                "last_run": s.metadata.get("last_run"),
            }
            for s in list_saved_strategies(t)
        ]
        for t in STRATEGY_TYPES
    })


@app.route("/")
def index():
    return render_template(
        "index.html",
        stored_datasets=list_stored_datasets(),
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
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
    target = _redirect_target(request.form)
    try:
        save_strategy_metadata(strategy_type, filename, {"description": description, "market": market})
        notice = f"Saved info for '{filename}'."
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
            return render_template("index.html", error=dataset_error, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400

        form = request.form
        strategy = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)

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
            return render_template("index.html", error=msg, stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400

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

        return render_template(
            "index.html",
            stored_datasets=list_stored_datasets(),
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
        return render_template("index.html", error=str(exc), stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", error=f"Unexpected error: {exc}", stored_datasets=list_stored_datasets(), saved_strategies_json=_saved_strategies_json()), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


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
    instrument: str, db_path: str,
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
        with _SEARCH_JOBS_LOCK:
            job = _SEARCH_JOBS[job_id]
            job["done"] = True
            job["error"] = str(exc)


@app.route("/search")
def search_form():
    return render_template(
        "search.html",
        stored_datasets=list_stored_datasets(),
        families=[{"name": n, "description": family_description(n)} for n in list_families()],
        saved_strategies_json=_saved_strategies_json(),
        strategy_notice=request.args.get("strategy_notice"),
    )


@app.route("/search/start", methods=["POST"])
def search_start():
    form = request.form
    try:
        df, active_label, import_note, dataset_error = _resolve_dataset(form, request.files)
        if dataset_error:
            return render_template(
                "search.html", error=dataset_error, stored_datasets=list_stored_datasets(),
                families=[{"name": n, "description": family_description(n)} for n in list_families()],
                saved_strategies_json=_saved_strategies_json(),
            ), 400

        mode_key = form.get("search_mode", "family_named")
        seed = int(form.get("seed", 42) or 42)
        max_candidates = int(form.get("max_candidates", 200) or 200)

        if mode_key == "single":
            strategy = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
            space = generate_search_space(mode="single", strategy=strategy)
        elif mode_key == "family_grid":
            strategy = _build_strategy(form.get("strategy_mode", "manual"), form, request.files)
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
            fitness_metric=form.get("fitness_metric", "composite_prop_score"),
            workers=int(workers_raw) if workers_raw else None,
            random_seed=seed,
        )
        risk = RiskConfig()
        rules = PropRules()

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
            args=(job_id, df, risk, rules, space, stage_cfg, active_label, db_path),
            daemon=True,
        )
        thread.start()
        return redirect(url_for("search_job", job_id=job_id))

    except StrategySpaceError as exc:
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 400
    except StrategyError as exc:
        return render_template(
            "search.html", error=str(exc), stored_datasets=list_stored_datasets(),
            families=[{"name": n, "description": family_description(n)} for n in list_families()],
            saved_strategies_json=_saved_strategies_json(),
        ), 400
    except Exception as exc:  # noqa: BLE001
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


def main():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
