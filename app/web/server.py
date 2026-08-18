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

import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv_bytes
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"

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
    if not code:
        raise StrategyError(f"Paste your {mode} strategy code into the strategy code box.")

    if mode == "python":
        tmp = Path(tempfile.mkdtemp()) / f"strategy_{uuid.uuid4().hex}.py"
        tmp.write_text(code, encoding="utf-8")
        return PythonStrategy(tmp)
    if mode == "pinescript":
        return PineScriptStrategy(code)
    if mode == "mql5":
        return MQL5Strategy(code)
    raise StrategyError(f"Unknown strategy mode: {mode}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json", mimetype="application/manifest+json")


@app.route("/run", methods=["POST"])
def run_pipeline():
    try:
        csv_file = request.files.get("csv_file")
        if not csv_file or not csv_file.filename:
            return render_template("index.html", error="Please choose a market data CSV file."), 400

        import_result = import_csv_bytes(csv_file.read())
        if not import_result.is_valid:
            return render_template("index.html", error="Data import failed: " + "; ".join(import_result.errors)), 400
        df = import_result.dataframe

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
            consistency_rule_pct=float(form.get("consistency", 30)) if form.get("consistency") else None,
            min_trading_days=int(form.get("min_days", 5)),
            payout_threshold_pct=float(form.get("payout_threshold", 0)),
            payout_cap_pct=float(payout_cap) if payout_cap else None,
            payout_frequency_days=int(form.get("payout_freq", 14)),
            required_buffer_pct=float(form.get("buffer", 0)),
        )

        bt_result = run_backtest(df, strategy, risk)

        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, rules)

        n_sims = int(form.get("n_sims", 5000))
        mc_cfg = MonteCarloConfig(n_simulations=min(n_sims, 50_000), method=form.get("mc_method", "bootstrap"))
        mc_result = run_monte_carlo(bt_result.trades, rules, mc_cfg)

        run_id = uuid.uuid4().hex[:10]
        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_full_report(
            output_dir=REPORTS_DIR,
            strategy_name=bt_result.strategy_name,
            strategy_source_type=strategy.source_type,
            instrument=csv_file.filename,
            timeframe="unknown",
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            basename=f"report_{run_id}",
        )

        return render_template(
            "index.html",
            result={
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
        return render_template("index.html", error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", error=f"Unexpected error: {exc}"), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
