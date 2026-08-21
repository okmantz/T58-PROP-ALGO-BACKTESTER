"""
Final comprehensive report generator.

Combines strategy info, historical backtest results, single-run prop-firm
results, and Monte Carlo results into one report object, then exports it as
JSON, CSV (flattened key metrics), and a self-contained HTML file (which can
be printed/saved to PDF from any browser -- avoids pulling in a heavy PDF
rendering dependency for the MVP).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.engine import BacktestResult
from app.monte_carlo.engine import MonteCarloResult
from app.prop.simulator import AccountSimResult, PropRules, summarize_single_run
from app.reports.charts import svg_histogram, svg_line_chart


def build_report(
    strategy_name: str,
    strategy_source_type: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
    backtest_result: BacktestResult,
    prop_rules: PropRules,
    prop_single_run: AccountSimResult,
    monte_carlo_result: MonteCarloResult,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": {
            "name": strategy_name,
            "source_type": strategy_source_type,
            "instrument": instrument,
            "timeframe": timeframe,
            "backtest_period_start": backtest_period[0],
            "backtest_period_end": backtest_period[1],
        },
        "historical_backtest": {
            "statistics": backtest_result.statistics.to_dict(),
            "total_trades": len(backtest_result.trades),
            "initial_balance": backtest_result.initial_balance,
            "final_equity": float(backtest_result.equity_curve["equity"].iloc[-1]) if len(backtest_result.equity_curve) else backtest_result.initial_balance,
        },
        "prop_firm_rules": asdict(prop_rules),
        "prop_firm_single_run": summarize_single_run(prop_single_run),
        "monte_carlo": monte_carlo_result.to_dict(),
    }


def export_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def export_trades_csv(backtest_result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [t.to_dict() for t in backtest_result.trades]
    if not rows:
        path.write_text("no trades generated\n")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_summary_csv(report: dict, path: str | Path) -> Path:
    """Flat key -> value CSV of the headline metrics for quick spreadsheet review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat: dict[str, Any] = {}
    flat["strategy_name"] = report["strategy"]["name"]
    flat["instrument"] = report["strategy"]["instrument"]
    flat["timeframe"] = report["strategy"]["timeframe"]

    for k, v in report["historical_backtest"]["statistics"].items():
        flat[f"backtest_{k}"] = v

    for k, v in report["prop_firm_single_run"].items():
        flat[f"prop_single_run_{k}"] = v

    mc = report["monte_carlo"]
    for k, v in mc.items():
        if isinstance(v, (dict, list)):
            continue
        flat[f"monte_carlo_{k}"] = v

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in flat.items():
            writer.writerow([k, v])
    return path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T58 Prop Algo Backtester — Report: {strategy_name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; background:#fafafa;}}
  h1 {{ font-size: 22px; border-bottom: 3px solid #111; padding-bottom: 8px; }}
  h2 {{ font-size: 17px; margin-top: 32px; background:#111; color:#fff; padding:6px 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; background:#fff;}}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .headline {{ display:flex; gap: 16px; flex-wrap: wrap; margin-top: 10px;}}
  .card {{ border:1px solid #ddd; background:#fff; padding:14px 18px; min-width:180px; }}
  .card .label {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:.04em;}}
  .card .value {{ font-size:24px; font-weight:700; margin-top:4px;}}
  .muted {{ color:#666; font-size:12px; }}
  .chart {{ border:1px solid #ddd; background:#fff; padding:10px; margin-top:8px; }}
  .chart-row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chart-row .chart {{ flex: 1 1 340px; }}
  .chart svg {{ width: 100%; height: auto; display:block; }}
</style>
</head>
<body>
<h1>T58 Trading — Prop Algo Backtester Report</h1>
<p class="muted">Generated {generated_at} &middot; Strategy: <b>{strategy_name}</b> ({source_type}) &middot;
Instrument: {instrument} &middot; Timeframe: {timeframe} &middot; Period: {period_start} → {period_end}</p>

<h2>The Number That Matters Most</h2>
<div class="headline">
  <div class="card"><div class="label">Evaluation Pass Probability</div><div class="value">{eval_pass:.1f}%</div></div>
  <div class="card"><div class="label">First Payout Probability</div><div class="value">{first_payout:.1f}%</div></div>
  <div class="card"><div class="label">Failure Before Payout</div><div class="value">{failure_before_payout:.1f}%</div></div>
  <div class="card"><div class="label">Median Days to Payout</div><div class="value">{median_days_payout}</div></div>
  <div class="card"><div class="label">Expected Payout</div><div class="value">${expected_payout:,.0f}</div></div>
  <div class="card"><div class="label">Risk of Ruin</div><div class="value">{risk_of_ruin:.1f}%</div></div>
</div>

<h2>Historical Backtest Statistics</h2>
{backtest_table}

<h2>Equity Curve (Historical Backtest)</h2>
<div class="chart">{equity_chart}</div>

<h2>Prop-Firm Rules Used</h2>
{rules_table}

<h2>Prop-Firm Single-Run Result (Historical Sequence)</h2>
{single_run_table}

<h2>Monte Carlo Simulation ({n_sims:,} simulated accounts)</h2>
{monte_carlo_table}

<div class="chart-row">
  <div class="chart">{return_chart}</div>
  <div class="chart">{drawdown_chart}</div>
</div>

<p class="muted">Report generated by T58 Trading — Prop Algo Backtester (MVP). All figures are simulated estimates based on historical data and resampling; past performance and simulated outcomes do not guarantee future results.</p>
</body>
</html>
"""


def _dict_to_table(d: dict) -> str:
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" if isinstance(v, float)
        else f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in d.items()
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _downsample(values: list[float], max_points: int = 400) -> list[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def export_html(report: dict, path: str | Path, backtest_result: BacktestResult | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mc = report["monte_carlo"]
    single = report["prop_firm_single_run"]

    if backtest_result is not None and len(backtest_result.equity_curve):
        equity_values = _downsample(backtest_result.equity_curve["equity"].tolist())
        equity_chart = svg_line_chart(
            equity_values, title="Account Equity Over the Historical Backtest", y_label="Equity ($)",
        )
    else:
        equity_chart = "<p>No trades were generated, so no equity curve is available.</p>"

    return_chart = svg_histogram(
        mc.get("return_distribution", []),
        title="Monte Carlo: Distribution of Simulated Account Returns",
        x_label="Return (%)",
        color="#2f6fed",
        markers={
            "P5": mc["return_percentiles"].get(5),
            "Median": mc["return_percentiles"].get(50),
            "P95": mc["return_percentiles"].get(95),
        },
    )
    drawdown_chart = svg_histogram(
        mc.get("drawdown_distribution", []),
        title="Monte Carlo: Distribution of Simulated Max Drawdown",
        x_label="Max Drawdown (%)",
        color="#F05B63",
        markers={
            "Median": mc["drawdown_percentiles"].get(50),
            "P95": mc["drawdown_percentiles"].get(95),
        },
    )

    html = _HTML_TEMPLATE.format(
        strategy_name=report["strategy"]["name"],
        source_type=report["strategy"]["source_type"],
        instrument=report["strategy"]["instrument"],
        timeframe=report["strategy"]["timeframe"],
        period_start=report["strategy"]["backtest_period_start"],
        period_end=report["strategy"]["backtest_period_end"],
        generated_at=report["generated_at"],
        eval_pass=mc["evaluation_pass_probability"],
        first_payout=mc["first_payout_probability"],
        failure_before_payout=mc["failure_before_payout_probability"],
        median_days_payout=mc["median_days_to_first_payout"] if mc["median_days_to_first_payout"] is not None else "N/A",
        expected_payout=mc["expected_payout"],
        risk_of_ruin=mc["risk_of_ruin_pct"],
        backtest_table=_dict_to_table(report["historical_backtest"]["statistics"]),
        equity_chart=equity_chart,
        rules_table=_dict_to_table(report["prop_firm_rules"]),
        single_run_table=_dict_to_table(single),
        monte_carlo_table=_dict_to_table({k: v for k, v in mc.items() if not isinstance(v, (dict, list))}),
        return_chart=return_chart,
        drawdown_chart=drawdown_chart,
        n_sims=mc["n_simulations"],
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def generate_full_report(
    output_dir: str | Path,
    strategy_name: str,
    strategy_source_type: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
    backtest_result: BacktestResult,
    prop_rules: PropRules,
    prop_single_run: AccountSimResult,
    monte_carlo_result: MonteCarloResult,
    basename: str = "report",
) -> dict[str, Path]:
    """Builds the report dict and writes JSON + summary CSV + trades CSV + HTML to output_dir."""
    report = build_report(
        strategy_name, strategy_source_type, instrument, timeframe, backtest_period,
        backtest_result, prop_rules, prop_single_run, monte_carlo_result,
    )
    output_dir = Path(output_dir)
    paths = {
        "json": export_json(report, output_dir / f"{basename}.json"),
        "summary_csv": export_summary_csv(report, output_dir / f"{basename}_summary.csv"),
        "trades_csv": export_trades_csv(backtest_result, output_dir / f"{basename}_trades.csv"),
        "html": export_html(report, output_dir / f"{basename}.html", backtest_result=backtest_result),
    }
    return paths
