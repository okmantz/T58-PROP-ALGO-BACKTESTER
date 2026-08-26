"""
Iterative Refinement report.

A second, self-contained report -- completely separate from the normal
report.html produced by app.reports.generator -- summarizing an
app.optimize.refinement.RefinementResult: the search settings, the
generation-by-generation convergence, the baseline-vs-optimized
comparison, the winning configuration, and an out-of-sample holdout
check on that winning configuration.

Deliberately mirrors the visual style (masthead, cards, tables, charts)
of the main report so the two feel like one product, while remaining a
fully independent file -- running Iterative Refinement never touches or
overwrites report.html.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.optimize.refinement import Candidate, RefinementResult
from app.reports._assets import T58_LOGO_BASE64
from app.reports.charts import svg_histogram, svg_line_chart, svg_multi_line_chart
from app.reports.trade_chart import build_trade_chart_html

FITNESS_METRIC_LABELS = {
    "composite_prop_score": "Composite Prop Score",
    "eval_pass_probability": "Evaluation Pass Probability (%)",
    "first_payout_probability": "First Payout Probability (%)",
    "expected_payout": "Expected Payout ($)",
    "net_profit": "Net Profit ($)",
    "profit_factor": "Profit Factor",
    "sharpe_ratio": "Sharpe Ratio",
}


def _candidate_summary(c: Candidate) -> dict[str, Any]:
    return {
        "generation": c.generation,
        "fitness": c.fitness,
        "statistics": c.statistics,
        "prop_summary": c.prop_summary,
        "mc_summary": c.mc_summary,
    }


def build_refinement_report(
    result: RefinementResult,
    strategy_name: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": {
            "name": strategy_name,
            "instrument": instrument,
            "timeframe": timeframe,
            "backtest_period_start": backtest_period[0],
            "backtest_period_end": backtest_period[1],
        },
        "search_settings": asdict(result.refinement_config),
        "fitness_metric": result.fitness_metric,
        "fitness_metric_label": FITNESS_METRIC_LABELS.get(result.fitness_metric, result.fitness_metric),
        "elapsed_seconds": result.elapsed_seconds,
        "warnings": result.warnings,
        "parameter_count": len(result.genes),
        "parameters": [
            {"label": g.label, "kind": g.kind, "lo": g.lo, "hi": g.hi, "is_int": g.is_int, "base_value": g.base_value}
            for g in result.genes
        ],
        "baseline": _candidate_summary(result.baseline),
        "best": _candidate_summary(result.best),
        "best_config": result.best.config,
        "generation_history": [asdict(g) for g in result.generation_history],
        "leaderboard": [_candidate_summary(c) for c in result.leaderboard[:25]],
        "holdout_comparison": result.holdout_comparison,
    }


def export_refinement_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def export_best_config_json(best_config: dict, path: str | Path) -> Path:
    """
    Writes just the winning Manual Strategy config dict, standalone -- this
    is the file to hand back to a ManualStrategy(...) call, or to eyeball
    directly, independent of the rest of the refinement report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T58 Prop Algo Backtester — Iterative Refinement: {strategy_name}</title>
<style>
  :root {{
    --ink: #14161a; --muted: #6b7280; --line: #e6e8eb; --panel: #ffffff;
    --bg: #f7f8fa; --accent: #B8862F; --accent-dark: #111827;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    margin: 0; color: var(--ink); background: var(--bg);
  }}
  .masthead {{
    background: linear-gradient(135deg, #0b0d10 0%, #14171c 100%);
    color: #e7e9ec; padding: 22px 40px; display:flex; align-items:center; gap:16px;
  }}
  .masthead img {{ height: 40px; display:block; }}
  .masthead .titles h1 {{ margin:0; font-size:18px; font-weight:700; letter-spacing:.01em; border:none; padding:0; color:#f2f3f5;}}
  .masthead .titles .sub {{ margin-top:3px; font-size:11px; color:#9aa1ac; letter-spacing:.03em; text-transform:uppercase; }}
  .content {{ max-width: 1040px; margin: 0 auto; padding: 28px 40px 60px; }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin: 2px 0 0; }}
  h2 {{
    font-size: 13px; margin-top: 34px; margin-bottom: 10px; color: var(--accent-dark);
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700;
    border-bottom: 2px solid var(--accent-dark); padding-bottom: 6px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 4px; background: var(--panel);
           box-shadow: 0 1px 2px rgba(16,24,40,0.04); border-radius: 6px; overflow: hidden; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 8px 12px; font-size: 13px; text-align: left; }}
  tr:last-child td {{ border-bottom: none; }}
  th {{ background: #f1f2f5; font-weight: 600; color: #374151; }}
  .headline {{ display:flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }}
  .card {{
    border: 1px solid var(--line); background: var(--panel); padding: 16px 20px;
    min-width: 180px; flex: 1 1 180px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.05);
  }}
  .card .label {{ font-size:10.5px; color: var(--muted); text-transform:uppercase; letter-spacing:.05em; font-weight:600;}}
  .card .value {{ font-size:26px; font-weight:700; margin-top:6px; color: var(--accent-dark); }}
  .card.win .value {{ color: #1a7f4b; }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  .chart {{ border: 1px solid var(--line); background: var(--panel); padding: 12px;
            margin-top: 8px; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
  .chart-row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .chart-row .chart {{ flex: 1 1 340px; }}
  .chart svg {{ width: 100%; height: auto; display:block; }}
  .warning {{
    background: #FFF7E6; border: 1px solid #F0C36D; color: #6B4E00; border-radius: 8px;
    padding: 14px 18px; margin-top: 14px; font-size: 13px; line-height: 1.5;
  }}
  .warning b {{ display:block; font-size: 12.5px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
  pre.config {{
    background: #0B0D10; color: #d6e3ff; padding: 16px; border-radius: 8px; overflow-x: auto;
    font-size: 12px; line-height: 1.5; max-height: 420px; overflow-y: auto;
  }}
  .footer-note {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line); }}
  .tabs {{ display:flex; gap:4px; margin-top:20px; border-bottom: 2px solid var(--line); }}
  .tab-btn {{
    background:none; border:none; padding:10px 18px; font-size:12.5px; font-weight:700;
    color: var(--muted); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px;
    text-transform:uppercase; letter-spacing:.04em; font-family:inherit;
  }}
  .tab-btn:hover {{ color: var(--accent-dark); }}
  .tab-btn.active {{ color: var(--accent-dark); border-bottom-color: var(--accent); }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  @media print {{ .masthead {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
</style>
</head>
<body>
<div class="masthead">
  <img src="data:image/png;base64,{logo_base64}" alt="T58"/>
  <div class="titles">
    <h1>Prop Algo Backtester — Iterative Refinement Report</h1>
    <div class="sub">Genetic-algorithm-style parameter search &middot; separate from the main strategy report</div>
  </div>
</div>
<div class="content">
<p class="meta">Generated {generated_at} &middot; Base strategy: <b>{strategy_name}</b> &middot;
Instrument: {instrument} &middot; Timeframe: {timeframe} &middot; Period: {period_start} → {period_end} &middot;
Search time: {elapsed_seconds:.1f}s</p>

<div class="warning">
  <b>This is an in-sample search — read before trusting the numbers below</b>
  Iterative Refinement re-runs this strategy dozens of times on the exact same historical
  window, keeping whatever scores best. That process will <em>always</em> find something
  that looks better on this specific data, even when the improvement is pure noise fitted
  to this one price history rather than a real edge. Treat the winning configuration as a
  <b>candidate worth falsification-testing</b> — not as a proven improvement — and weigh the
  Out-of-Sample Holdout Check section below heavily: a real edge should survive it; an
  overfit one usually won't.
</div>

{warnings_html}

<h2>Search Settings</h2>
{settings_table}

<h2>Baseline vs. Optimized</h2>
<div class="headline">
  <div class="card"><div class="label">Fitness Metric</div><div class="value" style="font-size:16px">{fitness_metric_label}</div></div>
  <div class="card"><div class="label">Baseline Fitness</div><div class="value">{baseline_fitness:.2f}</div></div>
  <div class="card win"><div class="label">Best Fitness Found</div><div class="value">{best_fitness:.2f}</div></div>
  <div class="card"><div class="label">Improvement</div><div class="value">{improvement_str}</div></div>
  <div class="card"><div class="label">Parameters Searched</div><div class="value">{parameter_count}</div></div>
  <div class="card"><div class="label">Best Found In</div><div class="value">Gen {best_generation}</div></div>
</div>

{comparison_table}

<h2>Convergence Across Generations</h2>
<p class="muted">Best and mean fitness per generation. A GA that is working should show best-fitness rising
(or holding, once it plateaus) and the population's parameter diversity (second chart) shrinking as the
search converges on a region of the parameter space.</p>
<div class="chart-row">
  <div class="chart">{convergence_chart}</div>
  <div class="chart">{diversity_chart}</div>
</div>

<h2>Parameter Drift: Baseline vs. Optimized</h2>
<p class="muted">Every tunable numeric parameter found in this strategy, its original value, the value in the best-found configuration, and its searched range.</p>
{parameter_drift_table}

<h2>Optimized Strategy — Backtest Statistics</h2>
{best_stats_table}

<h2>Optimized Strategy — Prop-Firm Single-Run Result</h2>
{best_prop_table}

<h2>Optimized Strategy — Monte Carlo Simulation ({n_sims:,} simulated accounts)</h2>
{best_mc_table}
<div class="chart-row">
  <div class="chart">{return_chart}</div>
  <div class="chart">{drawdown_chart}</div>
</div>

<h2>Equity Curve (Optimized Configuration, Same Historical Period)</h2>
<div class="chart">{equity_chart}</div>

{holdout_section}

<div class="tabs">
  <button class="tab-btn active" data-tab="leaderboard" type="button">Final Generation Leaderboard</button>
  <button class="tab-btn" data-tab="config" type="button">Winning Configuration (JSON)</button>
  <button class="tab-btn" data-tab="tradechart" type="button">Trade Visualization</button>
</div>

<div class="tab-panel active" id="tab-leaderboard">
<h2>Final Generation Leaderboard</h2>
<p class="muted">Every candidate in the last generation's population, ranked by fitness. Configs that produced zero trades on this data show as "no trades."</p>
{leaderboard_table}
</div>

<div class="tab-panel" id="tab-config">
<h2>Winning Configuration</h2>
<p class="muted">The full Manual Strategy configuration dict for the best candidate found. Copy this to reuse it directly with <code>ManualStrategy(...)</code>, or as a reference while updating the Strategy tab by hand. A standalone copy is also written to <code>{basename}_best_config.json</code> next to this report.</p>
<pre class="config">{best_config_json}</pre>
</div>

<div class="tab-panel" id="tab-tradechart">
<h2>Interactive Trade Chart (Optimized Configuration)</h2>
{trade_chart_html}
</div>

<p class="footer-note muted">Report generated by T58 Trading — Prop Algo Backtester, Iterative Refinement module. All figures are simulated estimates based on historical data and resampling; past performance and simulated/optimized outcomes do not guarantee future results. This report does not replace the main strategy report, the falsification-kit-style checks, or an out-of-sample forward test.</p>
</div>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<script>
(function() {{
  var tradeChartRendered = false;

  function renderTradeChartIfNeeded() {{
    var el = document.getElementById("t58-refine-trade-chart");
    if (!el || typeof Plotly === "undefined") return;
    if (tradeChartRendered) {{
      Plotly.Plots.resize(el);
      return;
    }}
    var payload = window.__t58TradeChartPayload;
    if (!payload) return;
    Plotly.newPlot(el, payload.data, payload.layout, {{responsive: true, displaylogo: false, scrollZoom: true}});
    tradeChartRendered = true;
  }}

  document.querySelectorAll(".tab-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      document.querySelectorAll(".tab-btn").forEach(function(b) {{ b.classList.remove("active"); }});
      document.querySelectorAll(".tab-panel").forEach(function(p) {{ p.classList.remove("active"); }});
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "tradechart") {{
        renderTradeChartIfNeeded();
      }}
    }});
  }});
}})();
</script>
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


def _warnings_html(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<div class='warning'><b>Note</b>{w}</div>" for w in warnings)
    return items


def _settings_table(report: dict) -> str:
    s = report["search_settings"]
    rows = {
        "Fitness metric": report["fitness_metric_label"],
        "Population size": s["population_size"],
        "Generations": s["generations"],
        "Elite count (carried unchanged each generation)": s["elite_count"],
        "Mutation rate": f"{s['mutation_rate']:.0%} of genes per bred child",
        "Mutation strength": f"±{s['mutation_strength']:.0%} of each parameter's search range",
        "Random immigrants": f"{s['random_immigrants_frac']:.0%} of each generation",
        "Monte Carlo sims during search": f"{s['search_monte_carlo_sims']:,}",
        "Random seed": s["random_seed"],
    }
    return _dict_to_table(rows)


def _pct_improvement(baseline: float, best: float) -> str:
    if baseline in (None, 0) or not isinstance(baseline, (int, float)):
        return "n/a"
    if baseline == float("-inf") or best == float("-inf"):
        return "n/a"
    if baseline == 0:
        return "n/a"
    pct = (best - baseline) / abs(baseline) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _comparison_table(report: dict) -> str:
    baseline_stats = report["baseline"]["statistics"] or {}
    best_stats = report["best"]["statistics"] or {}
    baseline_mc = report["baseline"]["mc_summary"] or {}
    best_mc = report["best"]["mc_summary"] or {}

    def row(label, key, source_a, source_b, fmt="{:,.2f}"):
        a, b = source_a.get(key), source_b.get(key)
        a_str = fmt.format(a) if isinstance(a, (int, float)) and math_finite(a) else "n/a"
        b_str = fmt.format(b) if isinstance(b, (int, float)) and math_finite(b) else "n/a"
        return f"<tr><td>{label}</td><td>{a_str}</td><td>{b_str}</td></tr>"

    def math_finite(v):
        try:
            return v == v and v not in (float("inf"), float("-inf"))
        except TypeError:
            return False

    rows = (
        row("Total trades", "total_trades", baseline_stats, best_stats, "{:,.0f}")
        + row("Net profit", "net_profit", baseline_stats, best_stats, "${:,.2f}")
        + row("Profit factor", "profit_factor", baseline_stats, best_stats, "{:,.2f}")
        + row("Win rate (%)", "win_rate", baseline_stats, best_stats, "{:.1f}")
        + row("Max drawdown (%)", "max_drawdown_pct", baseline_stats, best_stats, "{:.2f}")
        + row("Sharpe ratio", "sharpe_ratio", baseline_stats, best_stats, "{:.2f}")
        + row("Eval pass probability (%)", "evaluation_pass_probability", baseline_mc, best_mc, "{:.1f}")
        + row("First payout probability (%)", "first_payout_probability", baseline_mc, best_mc, "{:.1f}")
        + row("Expected payout ($)", "expected_payout", baseline_mc, best_mc, "${:,.0f}")
        + row("Risk of ruin (%)", "risk_of_ruin_pct", baseline_mc, best_mc, "{:.1f}")
    )
    return f"<table><tr><th>Metric</th><th>Baseline (Current)</th><th>Optimized (Best Found)</th></tr>{rows}</table>"


def _parameter_drift_table(report: dict, best_config: dict) -> str:
    params = report["parameters"]
    rows = []
    for p, genome_val in zip(params, _genome_for(report)):
        base = p["base_value"]
        change = "n/a" if base == 0 else f"{(genome_val - base) / abs(base) * 100:+.1f}%"
        kind = "integer" if p["is_int"] else "continuous"
        rows.append(
            f"<tr><td>{p['label']}</td><td>{base:,.4g}</td><td>{genome_val:,.4g}</td>"
            f"<td>{change}</td><td>[{p['lo']:,.4g}, {p['hi']:,.4g}] ({kind})</td></tr>"
        )
    header = "<tr><th>Parameter</th><th>Baseline</th><th>Optimized</th><th>Change</th><th>Searched Range</th></tr>"
    return f"<table>{header}{''.join(rows)}</table>"


def _genome_for(report: dict) -> list[float]:
    # The winning genome isn't serialized separately in the report dict (only
    # the applied config is), so we recover each gene's optimized value
    # straight from the winning config using its recorded path label. Simpler
    # and safer: re-walk best_config with the same paths used to build
    # `parameters`, via app.optimize.parameter_space.extract_genome on that
    # config and matching by label (labels are unique per config shape).
    from app.optimize.parameter_space import extract_genome

    best_genes = {g.label: g.base_value for g in extract_genome(report["best_config"])}
    return [best_genes.get(p["label"], p["base_value"]) for p in report["parameters"]]


def _leaderboard_table(report: dict) -> str:
    rows = []
    for c in report["leaderboard"]:
        stats = c["statistics"] or {}
        mc = c["mc_summary"] or {}
        if not stats:
            rows.append(f"<tr><td>{c['generation']}</td><td colspan='5'>No trades generated.</td></tr>")
            continue
        pf = stats.get("profit_factor", 0.0)
        pf_str = "∞" if pf == float("inf") else f"{pf:,.2f}"
        rows.append(
            f"<tr><td>{c['generation']}</td><td>{c['fitness']:.2f}</td>"
            f"<td>${stats.get('net_profit', 0):,.2f}</td><td>{pf_str}</td>"
            f"<td>{mc.get('evaluation_pass_probability', 0):.1f}%</td>"
            f"<td>{mc.get('expected_payout', 0):,.0f}</td></tr>"
        )
    header = (
        "<tr><th>Generation</th><th>Fitness</th><th>Net Profit</th><th>Profit Factor</th>"
        "<th>Eval Pass %</th><th>Expected Payout</th></tr>"
    )
    return f"<table>{header}{''.join(rows)}</table>"


def _holdout_section(holdout: dict | None) -> str:
    if not holdout:
        return (
            "<h2>Out-of-Sample Holdout Check</h2>"
            "<p class='muted'>Not available for this run (usually because the optimized configuration "
            "produced too few trades, or there wasn't enough data to split).</p>"
        )
    in_s = holdout.get("in_sample_statistics") or {}
    out_s = holdout.get("holdout_statistics") or {}
    frac = holdout.get("holdout_frac", 0.0) * 100
    in_period = holdout.get("in_sample_period", (None, None))
    out_period = holdout.get("holdout_period", (None, None))

    def row(label, key, fmt="{:,.2f}"):
        a = in_s.get(key)
        b = out_s.get(key)
        a_str = fmt.format(a) if isinstance(a, (int, float)) else "n/a"
        b_str = fmt.format(b) if isinstance(b, (int, float)) else "n/a"
        return f"<tr><td>{label}</td><td>{a_str}</td><td>{b_str}</td></tr>"

    table = (
        "<table><tr><th>Metric</th><th>In-Sample (Search Period)</th><th>Holdout (Never Searched On)</th></tr>"
        + row("Trades", "total_trades", "{:,.0f}")
        + row("Net profit", "net_profit", "${:,.2f}")
        + row("Profit factor", "profit_factor", "{:,.2f}")
        + row("Win rate (%)", "win_rate", "{:.1f}")
        + row("Max drawdown (%)", "max_drawdown_pct", "{:.2f}")
        + row("Sharpe ratio", "sharpe_ratio", "{:.2f}")
        + "</table>"
    )
    return f"""<h2>Out-of-Sample Holdout Check (Optimized Configuration)</h2>
<p class="muted">The final {frac:.0f}% of bars chronologically ({out_period[0]} &rarr; {out_period[1]}) were withheld from the search entirely, then run once with the winning configuration frozen. If this configuration found a real, repeatable edge rather than a fit to noise, the holdout column should look broadly similar to the in-sample column — not dramatically weaker or inverted.</p>
{table}"""


def _downsample(values: list[float], max_points: int = 400) -> list[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def export_refinement_html(
    report: dict,
    path: str | Path,
    best_bt_result=None,
    price_df: pd.DataFrame | None = None,
    basename: str = "refinement",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    best_mc = report["best"]["mc_summary"] or {}
    best_stats = report["best"]["statistics"] or {}
    best_prop = report["best"]["prop_summary"] or {}

    gen_history = report["generation_history"]
    generations = [g["generation"] for g in gen_history]
    best_series = [g["best_fitness"] for g in gen_history]
    mean_series = [g["mean_fitness"] for g in gen_history]
    diversity_series = [g["diversity"] for g in gen_history]

    convergence_chart = svg_multi_line_chart(
        [("Best fitness", best_series, "#1a7f4b"), ("Mean fitness", mean_series, "#2f6fed")],
        title="Fitness by Generation", x_label=f"Generation (0-{generations[-1] if generations else 0})",
        y_label=report["fitness_metric_label"],
    )
    diversity_chart = svg_line_chart(
        diversity_series, title="Population Parameter Diversity by Generation",
        y_label="Normalized diversity", color="#D9A441", fill="#D9A44122",
    )

    if best_bt_result is not None and len(best_bt_result.equity_curve):
        equity_values = _downsample(best_bt_result.equity_curve["equity"].tolist())
        equity_chart = svg_line_chart(equity_values, title="Optimized Configuration — Account Equity", y_label="Equity ($)")
    else:
        equity_chart = "<p>No trades were generated by the optimized configuration.</p>"

    return_dist = (report["best"].get("mc_full_return_distribution") or [])
    drawdown_dist = (report["best"].get("mc_full_drawdown_distribution") or [])
    return_chart = svg_histogram(
        return_dist, title="Monte Carlo: Distribution of Simulated Account Returns (Optimized)",
        x_label="Return (%)", color="#2f6fed",
    )
    drawdown_chart = svg_histogram(
        drawdown_dist, title="Monte Carlo: Distribution of Simulated Max Drawdown (Optimized)",
        x_label="Max Drawdown (%)", color="#F05B63",
    )

    if best_bt_result is not None:
        trade_chart_html = build_trade_chart_html(
            price_df=price_df,
            trades=best_bt_result.trades,
            equity_curve=best_bt_result.equity_curve,
            instrument=report["strategy"]["instrument"],
            div_id="t58-refine-trade-chart",
        )
    else:
        trade_chart_html = '<p class="muted">No backtest result was available to plot.</p>'

    baseline_fitness = report["baseline"]["fitness"]
    best_fitness = report["best"]["fitness"]
    best_generation = report["best"]["generation"]

    html = _HTML_TEMPLATE.format(
        logo_base64=T58_LOGO_BASE64,
        strategy_name=report["strategy"]["name"],
        instrument=report["strategy"]["instrument"],
        timeframe=report["strategy"]["timeframe"],
        period_start=report["strategy"]["backtest_period_start"],
        period_end=report["strategy"]["backtest_period_end"],
        generated_at=report["generated_at"],
        elapsed_seconds=report["elapsed_seconds"],
        warnings_html=_warnings_html(report["warnings"]),
        settings_table=_settings_table(report),
        fitness_metric_label=report["fitness_metric_label"],
        baseline_fitness=baseline_fitness if baseline_fitness != float("-inf") else float("nan"),
        best_fitness=best_fitness if best_fitness != float("-inf") else float("nan"),
        improvement_str=_pct_improvement(baseline_fitness, best_fitness),
        parameter_count=report["parameter_count"],
        best_generation=best_generation,
        comparison_table=_comparison_table(report),
        convergence_chart=convergence_chart,
        diversity_chart=diversity_chart,
        parameter_drift_table=_parameter_drift_table(report, report["best_config"]),
        best_stats_table=_dict_to_table(best_stats) if best_stats else "<p>No trades generated.</p>",
        best_prop_table=_dict_to_table(best_prop) if best_prop else "<p>No trades generated.</p>",
        best_mc_table=_dict_to_table(best_mc) if best_mc else "<p>No trades generated.</p>",
        n_sims=best_mc.get("n_simulations", 0),
        return_chart=return_chart,
        drawdown_chart=drawdown_chart,
        equity_chart=equity_chart,
        holdout_section=_holdout_section(report.get("holdout_comparison")),
        leaderboard_table=_leaderboard_table(report),
        best_config_json=json.dumps(report["best_config"], indent=2, default=str),
        basename=basename,
        trade_chart_html=trade_chart_html,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def generate_refinement_report(
    output_dir: str | Path,
    result: RefinementResult,
    strategy_name: str,
    instrument: str,
    timeframe: str,
    backtest_period: tuple[str, str],
    basename: str = "refinement",
    price_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Builds the refinement report dict and writes JSON + a standalone
    best-config JSON + HTML to output_dir. Mirrors
    app.reports.generator.generate_full_report's return shape.
    """
    report = build_refinement_report(result, strategy_name, instrument, timeframe, backtest_period)

    # Stash the best candidate's full Monte Carlo distributions into the
    # report dict (not part of build_refinement_report's plain-JSON shape,
    # since MonteCarloResult isn't otherwise serialized) purely so the HTML
    # renderer above can chart them without re-threading the object through
    # every helper signature.
    if result.best.mc_result is not None:
        report["best"]["mc_full_return_distribution"] = result.best.mc_result.return_distribution
        report["best"]["mc_full_drawdown_distribution"] = result.best.mc_result.drawdown_distribution

    output_dir = Path(output_dir)
    paths = {
        "json": export_refinement_json(report, output_dir / f"{basename}.json"),
        "best_config_json": export_best_config_json(report["best_config"], output_dir / f"{basename}_best_config.json"),
        "html": export_refinement_html(
            report, output_dir / f"{basename}.html",
            best_bt_result=result.best.bt_result, price_df=price_df, basename=basename,
        ),
    }
    return paths
