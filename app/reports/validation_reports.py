"""
Lightweight report generation for the Validation Lab / new optimization
modules (walk-forward optimization, CPCV/PBO, sensitivity, portfolio,
multi-objective, walk-forward-aware GA).

Deliberately simpler than app.reports.generator's full single-strategy
report (no trade-by-trade visualization tab, no logo masthead) -- each of
these six features produces its own focused JSON + a compact, readable
HTML summary with the charts that matter for that feature. All six share
the same minimal CSS block below so they look like they belong to the
same app without duplicating the full report generator's styling.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.reports.charts import svg_heatmap, svg_line_chart, svg_multi_line_chart

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 24px;
       background: #0e1013; color: #e7e9ec; }
.card { background: #171a1f; border: 1px solid #2a2e35; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px; }
h1 { font-size: 20px; margin: 0 0 4px 0; }
h2 { font-size: 15px; margin: 0 0 10px 0; color: #cfd3da; }
.sub { color: #9aa0aa; font-size: 12px; margin-bottom: 18px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid #2a2e35; }
th { color: #9aa0aa; font-weight: 600; }
.warn { background: #3a2a12; border: 1px solid #6b4a1a; color: #f0c98a; border-radius: 8px;
        padding: 10px 14px; font-size: 12px; margin-bottom: 12px; }
.pill { display: inline-block; background: #22262e; border-radius: 999px; padding: 2px 10px; font-size: 11px; color: #cfd3da; margin-right: 6px;}
.svgwrap { background: #ffffff; border-radius: 6px; padding: 8px; overflow-x: auto; }
"""


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _write_json(data, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, default=str)
    return path


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{_CSS}</style></head>
<body>
<div class="card"><h1>{title}</h1>
<div class="sub">T58 Trading — Prop Algo Backtester — generated {datetime.now(timezone.utc).isoformat()}</div>
</div>
{body}
</body></html>"""


def _table(rows: list[tuple], headers: tuple[str, str] = ("Metric", "Value")) -> str:
    body_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table><tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>{body_rows}</table>'


# ---------------------------------------------------------------------------
# 1. Walk-forward optimization
# ---------------------------------------------------------------------------

def generate_walk_forward_report(output_dir: str | Path, result, basename: str = "walk_forward_opt") -> dict[str, Path]:
    output_dir = Path(output_dir)
    json_path = _write_json(result.to_summary_dict(), output_dir / f"{basename}.json")

    stats = result.combined_statistics
    equity_vals = list(result.combined_equity_curve["equity"]) if len(result.combined_equity_curve) else []
    equity_svg = svg_line_chart(equity_vals, title="Chained Out-of-Sample Equity Curve", y_label="Equity ($)")

    fold_rows = [
        (f"Fold {f.fold_index}", f"{f.test_period[0][:10]} → {f.test_period[1][:10]}  "
                                  f"({f.test_trade_count} trades, net ${((f.test_statistics or {}).get('net_profit') or 0):,.2f})")
        for f in result.folds
    ]
    summary_rows = [
        ("Window mode", result.window_mode),
        ("Folds completed", len(result.folds)),
        ("Total chained OOS trades", stats.total_trades),
        ("Chained OOS net profit", f"${stats.net_profit:,.2f}"),
        ("Chained OOS profit factor", f"{stats.profit_factor:.2f}" if stats.profit_factor != float("inf") else "inf"),
        ("Chained OOS win rate", f"{stats.win_rate:.1f}%"),
        ("Chained OOS max drawdown", f"{stats.max_drawdown_pct:.1f}%"),
        ("Chained OOS Sharpe", f"{stats.sharpe_ratio:.2f}"),
        ("In-sample reference fitness (full-df GA)", f"{result.in_sample_reference_fitness:.3f}" if result.in_sample_reference_fitness is not None else "n/a"),
        ("Out-of-sample efficiency (OOS / in-sample)", f"{result.out_of_sample_efficiency:.2f}" if result.out_of_sample_efficiency is not None else "n/a"),
    ]
    warnings_html = "".join(f'<div class="warn">{w}</div>' for w in result.warnings)

    body = f"""
{warnings_html}
<div class="card"><h2>Chained Out-of-Sample Result</h2>{_table(summary_rows)}</div>
<div class="card"><h2>Equity Curve (stitched from every fold's held-out test window)</h2>
<div class="svgwrap">{equity_svg}</div>
<p style="font-size:11px;color:#9aa0aa">This is the number that matters: each fold's configuration was chosen
using only that fold's train window, then applied unchanged to data it never optimized against.</p>
</div>
<div class="card"><h2>Per-Fold Breakdown</h2>{_table(fold_rows, ("Fold", "Test window"))}</div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Walk-Forward Optimization Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# 2. CPCV / PBO
# ---------------------------------------------------------------------------

def generate_cpcv_report(output_dir: str | Path, result, basename: str = "cpcv") -> dict[str, Path]:
    output_dir = Path(output_dir)
    json_path = _write_json(result.to_dict(), output_dir / f"{basename}.json")

    rows = [
        ("Metric", result.metric),
        ("Groups / test groups per path", f"{result.n_groups} / {result.n_test_groups}"),
        ("Paths evaluated", result.n_paths),
        ("Mean in-sample metric", f"{result.mean_is_metric:.3f}"),
        ("Mean out-of-sample metric", f"{result.mean_oos_metric:.3f}"),
        ("Median out-of-sample metric", f"{result.median_oos_metric:.3f}"),
        ("Std dev of OOS metric across paths", f"{result.std_oos_metric:.3f}"),
        ("% paths with negative OOS metric", f"{result.pct_paths_oos_negative:.1f}%"),
        ("% paths where OOS < IS", f"{result.pct_paths_oos_below_is:.1f}%"),
        ("IS → OOS degradation", f"{result.degradation:.3f}"),
        ("Robust (OOS holds up vs threshold)", "Yes" if result.is_robust else "No"),
    ]
    verdict = (
        "This strategy's edge appears to survive most combinatorial train/test partitions."
        if result.is_robust else
        "This strategy's edge degrades substantially across combinatorial partitions — "
        "treat the original single-split backtest with real skepticism."
    )
    body = f"""
<div class="card"><h2>Combinatorial Purged Cross-Validation</h2>{_table(rows)}
<p style="font-size:12px;color:#e7e9ec;margin-top:10px">{verdict}</p></div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("CPCV Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def generate_pbo_report(output_dir: str | Path, result, basename: str = "pbo") -> dict[str, Path]:
    output_dir = Path(output_dir)
    json_path = _write_json(result.to_dict(), output_dir / f"{basename}.json")

    rows = [
        ("Metric", result.metric),
        ("Candidates", result.n_candidates),
        ("Groups / test groups per path", f"{result.n_groups} / {result.n_test_groups}"),
        ("Paths evaluated", result.n_paths),
        ("Probability of Backtest Overfitting (PBO)", f"{result.pbo * 100:.1f}%"),
        ("Overall best candidate (by mean IS metric)", f"#{result.overall_best_candidate_index}"),
    ]
    candidate_rows = [
        (f"Candidate #{i}", f"IS={is_v:.3f}  OOS={oos_v:.3f}")
        for i, (is_v, oos_v) in enumerate(zip(result.mean_is_by_candidate, result.mean_oos_by_candidate))
    ]
    interp = (
        "High PBO: the search process shown here is more likely to be selecting statistical "
        "noise than a real, tradeable edge. Don't trust the top-ranked in-sample candidate on "
        "its in-sample numbers alone."
        if result.pbo >= 0.5 else
        "Low-to-moderate PBO: the in-sample winner tends to also perform well out-of-sample "
        "across these candidates — a comparatively good sign, though not proof of a real edge."
    )
    body = f"""
<div class="card"><h2>Probability of Backtest Overfitting</h2>{_table(rows)}
<p style="font-size:12px;color:#e7e9ec;margin-top:10px">{interp}</p>
<p style="font-size:11px;color:#9aa0aa">{result.note}</p></div>
<div class="card"><h2>Per-Candidate Mean IS vs. OOS Metric</h2>{_table(candidate_rows, ("Candidate", "Scores"))}</div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Probability of Backtest Overfitting (PBO) Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# 3. Sensitivity (1D sweeps + optional 2D heatmap)
# ---------------------------------------------------------------------------

def generate_sensitivity_report(
    output_dir: str | Path,
    sweep_results: list,
    heatmap_result=None,
    basename: str = "sensitivity",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    payload = {
        "sweeps": [r.to_dict() for r in sweep_results],
        "heatmap": heatmap_result.to_dict() if heatmap_result else None,
    }
    json_path = _write_json(payload, output_dir / f"{basename}.json")

    sweep_cards = []
    for r in sweep_results:
        chart = svg_multi_line_chart(
            [("metric", r.metric_values, "#2f6fed")],
            title=f"{r.gene_label} sensitivity ({r.metric})",
            x_label="sweep step", y_label=r.metric,
        )
        cliff_note = (
            f'<div class="warn">Cliff detected: largest adjacent-step drop is '
            f'{r.max_pct_drop_between_adjacent_steps:.0f}% — this parameter likely sits on a narrow '
            f'edge rather than a stable plateau.</div>'
            if r.cliff_detected else ""
        )
        rows = [(f"value = {v:g}", f"{m:.3f}") for v, m in zip(r.values, r.metric_values)]
        sweep_cards.append(f"""
<div class="card"><h2>{r.gene_label}</h2>
<div class="pill">base value: {r.base_value:g}</div><div class="pill">base metric: {r.base_metric:.3f}</div>
{cliff_note}
<div class="svgwrap">{chart}</div>
{_table(rows, ("Swept value", "Metric"))}
</div>""")

    heatmap_card = ""
    if heatmap_result is not None:
        hm_svg = svg_heatmap(
            heatmap_result.a_values, heatmap_result.b_values, heatmap_result.grid,
            a_label=heatmap_result.gene_a_label, b_label=heatmap_result.gene_b_label,
            title=f"{heatmap_result.gene_a_label} × {heatmap_result.gene_b_label} ({heatmap_result.metric})",
        )
        heatmap_card = f"""
<div class="card"><h2>2D Sensitivity Heatmap</h2>
<div class="pill">base metric: {heatmap_result.base_metric:.3f}</div>
<div class="svgwrap">{hm_svg}</div>
<p style="font-size:11px;color:#9aa0aa">Green = better, red = worse. A stable strategy shows a
broad green region; a strategy fit to noise shows a narrow green spot surrounded by red.</p>
</div>"""

    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Parameter Sensitivity Report", heatmap_card + "".join(sweep_cards)), encoding="utf-8")
    return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# 4. Portfolio
# ---------------------------------------------------------------------------

def generate_portfolio_report(output_dir: str | Path, result, basename: str = "portfolio") -> dict[str, Path]:
    output_dir = Path(output_dir)
    json_path = _write_json(result.to_summary_dict(), output_dir / f"{basename}.json")

    stats = result.combined_statistics
    equity_vals = list(result.combined_equity_curve["equity"]) if len(result.combined_equity_curve) else []
    equity_svg = svg_line_chart(equity_vals, title="Combined Portfolio Equity Curve", y_label="Equity ($)")

    leg_rows = [
        (l.name, f"nominal={l.nominal_weight:.2f} → final={l.final_weight:.2f}  "
                  f"(avg corr with book: {l.avg_correlation_with_others:+.2f})  "
                  f"trades={l.trade_count}  net=${l.net_profit:,.2f}")
        for l in result.legs
    ]
    corr_rows = [
        (a, ", ".join(f"{b}={v:+.2f}" for b, v in row.items() if b != a))
        for a, row in result.correlation_matrix.items()
    ]
    summary_rows = [
        ("Legs", len(result.legs)),
        ("Combined net profit", f"${stats.net_profit:,.2f}"),
        ("Combined profit factor", f"{stats.profit_factor:.2f}" if stats.profit_factor != float("inf") else "inf"),
        ("Combined max drawdown", f"{stats.max_drawdown_pct:.1f}%"),
        ("Combined Sharpe", f"{stats.sharpe_ratio:.2f}"),
        ("Diversification ratio", f"{result.diversification_ratio:.2f}" if result.diversification_ratio else "n/a"),
    ]
    warnings_html = "".join(f'<div class="warn">{w}</div>' for w in result.warnings)

    mc_card = ""
    if getattr(result, "mc_result", None) is not None:
        mc = result.mc_result
        mc_rows = [
            ("Combined portfolio pass probability", f"{mc.evaluation_pass_probability:.1f}%"),
            ("Combined portfolio first-payout probability", f"{mc.first_payout_probability:.1f}%"),
            ("Combined portfolio risk of ruin", f"{mc.risk_of_ruin_pct:.1f}%"),
            ("Combined portfolio median drawdown", f"{mc.median_drawdown_pct:.1f}%"),
        ]
        mc_card = (
            '<div class="card"><h2>Combined Portfolio Pass Probability</h2>'
            '<p style="font-size:11px;color:#9aa0aa">This is the number that actually answers whether '
            "diversifying across these legs helped: the probability of THIS shared account (trading every "
            "leg together) reaching the profit target before hitting a daily-loss, max-drawdown, or "
            "consistency limit -- as opposed to any one leg's own individual pass probability.</p>"
            f"{_table(mc_rows)}</div>"
        )

    body = f"""
{warnings_html}
<div class="card"><h2>Portfolio Summary</h2>{_table(summary_rows)}</div>
{mc_card}
<div class="card"><h2>Combined Equity Curve</h2><div class="svgwrap">{equity_svg}</div></div>
<div class="card"><h2>Legs (correlation-adjusted risk weight)</h2>{_table(leg_rows, ("Instrument", "Detail"))}</div>
<div class="card"><h2>Correlation Matrix</h2>{_table(corr_rows, ("Instrument", "Correlation with others"))}</div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Multi-Asset Portfolio Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# 5. Multi-objective (Pareto front)
# ---------------------------------------------------------------------------

def generate_multi_objective_report(output_dir: str | Path, result, basename: str = "multi_objective") -> dict[str, Path]:
    output_dir = Path(output_dir)
    payload = {
        "objectives": result.config.objectives,
        "generation_history": [asdict(g) for g in result.generation_history],
        "pareto_front": [
            {"objective_values": dict(zip(result.config.objectives, c.objective_values)),
             "genome": c.genome, "feasible": c.feasible}
            for c in result.pareto_front
        ],
        "elapsed_seconds": result.elapsed_seconds,
        "warnings": result.warnings,
    }
    json_path = _write_json(payload, output_dir / f"{basename}.json")

    front_size_chart = svg_multi_line_chart(
        [("front-0 size", [g.front_0_size for g in result.generation_history], "#2f6fed")],
        title="Pareto Front Size by Generation", x_label="generation", y_label="candidates",
    )
    front_rows = [
        (f"#{i}", ", ".join(f"{obj}={v:.3f}" for obj, v in zip(result.config.objectives, c.objective_values)))
        for i, c in enumerate(result.pareto_front)
    ]
    warnings_html = "".join(f'<div class="warn">{w}</div>' for w in result.warnings)

    body = f"""
{warnings_html}
<div class="card"><h2>Search Summary</h2>
{_table([("Objectives", ", ".join(result.config.objectives)), ("Population size", result.config.population_size),
         ("Generations", result.config.generations), ("Final Pareto front size", len(result.pareto_front)),
         ("Elapsed", f"{result.elapsed_seconds:.1f}s")])}
</div>
<div class="card"><h2>Convergence</h2><div class="svgwrap">{front_size_chart}</div></div>
<div class="card"><h2>Pareto Front (no candidate here is strictly worse than any other on the front)</h2>
{_table(front_rows, ("Candidate", "Objective values"))}
<p style="font-size:11px;color:#9aa0aa">Picking a single winner from this list is a judgment call about
which trade-off matters most to you — that choice is deliberately not made automatically.</p>
</div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Multi-Objective Optimization Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# 6. Walk-forward-aware GA
# ---------------------------------------------------------------------------

def generate_walkforward_ga_report(output_dir: str | Path, result, basename: str = "walkforward_ga") -> dict[str, Path]:
    output_dir = Path(output_dir)
    payload = {
        "n_folds": result.n_folds,
        "window_mode": result.window_mode,
        "best": {"fitness": result.best.fitness, "in_sample_fitness": result.best.in_sample_fitness,
                 "oos_trade_count": result.best.oos_trade_count, "genome": result.best.genome},
        "overfitting_gap": result.overfitting_gap,
        "generation_history": [asdict(g) for g in result.generation_history],
        "warnings": result.warnings,
        "elapsed_seconds": result.elapsed_seconds,
    }
    json_path = _write_json(payload, output_dir / f"{basename}.json")

    convergence_chart = svg_multi_line_chart(
        [
            ("best OOS fitness", [g.best_fitness for g in result.generation_history], "#2f6fed"),
            ("mean OOS fitness", [g.mean_fitness for g in result.generation_history], "#9aa0aa"),
        ],
        title="Walk-Forward-Aware GA Convergence (fitness scored on chained OOS folds only)",
        x_label="generation", y_label="fitness",
    )
    rows = [
        ("Folds used for fitness", result.n_folds),
        ("Window mode", result.window_mode),
        ("Best genome's chained-OOS fitness", f"{result.best.fitness:.3f}"),
        ("Same genome's in-sample (full-df) fitness", f"{result.best.in_sample_fitness:.3f}" if result.best.in_sample_fitness is not None else "n/a"),
        ("Overfitting gap (in-sample − OOS)", f"{result.overfitting_gap:.3f}" if result.overfitting_gap is not None else "n/a"),
        ("Chained-OOS trade count for best genome", result.best.oos_trade_count),
    ]
    warnings_html = "".join(f'<div class="warn">{w}</div>' for w in result.warnings)

    body = f"""
{warnings_html}
<div class="card"><h2>Result</h2>{_table(rows)}</div>
<div class="card"><h2>Convergence</h2><div class="svgwrap">{convergence_chart}</div>
<p style="font-size:11px;color:#9aa0aa">Selection pressure across every generation was based only on
out-of-sample fold performance — a genome that only fit the training data never gets a chance to
look good here the way it would in a plain in-sample GA.</p>
</div>
"""
    html_path = output_dir / f"{basename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_page("Walk-Forward-Aware GA Report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path}
