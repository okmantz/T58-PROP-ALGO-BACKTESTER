"""
Payout Probability report -- the full-lifecycle view of
app.prop.survival_engine.PropSurvivalResult, in the same "N simulated
accounts, cascading through stages" shape a trader actually thinks in:

    Start -> Evaluation -> Funded -> Min trading days -> Consistency
           -> Payout #1 -> Payout #2 -> Payout #3 -> ... -> Reset/Continue

Deliberately a separate, self-contained report -- mirrors the visual
style (masthead, cards, tables) of the other standalone reports
(app.reports.refinement_report, app.reports.validation_reports) so it
feels like one product, while never touching or overwriting the main
report.html or any other report this app produces.

This module is a THIN presentation layer: every number it displays
comes from app.prop.survival_engine.run_prop_survival_analysis. It
never recomputes or re-derives a probability -- if a number here looks
wrong, the bug is in survival_engine, not here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, PropSurvivalResult
from app.reports._assets import T58_LOGO_BASE64


# ---------------------------------------------------------------------------
# Report dict assembly
# ---------------------------------------------------------------------------

def build_survival_report(
    result: PropSurvivalResult,
    strategy_name: str,
    instrument: str = "",
    rules: PropRules | None = None,
    cfg: PropSurvivalConfig | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "instrument": instrument,
        "rules": asdict(rules) if rules is not None else {},
        "config": {
            "n_simulations": cfg.n_simulations if cfg else result.n_simulations,
            "method": cfg.method if cfg else None,
            "funding_approval_probability": cfg.funding_approval_probability if cfg else 100.0,
            "reset_max_attempts": cfg.reset_economics.max_attempts if cfg else None,
            "reset_profit_split_pct": cfg.reset_economics.profit_split_pct if cfg else None,
            "reset_evaluation_fee": cfg.reset_economics.evaluation_fee if cfg else None,
        },
        "result": result.to_dict(),
    }


def export_survival_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _funnel_svg(funnel: dict, width: int = 760) -> str:
    """A horizontal cascading bar chart -- one bar per stage, width
    proportional to that stage's count relative to the starting
    n_accounts, so the drop-off between stages is visually obvious in a
    way a table of percentages alone isn't."""
    stages: list[tuple[str, int]] = [
        ("Start", funnel["n_accounts"]),
        ("Passed Evaluation", funnel["passed_evaluation_count"]),
        ("Reached Funded", funnel["reached_funded_count"]),
    ]
    for i, count in enumerate(funnel["payout_counts"], start=1):
        stages.append((f"Payout #{i}", count))

    n0 = max(funnel["n_accounts"], 1)
    bar_h = 28
    gap = 14
    label_w = 190
    max_bar_w = width - label_w - 90
    height = len(stages) * (bar_h + gap) + gap

    colors = ["#5b6472", "#2f6fed", "#1a7f4b"] + ["#B8862F"] * max(0, len(stages) - 3)

    rows = []
    y = gap
    for (label, count), color in zip(stages, colors):
        frac = count / n0
        bar_w = max(2, frac * max_bar_w)
        pct = frac * 100.0
        rows.append(
            f'<text x="0" y="{y + bar_h * 0.68:.1f}" font-size="12" fill="#374151">{label}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>'
            f'<text x="{label_w + bar_w + 8:.1f}" y="{y + bar_h * 0.68:.1f}" font-size="12" fill="#374151">'
            f'{count:,} ({pct:.1f}%)</text>'
        )
        y += bar_h + gap

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;display:block;">{"".join(rows)}</svg>'
    )


def _score_breakdown_table(breakdown: dict) -> str:
    rows = "".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v:+.1f}</td></tr>"
        for k, v in breakdown.items()
    )
    return f"<table><tr><th>Component</th><th>Contribution</th></tr>{rows}</table>"


def _notes_html(notes: list[str]) -> str:
    if not notes:
        return ""
    items = "".join(f"<li>{n}</li>" for n in notes)
    return f'<div class="warning"><b>Read before trusting these numbers</b><ul>{items}</ul></div>'


def _kv_table(d: dict[str, Any], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    rows = []
    for k, v in d.items():
        label = k.replace("_", " ").title()
        kind = fmt.get(k)
        if v is None:
            disp = "n/a"
        elif kind == "pct":
            disp = _fmt_pct(v)
        elif kind == "money":
            disp = _fmt_money(v)
        elif isinstance(v, float):
            disp = f"{v:.2f}"
        else:
            disp = str(v)
        rows.append(f"<tr><td>{label}</td><td>{disp}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>T58 Prop Algo Backtester — Payout Probability: {strategy_name}</title>
<style>
  :root {{
    --ink: #14161a; --muted: #6b7280; --line: #e6e8eb; --panel: #ffffff;
    --bg: #f7f8fa; --accent: #B8862F; --accent-dark: #111827;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; color: var(--ink); background: var(--bg); }}
  .masthead {{ background: linear-gradient(135deg, #0b0d10 0%, #14171c 100%); color: #e7e9ec; padding: 22px 40px; display:flex; align-items:center; gap:16px; }}
  .masthead img {{ height: 40px; display:block; }}
  .masthead .titles h1 {{ margin:0; font-size:18px; font-weight:700; letter-spacing:.01em; color:#f2f3f5; }}
  .masthead .titles .sub {{ margin-top:3px; font-size:11px; color:#9aa1ac; letter-spacing:.03em; text-transform:uppercase; }}
  .content {{ max-width: 1040px; margin: 0 auto; padding: 28px 40px 60px; }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin: 2px 0 0; }}
  h2 {{ font-size: 13px; margin-top: 34px; margin-bottom: 10px; color: var(--accent-dark); text-transform: uppercase; letter-spacing: .06em; font-weight: 700; border-bottom: 2px solid var(--accent-dark); padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 4px; background: var(--panel); box-shadow: 0 1px 2px rgba(16,24,40,0.04); border-radius: 6px; overflow: hidden; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 8px 12px; font-size: 13px; text-align: left; }}
  tr:last-child td {{ border-bottom: none; }}
  th {{ background: #f1f2f5; font-weight: 600; color: #374151; }}
  .headline {{ display:flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }}
  .card {{ border: 1px solid var(--line); background: var(--panel); padding: 16px 20px; min-width: 180px; flex: 1 1 180px; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
  .card .label {{ font-size:10.5px; color: var(--muted); text-transform:uppercase; letter-spacing:.05em; font-weight:600; }}
  .card .value {{ font-size:26px; font-weight:700; margin-top:6px; color: var(--accent-dark); }}
  .card.win .value {{ color: #1a7f4b; }}
  .muted {{ color: var(--muted); font-size: 12px; }}
  .chart {{ border: 1px solid var(--line); background: var(--panel); padding: 16px; margin-top: 8px; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
  .warning {{ background: #FFF7E6; border: 1px solid #F0C36D; color: #6B4E00; border-radius: 8px; padding: 14px 18px; margin-top: 14px; font-size: 13px; line-height: 1.5; }}
  .warning b {{ display:block; font-size: 12.5px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }}
  .warning ul {{ margin: 6px 0 0; padding-left: 18px; }}
  .footer-note {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="masthead">
  <img src="data:image/png;base64,{logo_base64}" alt="T58"/>
  <div class="titles">
    <h1>Prop Algo Backtester — Payout Probability Report</h1>
    <div class="sub">Full-lifecycle prop-firm survival simulation &middot; separate from the main strategy report</div>
  </div>
</div>
<div class="content">
<p class="meta">Generated {generated_at} &middot; Strategy: <b>{strategy_name}</b> &middot; Instrument: {instrument} &middot;
{n_simulations:,} simulated account lifetimes</p>

<div class="headline">
  <div class="card win"><div class="label">Prop Survival Score</div><div class="value">{score:.0f} / 100</div></div>
  <div class="card"><div class="label">Passed Evaluation</div><div class="value">{eval_pass_pct:.1f}%</div></div>
  <div class="card"><div class="label">Reached Funded</div><div class="value">{funded_pct:.1f}%</div></div>
  <div class="card"><div class="label">Payout #1</div><div class="value">{payout1_pct:.1f}%</div></div>
  <div class="card"><div class="label">Net Positive After Resets</div><div class="value">{net_positive_pct:.1f}%</div></div>
</div>

{notes_html}

<h2>Account Lifecycle Funnel</h2>
<p class="muted">Of {n_accounts:,} simulated accounts starting an evaluation, this is how many made it through each
stage of Start &rarr; Evaluation &rarr; Funded &rarr; Payout #1 &rarr; Payout #2 &rarr; ... A strategy worth trading at
a prop firm should hold up reasonably well past the first couple of bars, not fall off a cliff immediately after
"Passed Evaluation."</p>
<div class="chart">{funnel_svg}</div>

<h2>Evaluation-Phase Detail</h2>
{evaluation_table}

<h2>Funded-Phase Detail</h2>
{funded_table}

<h2>Reset / Retry Economics</h2>
<p class="muted">What it actually costs, in real dollars, to keep re-attempting the evaluation after a failure --
and whether that's worth it net of fees, at the configured profit split.</p>
{reset_table}

<h2>Prop Survival Score Breakdown</h2>
<p class="muted">The single 0-100 score above, and the weighted components that produced it -- see
app.prop.survival_engine._compute_survival_score for the full rationale behind each weight.</p>
{score_breakdown_table}

<div class="footer-note">
  Every number on this page comes from app.prop.survival_engine.run_prop_survival_analysis, run on top of the same
  historical trade sequence as this strategy's main backtest -- resampled {n_simulations:,} times using the same
  Monte Carlo machinery (app.monte_carlo.engine) the rest of this app uses, so these numbers are directly
  comparable to (and always at least as conservative as) the plain evaluation-pass probability shown elsewhere.
</div>
</div>
</body>
</html>
"""


def export_survival_html(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result = report["result"]
    funnel = result["funnel"]
    evaluation = result["evaluation"]
    funded = result["funded"]
    reset_econ = result["reset_economics"]

    html = _HTML_TEMPLATE.format(
        logo_base64=T58_LOGO_BASE64,
        strategy_name=report["strategy_name"],
        instrument=report["instrument"] or "n/a",
        generated_at=report["generated_at"],
        n_simulations=result["n_simulations"],
        n_accounts=funnel["n_accounts"],
        score=result["prop_survival_score"],
        eval_pass_pct=evaluation["probability_pass_evaluation"],
        funded_pct=funnel["reached_funded_pct"],
        payout1_pct=funnel["payout_probabilities"][0] if funnel["payout_probabilities"] else 0.0,
        net_positive_pct=reset_econ["probability_net_positive_after_resets"],
        notes_html=_notes_html(result.get("notes", [])),
        funnel_svg=_funnel_svg(funnel),
        evaluation_table=_kv_table(evaluation, fmt={
            "probability_reach_profit_target": "pct", "probability_pass_evaluation": "pct",
            "probability_hit_daily_loss": "pct", "probability_hit_max_drawdown": "pct",
            "consistency_conditional_pass_rate": "pct", "never_recovered_pct": "pct",
            "average_daily_pnl": "money",
        }),
        funded_table=_kv_table(funded, fmt={
            "probability_first_payout": "pct", "probability_second_payout": "pct",
            "probability_third_payout": "pct", "expected_payout_before_failure": "money",
            "median_payout_amount": "money",
        }),
        reset_table=_kv_table(reset_econ, fmt={
            "expected_net_profit_after_resets": "money", "median_net_profit_after_resets": "money",
            "probability_net_positive_after_resets": "pct", "probability_exhausts_resets_without_profit": "pct",
            "expected_fees_paid": "money", "expected_gross_payouts": "money", "profit_split_pct": "pct",
        }),
        score_breakdown_table=_score_breakdown_table(result["score_breakdown"]),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def generate_survival_report(
    output_dir: str | Path,
    result: PropSurvivalResult,
    strategy_name: str,
    instrument: str = "",
    rules: PropRules | None = None,
    cfg: PropSurvivalConfig | None = None,
    basename: str = "survival",
) -> dict[str, Path]:
    """One-call entry point mirroring every other app.reports.generate_*
    function's shape: builds the report dict, writes JSON + HTML to
    output_dir, returns {"html": Path, "json": Path}."""
    output_dir = Path(output_dir)
    report = build_survival_report(result, strategy_name, instrument, rules, cfg)
    json_path = export_survival_json(report, output_dir / f"{basename}.json")
    html_path = export_survival_html(report, output_dir / f"{basename}.html")
    return {"html": html_path, "json": json_path}
