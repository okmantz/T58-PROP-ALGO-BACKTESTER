"""
Search Lab leaderboard report.

A batch search produces too many candidates to review as individual HTML
reports (that's the whole point of app.search.results_db). This module
renders one summary page instead: the funnel counts (how many candidates
went in, how many survived each stage and why) plus a ranked table of every
Stage 3 candidate with its key metrics and pass/fail gate flags. Deliberately
plain, dependency-free HTML/CSS (matches this app's existing
app/reports/generator.py convention of hand-written templates, no new
templating dependency).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.search.batch_runner import SearchSummary
from app.search.strategy_space import SearchSpace


def _fmt(v, digits=2):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (int, float)):
        return f"{v:,.{digits}f}"
    return str(v)


def _row_html(row: dict, rank: int) -> str:
    stats = row.get("statistics") or {}
    mc = row.get("mc_summary") or {}
    dsr = row.get("deflated_sharpe") or {}
    passed = bool(row.get("passed_stage3_gate"))
    status = "PASSED" if passed else "FAILED"
    status_class = "pass" if passed else "fail"
    notes = row.get("gate_notes") or ""
    return f"""
    <tr class="{status_class}">
      <td>{rank}</td>
      <td>{row.get('candidate_id', '')}</td>
      <td>{row.get('source_type', 'manual')}</td>
      <td>{row.get('family', '')}</td>
      <td>{_fmt(row.get('composite_score'))}</td>
      <td>{_fmt(dsr.get('probabilistic_sharpe'))}</td>
      <td>{_fmt(stats.get('net_profit'))}</td>
      <td>{_fmt(stats.get('profit_factor'))}</td>
      <td>{_fmt(stats.get('win_rate'), 1)}%</td>
      <td>{_fmt(stats.get('total_trades'), 0)}</td>
      <td>{_fmt(mc.get('evaluation_pass_probability'), 1)}%</td>
      <td>{_fmt(mc.get('first_payout_probability'), 1)}%</td>
      <td>{_fmt(mc.get('risk_of_ruin_pct'), 1)}%</td>
      <td class="{status_class}-badge">{status}</td>
      <td class="notes">{notes}</td>
    </tr>"""


_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>T58 Search Lab -- Leaderboard ({run_id})</title>
<style>
  body {{ background:#0B0D10; color:#E8E6E1; font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif; margin:0; padding:24px; }}
  h1 {{ color:#D4AF37; font-size:20px; margin-bottom:4px; }}
  .sub {{ color:#9AA0A6; font-size:13px; margin-bottom:20px; }}
  .funnel {{ display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }}
  .stage-card {{ background:#14171B; border:1px solid #2A2E33; border-radius:8px; padding:14px 18px; min-width:140px; }}
  .stage-card .n {{ font-size:26px; color:#D4AF37; font-weight:600; }}
  .stage-card .label {{ font-size:11px; color:#9AA0A6; text-transform:uppercase; letter-spacing:.04em; }}
  table {{ border-collapse:collapse; width:100%; font-size:12px; }}
  th {{ text-align:left; padding:8px 10px; background:#14171B; color:#9AA0A6; border-bottom:1px solid #2A2E33; position:sticky; top:0; }}
  td {{ padding:7px 10px; border-bottom:1px solid #1C1F24; }}
  tr.pass td.pass-badge {{ color:#3CCB7F; font-weight:600; }}
  tr.fail td.fail-badge {{ color:#E5533D; font-weight:600; }}
  td.notes {{ color:#9AA0A6; font-size:11px; max-width:320px; }}
  .warn {{ background:#241C10; border:1px solid #7A5A17; color:#E8C77A; padding:12px 16px; border-radius:6px; margin-bottom:20px; font-size:13px; }}
</style>
</head>
<body>
  <h1>T58 Search Lab &mdash; Leaderboard</h1>
  <div class="sub">
    Run {run_id} &middot; mode={mode} &middot; family={family} &middot;
    {total_candidates:,} candidate(s) generated &middot; {instrument} ({timeframe})
  </div>

  <div class="funnel">
    <div class="stage-card"><div class="n">{total_candidates:,}</div><div class="label">Generated</div></div>
    <div class="stage-card"><div class="n">{stage1_survivors:,}</div><div class="label">Stage 1 survivors</div></div>
    <div class="stage-card"><div class="n">{stage2_survivors:,}</div><div class="label">Stage 2 (GA) survivors</div></div>
    <div class="stage-card"><div class="n">{stage3_survivors:,}</div><div class="label">Stage 3 evaluated</div></div>
    <div class="stage-card"><div class="n">{n_passed:,}</div><div class="label">Passed every gate</div></div>
  </div>

  {champion_banner}

  <table>
    <thead>
      <tr>
        <th>#</th><th>Candidate</th><th>Type</th><th>Family</th><th>Composite</th><th>PSR</th>
        <th>Net Profit</th><th>PF</th><th>Win %</th><th>Trades</th>
        <th>Eval Pass %</th><th>1st Payout %</th><th>Risk of Ruin %</th><th>Gate</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <p style="color:#5A5F66; font-size:11px; margin-top:24px;">
    PSR = Probabilistic Sharpe Ratio, deflated against the chance benchmark for
    {n_trials:,} independent trials in this search's own Stage 1 pass (see
    app.search.robustness.deflated_sharpe_ratio). A candidate can show a good raw
    net profit / profit factor and still fail the gate on lookahead, walk-forward,
    or parameter-neighborhood grounds -- see its Notes column.
  </p>
</body>
</html>
"""


def generate_search_report(
    output_dir: str, summary: SearchSummary, space: SearchSpace,
    instrument: str = "unknown", timeframe: str = "unknown",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_passed = sum(1 for r in summary.leaderboard if r.get("passed_stage3_gate"))
    n_trials = summary.stage1_survivors and next(
        (r.get("deflated_sharpe", {}).get("n_trials") for r in summary.leaderboard if r.get("deflated_sharpe")),
        summary.total_candidates,
    ) or summary.total_candidates

    if summary.champion_candidate_id:
        champ = next((r for r in summary.leaderboard if r["candidate_id"] == summary.champion_candidate_id), None)
        champion_banner = (
            f'<div class="warn">Champion: <strong>{summary.champion_candidate_id}</strong> &mdash; '
            f'composite score {_fmt(champ.get("composite_score") if champ else None)}. '
            f'This is the search\'s best surviving candidate, not a guarantee -- promote it through '
            f'Stage 5 (the normal report pipeline) and forward-test on a demo account before risking '
            f'real capital.</div>'
        )
    else:
        champion_banner = (
            '<div class="warn">No candidate passed every Stage 3 gate this run. That is a real, '
            'honest result -- consider it evidence this family/grid does not contain a validated edge '
            'on this data, not a failed search.</div>'
        )

    rows_html = "\n".join(
        _row_html(row, i + 1) for i, row in enumerate(summary.leaderboard)
    ) or '<tr><td colspan="15" style="text-align:center; color:#5A5F66;">No candidates reached Stage 3.</td></tr>'

    html = _TEMPLATE.format(
        run_id=summary.run_id, mode=summary.mode, family=summary.family or "single",
        total_candidates=summary.total_candidates, stage1_survivors=summary.stage1_survivors,
        stage2_survivors=summary.stage2_survivors, stage3_survivors=summary.stage3_survivors,
        n_passed=n_passed, instrument=instrument, timeframe=timeframe,
        champion_banner=champion_banner, rows=rows_html, n_trials=n_trials,
    )

    html_path = out / f"search_leaderboard_{summary.run_id}.html"
    json_path = out / f"search_leaderboard_{summary.run_id}.json"
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(
        {
            "run_id": summary.run_id, "mode": summary.mode, "family": summary.family,
            "total_candidates": summary.total_candidates, "stage1_survivors": summary.stage1_survivors,
            "stage2_survivors": summary.stage2_survivors, "stage3_survivors": summary.stage3_survivors,
            "champion_candidate_id": summary.champion_candidate_id,
            "elapsed_seconds": summary.elapsed_seconds, "leaderboard": summary.leaderboard,
        },
        indent=2, default=str,
    ), encoding="utf-8")

    return {"html": html_path, "json": json_path}
