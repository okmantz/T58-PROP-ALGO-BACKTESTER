"""
Persistent run-history store.

Every time `generate_full_report()` finishes a run (desktop app, web app,
CLI, or the Search Lab batch runner -- they all funnel through that one
function), a compact summary of the run is appended here. The Dashboard
(desktop and web) reads this file to show live stats across every strategy
that has ever been run through the app, with no separate "sync" step.

Kept deliberately small per entry (no raw trade lists) so the file stays
cheap to read/write even after thousands of runs:
  - a downsampled equity curve (<=60 points) for the dashboard's equity chart
  - a 7x24 weekday x hour PnL grid for the aggregate heatmap
  - headline stats or the scorecard table

Recording is always best-effort: a failure here must never break a
backtest run that would otherwise have succeeded.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.data.storage import get_app_base_dir

MAX_ENTRIES = 1000
EQUITY_POINTS = 60


def history_path() -> Path:
    path = get_app_base_dir() / "data" / "run_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _downsample(values: list[float], n: int = EQUITY_POINTS) -> list[float]:
    if len(values) <= n:
        return [round(float(v), 2) for v in values]
    step = len(values) / n
    return [round(float(values[min(int(i * step), len(values) - 1)]), 2) for i in range(n)]


def _heatmap_grid(trades, equity_before: float) -> list[list[float]]:
    """7 (Mon..Sun) x 24 (hour) grid of summed PnL, so many runs can be
    combined later with plain elementwise addition."""
    grid = [[0.0 for _ in range(24)] for _ in range(7)]
    for t in trades:
        try:
            ts = t.entry_time
            weekday = int(ts.weekday())
            hour = int(ts.hour)
            grid[weekday][hour] += float(t.pnl)
        except Exception:
            continue
    return [[round(v, 2) for v in row] for row in grid]


def _lookup_tags(strategy_name: str, source_type: str) -> list[str]:
    """Best-effort tag lookup from the strategy library. Manual strategies
    and one-off uploads that were never saved to the library simply get no
    tags -- the graph still places them by instrument alone."""
    try:
        from app.strategy.library import list_saved_strategies
    except Exception:
        return []
    try:
        stype = source_type if source_type in ("python", "pinescript", "mql5") else None
        if not stype:
            return []
        for item in list_saved_strategies(stype):
            if Path(item.filename).stem == strategy_name or item.filename == strategy_name:
                return list(item.metadata.get("tags") or [])
    except Exception:
        pass
    return []


def load_runs() -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_runs(runs: list[dict[str, Any]]) -> None:
    path = history_path()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(runs[-MAX_ENTRIES:], f, indent=2, default=str)
    tmp.replace(path)


def record_run(report: dict, paths: dict, backtest_result=None) -> None:
    """Append a compact summary of a just-generated report. Never raises --
    a history-recording failure must not surface as a broken backtest run."""
    try:
        strat = report.get("strategy", {})
        stats = report.get("historical_backtest", {}).get("statistics", {})
        prop_single = report.get("prop_firm_single_run", {})
        mc = report.get("monte_carlo", {})

        equity_curve: list[float] = []
        heatmap = [[0.0] * 24 for _ in range(7)]
        if backtest_result is not None:
            try:
                equity_curve = _downsample(list(backtest_result.equity_curve["equity"].astype(float)))
            except Exception:
                equity_curve = []
            try:
                heatmap = _heatmap_grid(backtest_result.trades, backtest_result.initial_balance)
            except Exception:
                pass

        entry = {
            "run_id": uuid.uuid4().hex[:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_name": strat.get("name", "Strategy"),
            "source_type": strat.get("source_type", "unknown"),
            "instrument": (strat.get("instrument") or "Unknown").strip() or "Unknown",
            "timeframe": strat.get("timeframe", "unknown"),
            "tags": _lookup_tags(strat.get("name", ""), strat.get("source_type", "")),
            "trades": int(stats.get("total_trades", 0) or 0),
            "net_profit": float(stats.get("net_profit", 0.0) or 0.0),
            "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
            "max_drawdown_pct": float(stats.get("max_drawdown_pct", 0.0) or 0.0),
            "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0) if math.isfinite(stats.get("profit_factor", 0.0) or 0.0) else 0.0,
            "sharpe_ratio": float(stats.get("sharpe_ratio", 0.0) or 0.0) if math.isfinite(stats.get("sharpe_ratio", 0.0) or 0.0) else 0.0,
            "eval_pass_probability": float(mc.get("evaluation_pass_probability", 0.0) or 0.0),
            "risk_of_ruin_pct": float(mc.get("risk_of_ruin_pct", 0.0) or 0.0),
            "expected_payout": float(mc.get("expected_payout", 0.0) or 0.0),
            "single_run_passed": bool(prop_single.get("evaluation_pass_pct", 0.0) == 100.0),
            "equity_curve": equity_curve,
            "heatmap": heatmap,
            "report_html": str(paths.get("html", "")),
        }

        runs = load_runs()
        runs.append(entry)
        _save_runs(runs)
    except Exception:
        pass


def _strategy_key(entry: dict) -> tuple[str, str]:
    return (entry.get("strategy_name", ""), entry.get("source_type", ""))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(t.lower() for t in a), set(t.lower() for t in b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_graph(latest_rows: list[dict]) -> dict:
    """Strategy-universe graph: nodes clustered by instrument, edges weighted
    by shared tags/indicators plus a same-instrument/timeframe bonus."""
    nodes = []
    instruments = sorted({r["instrument"] for r in latest_rows})
    for r in latest_rows:
        nodes.append({
            "id": f"{r['strategy_name']}::{r['source_type']}",
            "name": r["strategy_name"],
            "instrument": r["instrument"],
            "cluster": instruments.index(r["instrument"]),
            "passed": r["single_run_passed"],
            "sharpe": r["sharpe_ratio"],
        })

    edges = []
    for i, a in enumerate(latest_rows):
        scored = []
        for j, b in enumerate(latest_rows):
            if i == j:
                continue
            weight = _jaccard(a.get("tags", []), b.get("tags", []))
            if a["instrument"] == b["instrument"]:
                weight += 0.3
            if a["timeframe"] == b["timeframe"] and a["timeframe"] != "unknown":
                weight += 0.15
            if weight > 0.15:
                scored.append((weight, j))
        scored.sort(reverse=True)
        for weight, j in scored[:3]:
            pair = tuple(sorted((i, j)))
            edges.append((pair, round(min(weight, 1.0), 2)))

    seen = {}
    for pair, weight in edges:
        seen[pair] = max(weight, seen.get(pair, 0.0))
    edge_list = [{"source": p[0], "target": p[1], "weight": w} for p, w in seen.items()]

    return {"nodes": nodes, "edges": edge_list, "instruments": instruments}


def dashboard_data() -> dict:
    """Aggregates run_history.json into everything the Dashboard needs."""
    runs = load_runs()
    if not runs:
        return {
            "total_strategies": 0, "total_runs": 0, "pass_rate": 0.0,
            "best": None, "strategies": [], "graph": {"nodes": [], "edges": [], "instruments": []},
            "heatmap": [[0.0] * 24 for _ in range(7)], "equity_series": [],
        }

    latest_by_key: dict[tuple, dict] = {}
    run_counts: dict[tuple, int] = {}
    for r in runs:
        key = _strategy_key(r)
        run_counts[key] = run_counts.get(key, 0) + 1
        prev = latest_by_key.get(key)
        if prev is None or r["timestamp"] >= prev["timestamp"]:
            latest_by_key[key] = r

    latest_rows = list(latest_by_key.values())
    for r in latest_rows:
        r["run_count"] = run_counts[_strategy_key(r)]

    passed = sum(1 for r in latest_rows if r["single_run_passed"])
    pass_rate = round(100.0 * passed / len(latest_rows), 1) if latest_rows else 0.0
    best = max(latest_rows, key=lambda r: r["sharpe_ratio"], default=None)

    combined_heatmap = [[0.0] * 24 for _ in range(7)]
    for r in runs:
        hm = r.get("heatmap") or []
        for wd in range(min(7, len(hm))):
            for hr in range(min(24, len(hm[wd]))):
                combined_heatmap[wd][hr] += hm[wd][hr]

    top_equity = sorted(latest_rows, key=lambda r: r["net_profit"], reverse=True)[:5]
    equity_series = [
        {"name": r["strategy_name"], "passed": r["single_run_passed"], "values": r["equity_curve"]}
        for r in top_equity if r["equity_curve"]
    ]

    strategies = sorted(latest_rows, key=lambda r: r["timestamp"], reverse=True)

    return {
        "total_strategies": len(latest_rows),
        "total_runs": len(runs),
        "pass_rate": pass_rate,
        "best": best,
        "strategies": strategies,
        "graph": build_graph(latest_rows),
        "heatmap": combined_heatmap,
        "equity_series": equity_series,
    }
