"""
Global search -- backs the search box in the app-wide top bar (both desktop
and web). One free-text query fans out across the four kinds of things this
app accumulates over time, so "where did I put that" has a single answer box
instead of five different tabs to check by hand:

  strategies  -- app.strategy.library (Strategy Library: python/pinescript/
                 mql5/manual configs), via its own existing free-text query
                 support (matches filename, description, market, tags).
  datasets    -- app.data.storage.list_stored_datasets() (every imported
                 market-data CSV/parquet under data/raw/), filtered by
                 substring match against its relative path.
  reports     -- every saved HTML report under the reports/ output
                 directory (Full Pipeline, Speed Run, Search Lab, etc.),
                 filtered by substring match against the filename. Report
                 filenames already carry the strategy/candidate name (see
                 e.g. full_pipeline.py's report_basename convention), so a
                 filename match is a reasonable proxy for content match
                 without having to open and grep every HTML file on each
                 keystroke.
  runs        -- app.ai.experiment_memory's "T58 Research Memory" SQLite
                 log (one row per Full Pipeline / Quick Optimize / Batch
                 Test / Evolution Lab run), via its existing keyword search.

Deliberately NOT a new search index or database -- every source above
already exists and already supports (or trivially supports) a substring
query; this module is just the fan-out + a common result shape so the UI
layer (desktop header, web nav) doesn't need to know about four different
backends.

This is a read-only, best-effort convenience feature: any one source
failing (e.g. the experiments DB not existing yet on a fresh install) must
never break search against the other three, so each source is wrapped in
its own try/except.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["strategy", "dataset", "report", "run"]


@dataclass
class GlobalSearchResult:
    kind: Kind
    title: str
    subtitle: str
    # For "strategy": the StoredStrategy's path. For "dataset": the CSV/
    # parquet path. For "report": the HTML report path. For "run": None
    # (there's no single file to open -- see experiment_id instead).
    path: Path | None = None
    experiment_id: str | None = None


def _search_strategies(query: str, max_results: int) -> list[GlobalSearchResult]:
    try:
        from app.strategy.library import list_saved_strategies

        out = []
        for s in list_saved_strategies(query=query)[:max_results]:
            market = (s.metadata or {}).get("market") or "any market"
            out.append(GlobalSearchResult(
                kind="strategy",
                title=s.name,
                subtitle=f"{s.strategy_type} strategy -- {s.status_display} -- {market}",
                path=s.path,
            ))
        return out
    except Exception:
        return []


def _search_datasets(query: str, max_results: int) -> list[GlobalSearchResult]:
    try:
        from app.data.storage import list_stored_datasets

        q = query.strip().lower()
        out = []
        for d in list_stored_datasets():
            if q and q not in d.name.lower():
                continue
            size_mb = d.size_bytes / (1024 * 1024)
            out.append(GlobalSearchResult(
                kind="dataset", title=d.name,
                subtitle=f"dataset -- {size_mb:.1f} MB", path=d.path,
            ))
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []


def _search_reports(query: str, max_results: int, reports_dir: Path | None = None) -> list[GlobalSearchResult]:
    try:
        if reports_dir is None:
            reports_dir = Path.cwd() / "reports"
        if not reports_dir.exists():
            return []
        q = query.strip().lower()
        out = []
        # newest-first so a recent run's report surfaces before old ones
        # with a similar name
        candidates = sorted(reports_dir.rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in candidates:
            if q and q not in p.stem.lower():
                continue
            out.append(GlobalSearchResult(
                kind="report", title=p.stem,
                subtitle=f"report -- {p.relative_to(reports_dir)}", path=p,
            ))
            if len(out) >= max_results:
                break
        return out
    except Exception:
        return []


def _search_runs(query: str, max_results: int) -> list[GlobalSearchResult]:
    try:
        from app.ai.experiment_memory import _keyword_search

        out = []
        for exp in _keyword_search(query, max_results):
            out.append(GlobalSearchResult(
                kind="run",
                title=f"{exp.strategy_name} ({exp.origin})",
                subtitle=f"{exp.verdict} -- eval pass {exp.eval_pass_probability:.1f}% -- {exp.instrument}",
                experiment_id=exp.id,
            ))
        return out
    except Exception:
        return []


def global_search(query: str, max_per_kind: int = 6, reports_dir: Path | None = None) -> list[GlobalSearchResult]:
    """Runs all four searches and returns a single flat list, grouped by
    kind (strategies, then datasets, then reports, then runs) rather than
    interleaved -- easier to scan than a relevance-sorted mix across four
    unrelated kinds of thing. Returns [] for a blank/whitespace-only query
    rather than dumping every strategy/dataset/report/run in the app."""
    query = (query or "").strip()
    if not query:
        return []
    results: list[GlobalSearchResult] = []
    results.extend(_search_strategies(query, max_per_kind))
    results.extend(_search_datasets(query, max_per_kind))
    results.extend(_search_reports(query, max_per_kind, reports_dir))
    results.extend(_search_runs(query, max_per_kind))
    return results
