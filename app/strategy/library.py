"""
Persistent strategy library.

Lets a strategy (Python / PineScript / MQL5) be saved *inside* the app's own
data folder — the same persistent, writable, frozen-exe-aware location that
app.data.storage already uses for market-data CSVs — instead of only ever
being pulled from wherever it happens to live on a particular computer or
phone. Once saved, a strategy shows up in the library on every future run,
no re-browsing required. Uploading straight from a device is still fully
supported and is in fact how strategies get into the library in the first
place (save_strategy_path / save_strategy_bytes / save_strategy_text).

Folder layout (created on demand):

    <app base dir>/strategies/python/*.py            (+ *.py.meta.json sidecars)
    <app base dir>/strategies/pinescript/*.pine       (+ *.pine.meta.json sidecars)
    <app base dir>/strategies/mql5/*.mq5              (+ *.mq5.meta.json sidecars)

Mirrors app.data.storage.py's StoredDataset / list_stored_datasets /
store_csv_path / store_csv_bytes naming and behavior on purpose, so the two
subsystems stay easy to reason about side by side.

IMPORTANT — packaged .exe vs. the git repo: get_app_base_dir() (see
app/data/storage.py) resolves next to the running .exe for a frozen/
packaged build, but resolves to the repo root during normal development.
That means a strategy saved from a *built .exe* lands next to that .exe,
not inside your git repo, so it won't show up on GitHub until you copy it
over yourself (or use export_library_zip() below and unzip it into your
repo's strategies/ folder). Saving from the app run out of the repo
(`python -m app.web.server` / `python run_app.py`) writes directly into
the repo's strategies/ folder and needs no extra step.

Metadata sidecars (<file>.meta.json) hold everything besides the raw
source: description, market, timeframe, tags (list[str]), status (one of
STRATEGY_STATUSES -- the "draft -> validated -> live" lifecycle), and
results other features stamp onto a saved strategy after they run against
it: last_run (record_backtest_result), lookahead (record_lookahead_result),
last_search (record_search_result). None of that is required -- an entry
with no metadata at all is just a file with defaults (status "draft").
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.data.storage import get_app_base_dir

STRATEGY_TYPES = ("python", "pinescript", "mql5")

_EXTENSIONS = {
    "python": ".py",
    "pinescript": ".pine",
    "mql5": ".mq5",
}

_META_SUFFIX = ".meta.json"

# The strategy lifecycle this library tracks, cheapest-to-riskiest:
#   draft     -- default; being built/edited, not yet trusted
#   validated -- passed whatever checks you consider sufficient (lookahead
#                check, falsification kit, holdout, etc.)
#   live      -- actually trading / deployed
# Nothing in this module enforces the *order* of transitions; it's a status
# label the person sets deliberately, not a state machine that blocks you.
STRATEGY_STATUSES = ("draft", "validated", "live")
DEFAULT_STATUS = "draft"


class StrategyAlreadyExists(Exception):
    """Raised by the non-overwriting save/rename calls when the destination
    filename is already taken, so callers (UI code) can ask the user
    "overwrite or save as a new file?" instead of silently renaming."""

    def __init__(self, strategy_type: str, filename: str):
        self.strategy_type = strategy_type
        self.filename = filename
        super().__init__(f"A saved {strategy_type} strategy named '{filename}' already exists.")


def _normalize_type(strategy_type: str) -> str:
    t = (strategy_type or "").strip().lower()
    if t not in STRATEGY_TYPES:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. Expected one of {STRATEGY_TYPES}."
        )
    return t


def _normalize_status(status: str) -> str:
    s = (status or "").strip().lower()
    if s not in STRATEGY_STATUSES:
        raise ValueError(f"Unknown status '{status}'. Expected one of {STRATEGY_STATUSES}.")
    return s


def _ensure_extension(filename: str, strategy_type: str) -> str:
    ext = _EXTENSIONS[strategy_type]
    name = Path(filename).name.strip() or f"strategy{ext}"
    if not name.lower().endswith(ext):
        name += ext
    return name


def get_strategy_library_dir(strategy_type: str | None = None) -> Path:
    """Return the persistent strategies/ folder, creating it (and its three
    per-language subfolders) as needed. Pass a strategy_type to get that
    one subfolder directly."""
    base = get_app_base_dir() / "strategies"
    if strategy_type is None:
        for t in STRATEGY_TYPES:
            (base / t).mkdir(parents=True, exist_ok=True)
        return base
    d = base / _normalize_type(strategy_type)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metadata_path(directory: Path, filename: str) -> Path:
    return directory / f"{filename}{_META_SUFFIX}"


@dataclass
class StoredStrategy:
    name: str                       # filename only, e.g. "ny_liquidity_fvg.py"
    strategy_type: str              # "python" | "pinescript" | "mql5"
    path: Path
    size_bytes: int
    modified: float                 # unix timestamp — newest-first sort key
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.metadata.get("status") or DEFAULT_STATUS

    @property
    def tags(self) -> list[str]:
        return list(self.metadata.get("tags") or [])


def strategy_exists(strategy_type: str, filename: str) -> bool:
    """Whether a saved strategy with this exact filename already exists —
    check this before saving/renaming so the caller can ask "overwrite or
    save as a new file?" instead of silently getting a " (2)" duplicate."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    return (get_strategy_library_dir(t) / name).exists()


def list_saved_strategies(
    strategy_type: str | None = None,
    query: str = "",
    tag: str | None = None,
    market: str | None = None,
    status: str | None = None,
) -> list[StoredStrategy]:
    """Return saved strategies, newest first.

    - strategy_type filters to one language; omit for all three.
    - query is a free-text, case-insensitive substring match against the
      filename, description, market, and tags (an OR-style search box).
    - tag/market/status are exact-match filters (case-insensitive) for
      *browsing* -- e.g. every strategy tagged "mean-reversion", or every
      XAUUSD strategy, or everything still in "draft" -- as opposed to
      query's free-text search. All provided filters combine with AND.
    """
    types = (_normalize_type(strategy_type),) if strategy_type else STRATEGY_TYPES
    q = query.strip().lower()
    tag_q = tag.strip().lower() if tag else None
    market_q = market.strip().lower() if market else None
    status_q = _normalize_status(status) if status else None

    out: list[StoredStrategy] = []
    for t in types:
        d = get_strategy_library_dir(t)
        for f in sorted(d.glob(f"*{_EXTENSIONS[t]}")):
            if not f.is_file():
                continue
            stat = f.stat()
            meta = _read_metadata_file(_metadata_path(d, f.name))
            item = StoredStrategy(
                name=f.name, strategy_type=t, path=f,
                size_bytes=stat.st_size, modified=stat.st_mtime, metadata=meta,
            )
            if q and not _matches_query(item, q):
                continue
            if tag_q and tag_q not in [x.lower() for x in item.tags]:
                continue
            if market_q and market_q != str(item.metadata.get("market", "")).strip().lower():
                continue
            if status_q and status_q != item.status:
                continue
            out.append(item)
    out.sort(key=lambda s: s.modified, reverse=True)
    return out


def list_all_tags(strategy_type: str | None = None) -> list[str]:
    """Every distinct tag in use, sorted, for populating a "browse by tag"
    filter control."""
    tags: set[str] = set()
    for item in list_saved_strategies(strategy_type):
        tags.update(item.tags)
    return sorted(tags, key=str.lower)


def list_all_markets(strategy_type: str | None = None) -> list[str]:
    """Every distinct non-empty market value in use, sorted, for populating
    a "browse by market" filter control."""
    markets = {
        str(item.metadata.get("market", "")).strip()
        for item in list_saved_strategies(strategy_type)
        if str(item.metadata.get("market", "")).strip()
    }
    return sorted(markets, key=str.lower)


def _matches_query(item: StoredStrategy, q: str) -> bool:
    haystacks = [
        item.name,
        str(item.metadata.get("description", "")),
        str(item.metadata.get("market", "")),
        " ".join(item.tags),
    ]
    return any(q in h.lower() for h in haystacks)


def _unique_destination(directory: Path, filename: str) -> Path:
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (directory / f"{stem} ({i}){suffix}").exists():
        i += 1
    return directory / f"{stem} ({i}){suffix}"


def save_strategy_path(source_path: str | Path, strategy_type: str, overwrite: bool = False) -> Path:
    """Copy an external strategy file (from BROWSE STRATEGY FILE, a phone's
    share sheet, anywhere on disk) into the persistent library.

    By default (overwrite=False) a name collision raises StrategyAlreadyExists
    rather than silently creating a " (2)" duplicate -- check strategy_exists()
    first and ask the user, or pass overwrite=True once they've confirmed.
    Already-in-place files (re-selecting a file that's already the stored
    copy) are always a no-op, never an error."""
    t = _normalize_type(strategy_type)
    source_path = Path(source_path)
    d = get_strategy_library_dir(t)
    try:
        if source_path.resolve().parent == d.resolve():
            return source_path
    except OSError:
        pass
    dest = d / source_path.name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def save_strategy_bytes(content: bytes, filename: str, strategy_type: str, overwrite: bool = False) -> Path:
    """Write uploaded strategy-file bytes into the persistent library.
    Mirrors app.data.storage.store_csv_bytes. See save_strategy_path() for
    the overwrite/duplicate behavior."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    dest = d / name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, name)
    dest.write_bytes(content)
    return dest


def save_strategy_text(text: str, filename: str, strategy_type: str, overwrite: bool = False) -> Path:
    """Write pasted/edited strategy source text into the persistent library
    (used by the web app, where a strategy may arrive as pasted text rather
    than an uploaded file). See save_strategy_path() for the
    overwrite/duplicate behavior."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    d = get_strategy_library_dir(t)
    dest = d / name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, name)
    dest.write_text(text, encoding="utf-8")
    return dest


def resolve_saved_strategy_path(strategy_type: str, filename: str) -> Path:
    """Resolve `filename` to a path inside strategies/<type>/. Only the
    filename's basename is used (any directory components are stripped),
    which also rules out path-traversal via '../'."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    candidate = d / Path(filename).name
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"No saved {t} strategy named '{filename}'.")
    return candidate


def load_strategy_text(strategy_type: str, filename: str) -> str:
    """Read back a saved strategy's source text by type + filename."""
    return resolve_saved_strategy_path(strategy_type, filename).read_text(encoding="utf-8")


def delete_saved_strategy(strategy_type: str, filename: str) -> None:
    path = resolve_saved_strategy_path(strategy_type, filename)
    path.unlink()
    meta_path = _metadata_path(path.parent, path.name)
    meta_path.unlink(missing_ok=True)


def delete_many(items: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Bulk delete. `items` is an iterable of (strategy_type, filename)
    pairs. Returns (deleted, failed) -- `deleted` is a list of "type/name"
    strings that were removed, `failed` is a list of "type/name: reason"
    strings for ones that couldn't be. One bad item never aborts the rest."""
    deleted, failed = [], []
    for strategy_type, filename in items:
        label = f"{strategy_type}/{filename}"
        try:
            delete_saved_strategy(strategy_type, filename)
            deleted.append(label)
        except (ValueError, FileNotFoundError) as exc:
            failed.append(f"{label}: {exc}")
    return deleted, failed


def rename_saved_strategy(
    strategy_type: str, old_filename: str, new_filename: str, overwrite: bool = False
) -> Path:
    """Rename a saved strategy (and its metadata sidecar, if any) in place.
    Raises StrategyAlreadyExists if new_filename is already taken and
    overwrite is False."""
    t = _normalize_type(strategy_type)
    old_path = resolve_saved_strategy_path(t, old_filename)
    new_name = _ensure_extension(new_filename, t)
    new_path = old_path.parent / new_name

    if new_path == old_path:
        return old_path
    if new_path.exists() and not overwrite:
        raise StrategyAlreadyExists(t, new_name)

    old_meta_path = _metadata_path(old_path.parent, old_path.name)
    new_meta_path = _metadata_path(new_path.parent, new_path.name)

    old_path.rename(new_path)
    if old_meta_path.exists():
        old_meta_path.rename(new_meta_path)
    return new_path


# ---------------------------------------------------------------------------
# Metadata sidecars — market/timeframe/description/tags/status/results
# ---------------------------------------------------------------------------

def _read_metadata_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_strategy_metadata(strategy_type: str, filename: str) -> dict[str, Any]:
    """Return the saved metadata dict for a strategy (description, market,
    tags, status, last_run, lookahead, last_search, etc.), or {} if none
    has been saved yet."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    return _read_metadata_file(_metadata_path(d, name))


def save_strategy_metadata(strategy_type: str, filename: str, metadata: dict[str, Any], merge: bool = True) -> Path:
    """Write (or merge into) a strategy's metadata sidecar. The strategy
    file itself does not need to exist yet -- this can be called right
    after save_strategy_* with the same filename. `merge=True` (default)
    updates only the keys provided, keeping the rest of any existing
    metadata (e.g. updating last_run without clobbering description)."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    path = _metadata_path(d, name)

    existing = _read_metadata_file(path) if merge else {}
    existing.update(metadata)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    return path


def set_strategy_tags(strategy_type: str, filename: str, tags: list[str]) -> Path:
    """Replace a strategy's tag list wholesale (dedupes, strips, drops
    blanks, case-preserving)."""
    seen: dict[str, str] = {}
    for tag in tags:
        clean = tag.strip()
        if clean and clean.lower() not in seen:
            seen[clean.lower()] = clean
    return save_strategy_metadata(strategy_type, filename, {"tags": list(seen.values())})


def set_strategy_status(strategy_type: str, filename: str, status: str) -> Path:
    """Set a strategy's lifecycle status: 'draft', 'validated', or 'live'.
    This is a plain label the person sets deliberately -- nothing here
    enforces moving through the stages in order."""
    return save_strategy_metadata(strategy_type, filename, {"status": _normalize_status(status)})


def record_backtest_result(strategy_type: str, filename: str, stats: dict[str, Any]) -> Path:
    """Convenience wrapper for saving a "last_run" block into a strategy's
    metadata right after a backtest/report finishes, e.g.:

        record_backtest_result("python", "fvg_v1.py", {
            "trades": 173, "net_profit": 34973.31, "win_rate": 51.7,
            "max_dd": 8.4, "report_html": "/reports/fvg_v1_20260827.html",
        })
    """
    return save_strategy_metadata(strategy_type, filename, {"last_run": stats}, merge=True)


def record_lookahead_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping the lookahead checker's verdict onto
    a saved strategy, e.g.:

        record_lookahead_result("python", "fvg_v1.py", {
            "clean": False, "summary": "Lookahead detected at 2 checkpoint(s)",
        })

    This is what makes the library list itself a trust signal instead of
    just a list of filenames -- a strategy that's never been checked, one
    that's clean, and one with a known leak all look different at a glance.
    """
    return save_strategy_metadata(strategy_type, filename, {"lookahead": result}, merge=True)


def record_search_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a Search Lab run's outcome onto the
    base strategy it searched around (single/family_grid modes), e.g.:

        record_search_result("python", "fvg_v1.py", {
            "candidates_tested": 200, "best_fitness": 1.42,
            "fitness_metric": "composite_prop_score",
        })
    """
    return save_strategy_metadata(strategy_type, filename, {"last_search": result}, merge=True)


# ---------------------------------------------------------------------------
# Backup / export
# ---------------------------------------------------------------------------

def _iter_export_files(selection: Iterable[tuple[str, str]] | None):
    """Yield (real_path, arcname) pairs for either the whole library
    (selection=None) or just the given (strategy_type, filename) pairs,
    each strategy file paired with its metadata sidecar if it has one."""
    base = get_strategy_library_dir()
    if selection is None:
        for t in STRATEGY_TYPES:
            d = get_strategy_library_dir(t)
            for f in sorted(d.iterdir()):
                if f.is_file():
                    yield f, str(f.relative_to(base.parent))
        return

    for strategy_type, filename in selection:
        path = resolve_saved_strategy_path(strategy_type, filename)
        yield path, str(path.relative_to(base.parent))
        meta_path = _metadata_path(path.parent, path.name)
        if meta_path.exists():
            yield meta_path, str(meta_path.relative_to(base.parent))


def export_library_zip_bytes(selection: Iterable[tuple[str, str]] | None = None) -> bytes:
    """Zip either the entire strategy library (selection=None) or just the
    given (strategy_type, filename) pairs (bulk-export a selection), and
    return the zip content as bytes -- for streaming a download (the web
    app) without writing a temp file. The zip's internal paths are rooted
    at "strategies/<type>/<file>" so unzipping it directly into a repo
    checkout drops files in the right place -- this is also the fix for a
    packaged .exe's library (which lives next to the .exe, not in the git
    repo): export, unzip into the repo's strategies/ folder, commit."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for real_path, arcname in _iter_export_files(selection):
            zf.write(real_path, arcname=arcname)
    return buffer.getvalue()


def export_library_zip(destination_path: str | Path, selection: Iterable[tuple[str, str]] | None = None) -> Path:
    """Same as export_library_zip_bytes but writes straight to
    `destination_path` on disk (the desktop app's EXPORT LIBRARY button).
    Returns the path written."""
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(export_library_zip_bytes(selection))
    return destination_path
