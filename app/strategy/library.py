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
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.data.storage import get_app_base_dir

STRATEGY_TYPES = ("python", "pinescript", "mql5")

_EXTENSIONS = {
    "python": ".py",
    "pinescript": ".pine",
    "mql5": ".mq5",
}

_META_SUFFIX = ".meta.json"


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


def strategy_exists(strategy_type: str, filename: str) -> bool:
    """Whether a saved strategy with this exact filename already exists —
    check this before saving/renaming so the caller can ask "overwrite or
    save as a new file?" instead of silently getting a " (2)" duplicate."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    return (get_strategy_library_dir(t) / name).exists()


def list_saved_strategies(strategy_type: str | None = None, query: str = "") -> list[StoredStrategy]:
    """Return saved strategies, newest first. Pass strategy_type to filter
    to one language; omit it to list across all three. Pass `query` to
    filter by a case-insensitive substring match against the filename or
    the metadata description/market/tags (empty query returns everything)."""
    types = (_normalize_type(strategy_type),) if strategy_type else STRATEGY_TYPES
    q = query.strip().lower()
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
            out.append(item)
    out.sort(key=lambda s: s.modified, reverse=True)
    return out


def _matches_query(item: StoredStrategy, q: str) -> bool:
    haystacks = [
        item.name,
        str(item.metadata.get("description", "")),
        str(item.metadata.get("market", "")),
        str(item.metadata.get("timeframe", "")),
        " ".join(item.metadata.get("tags", []) or []),
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
# Metadata sidecars — market/timeframe/description/last-run-stats per strategy
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
    timeframe, tags, last_run, etc.), or {} if none has been saved yet."""
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


def record_backtest_result(strategy_type: str, filename: str, stats: dict[str, Any]) -> Path:
    """Convenience wrapper for saving a "last_run" block into a strategy's
    metadata right after a backtest/report finishes, e.g.:

        record_backtest_result("python", "fvg_v1.py", {
            "trades": 173, "net_profit": 34973.31, "win_rate": 51.7,
            "max_dd": 8.4, "report_html": "/reports/fvg_v1_20260827.html",
        })
    """
    return save_strategy_metadata(strategy_type, filename, {"last_run": stats}, merge=True)


# ---------------------------------------------------------------------------
# Backup / export
# ---------------------------------------------------------------------------

def export_library_zip_bytes() -> bytes:
    """Zip the entire strategy library (every language, every strategy file
    plus its metadata sidecar) and return the zip content as bytes -- for
    streaming a download (the web app) without writing a temp file. The
    zip's internal paths are rooted at "strategies/<type>/<file>" so
    unzipping it directly into a repo checkout drops files in the right
    place -- this is also the fix for a packaged .exe's library (which
    lives next to the .exe, not in the git repo): export, unzip into the
    repo's strategies/ folder, commit."""
    base = get_strategy_library_dir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in STRATEGY_TYPES:
            d = get_strategy_library_dir(t)
            for f in sorted(d.iterdir()):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(base.parent)))
    return buffer.getvalue()


def export_library_zip(destination_path: str | Path) -> Path:
    """Same as export_library_zip_bytes but writes straight to
    `destination_path` on disk (the desktop app's EXPORT LIBRARY button).
    Returns the path written."""
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(export_library_zip_bytes())
    return destination_path
