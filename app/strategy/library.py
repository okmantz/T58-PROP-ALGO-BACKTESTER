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

    <app base dir>/strategies/python/*.py
    <app base dir>/strategies/pinescript/*.pine
    <app base dir>/strategies/mql5/*.mq5

Mirrors app.data.storage.py's StoredDataset / list_stored_datasets /
store_csv_path / store_csv_bytes naming and behavior on purpose, so the two
subsystems stay easy to reason about side by side.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.data.storage import get_app_base_dir

STRATEGY_TYPES = ("python", "pinescript", "mql5")

_EXTENSIONS = {
    "python": ".py",
    "pinescript": ".pine",
    "mql5": ".mq5",
}


def _normalize_type(strategy_type: str) -> str:
    t = (strategy_type or "").strip().lower()
    if t not in STRATEGY_TYPES:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. Expected one of {STRATEGY_TYPES}."
        )
    return t


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


@dataclass
class StoredStrategy:
    name: str            # filename only, e.g. "ny_liquidity_fvg.py"
    strategy_type: str   # "python" | "pinescript" | "mql5"
    path: Path
    size_bytes: int
    modified: float       # unix timestamp — newest-first sort key


def list_saved_strategies(strategy_type: str | None = None) -> list[StoredStrategy]:
    """Return saved strategies, newest first. Pass strategy_type to filter
    to one language; omit it to list across all three."""
    types = (_normalize_type(strategy_type),) if strategy_type else STRATEGY_TYPES
    out: list[StoredStrategy] = []
    for t in types:
        d = get_strategy_library_dir(t)
        for f in sorted(d.glob(f"*{_EXTENSIONS[t]}")):
            if f.is_file():
                stat = f.stat()
                out.append(
                    StoredStrategy(
                        name=f.name, strategy_type=t, path=f,
                        size_bytes=stat.st_size, modified=stat.st_mtime,
                    )
                )
    out.sort(key=lambda s: s.modified, reverse=True)
    return out


def _unique_destination(directory: Path, filename: str) -> Path:
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (directory / f"{stem} ({i}){suffix}").exists():
        i += 1
    return directory / f"{stem} ({i}){suffix}"


def save_strategy_path(source_path: str | Path, strategy_type: str) -> Path:
    """Copy an external strategy file (from BROWSE STRATEGY FILE, a phone's
    share sheet, anywhere on disk) into the persistent library, unless it's
    already there. Mirrors app.data.storage.store_csv_path."""
    t = _normalize_type(strategy_type)
    source_path = Path(source_path)
    d = get_strategy_library_dir(t)
    try:
        if source_path.resolve().parent == d.resolve():
            return source_path
    except OSError:
        pass
    dest = _unique_destination(d, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def save_strategy_bytes(content: bytes, filename: str, strategy_type: str) -> Path:
    """Write uploaded strategy-file bytes into the persistent library.
    Mirrors app.data.storage.store_csv_bytes."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    dest = _unique_destination(d, Path(filename).name)
    dest.write_bytes(content)
    return dest


def save_strategy_text(text: str, filename: str, strategy_type: str) -> Path:
    """Write pasted/edited strategy source text into the persistent library
    (used by the web app, where a strategy may arrive as pasted text rather
    than an uploaded file)."""
    t = _normalize_type(strategy_type)
    ext = _EXTENSIONS[t]
    name = Path(filename).name.strip() or f"strategy{ext}"
    if not name.lower().endswith(ext):
        name += ext
    d = get_strategy_library_dir(t)
    dest = _unique_destination(d, name)
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
    resolve_saved_strategy_path(strategy_type, filename).unlink()
