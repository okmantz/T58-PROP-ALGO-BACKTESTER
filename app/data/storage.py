"""
Persistent local data storage.

Resolves a stable, writable `data/raw/` directory both in normal `python -m
app.main` development runs and inside a PyInstaller-frozen `.exe`, and
provides helpers to list and store market-data CSVs there so datasets a
user has imported (or dropped in manually) are automatically available the
next time the app starts -- no re-upload needed.

Why this matters for the packaged .exe specifically: PyInstaller's
`--onefile` build extracts bundled data into a temporary, read-only folder
(`sys._MEIPASS`) that's wiped after the process exits, so anything written
there (or expected to already be there, like a hand-placed `data/raw/`
folder) will not persist and will not be found. This module always resolves
`data/raw/` next to the actual `.exe` (or, in dev mode, at the project
root) instead -- a normal, writable, persistent folder on disk.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def get_app_base_dir() -> Path:
    """Directory the app should treat as 'home' for user data.

    - Frozen .exe (PyInstaller): the folder containing the .exe itself.
    - Normal `python -m app.main` run: the project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_raw_data_dir() -> Path:
    """The persistent data/raw/ folder, created if it doesn't exist yet."""
    d = get_app_base_dir() / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class StoredDataset:
    name: str
    path: Path
    size_bytes: int


def list_stored_datasets() -> list[StoredDataset]:
    """All CSVs currently sitting in data/raw/, newest first."""
    raw_dir = get_raw_data_dir()
    files = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [StoredDataset(name=f.name, path=f, size_bytes=f.stat().st_size) for f in files]


def _unique_destination(raw_dir: Path, filename: str) -> Path:
    """Avoid clobbering an existing file with different content under the same name."""
    dest = raw_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (raw_dir / f"{stem} ({i}){suffix}").exists():
        i += 1
    return raw_dir / f"{stem} ({i}){suffix}"


def store_csv_path(source_path: str | Path) -> Path:
    """Copy an on-disk CSV into data/raw/ (unless it's already there) and return its stored path."""
    source_path = Path(source_path)
    raw_dir = get_raw_data_dir()
    try:
        if source_path.resolve().parent == raw_dir.resolve():
            return source_path  # already stored, nothing to do
    except OSError:
        pass
    dest = _unique_destination(raw_dir, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def store_csv_bytes(content: bytes, filename: str) -> Path:
    """Write uploaded CSV bytes into data/raw/ and return the stored path."""
    raw_dir = get_raw_data_dir()
    dest = _unique_destination(raw_dir, filename)
    dest.write_bytes(content)
    return dest
