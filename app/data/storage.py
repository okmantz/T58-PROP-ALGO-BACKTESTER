"""Persistent local market-data storage for development and packaged builds."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def get_app_base_dir() -> Path:
    """Return a writable persistent application-data root."""
    if getattr(sys, "frozen", False):
        preferred = Path(sys.executable).resolve().parent
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            probe = preferred / ".t58_write_test"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return preferred
        except OSError:
            local_app_data = os.environ.get("LOCALAPPDATA")
            fallback = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
            return fallback / "T58 Prop Algo Backtester"
    return Path(__file__).resolve().parents[2]


def _seed_bundled_raw_data(raw_dir: Path) -> None:
    """Copy CSVs embedded by PyInstaller into the persistent raw-data folder."""
    if not getattr(sys, "frozen", False):
        return
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    bundled_raw = Path(bundle_root) / "data" / "raw"
    if not bundled_raw.exists():
        return
    for source in bundled_raw.glob("*.csv"):
        destination = raw_dir / source.name
        if not destination.exists():
            try:
                shutil.copy2(source, destination)
            except OSError:
                continue


def get_raw_data_dir() -> Path:
    """Return the persistent data/raw directory, creating and seeding it."""
    raw_dir = get_app_base_dir() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundled_raw_data(raw_dir)
    return raw_dir


@dataclass
class StoredDataset:
    name: str
    path: Path
    size_bytes: int


def list_stored_datasets() -> list[StoredDataset]:
    """Return all CSVs in data/raw/, newest first."""
    raw_dir = get_raw_data_dir()
    files = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [StoredDataset(name=f.name, path=f, size_bytes=f.stat().st_size) for f in files]


def _unique_destination(raw_dir: Path, filename: str) -> Path:
    dest = raw_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (raw_dir / f"{stem} ({i}){suffix}").exists():
        i += 1
    return raw_dir / f"{stem} ({i}){suffix}"


def store_csv_path(source_path: str | Path) -> Path:
    """Copy an external CSV into persistent data/raw/ unless already there."""
    source_path = Path(source_path)
    raw_dir = get_raw_data_dir()
    try:
        if source_path.resolve().parent == raw_dir.resolve():
            return source_path
    except OSError:
        pass
    dest = _unique_destination(raw_dir, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def store_csv_bytes(content: bytes, filename: str) -> Path:
    """Write uploaded CSV bytes into persistent data/raw/."""
    raw_dir = get_raw_data_dir()
    dest = _unique_destination(raw_dir, filename)
    dest.write_bytes(content)
    return dest
