"""Persistent local market-data storage for development and packaged builds."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Every extension the importer (app.data.importer.SUPPORTED_DATA_EXTENSIONS)
# can actually read as tabular market data. list_stored_datasets() and the
# bundled-data seeder below used to hardcode "*.csv" only, which meant a
# .parquet (or .tsv/.txt) file sitting right there in data/raw was invisible
# to the desktop app's stored-dataset list AND the web app's dropdown even
# though selecting/importing it would have worked fine -- kept in sync with
# the importer's own set rather than duplicated as a separate literal so the
# two can't drift apart again.
RAW_DATA_EXTENSIONS = (".csv", ".tsv", ".txt", ".parquet")


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
    """Copy CSVs embedded by PyInstaller into the persistent raw-data folder,
    preserving any instrument subfolders in the bundle rather than flattening
    them (mirrors list_stored_datasets()'s recursive scan below)."""
    if not getattr(sys, "frozen", False):
        return
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    bundled_raw = Path(bundle_root) / "data" / "raw"
    if not bundled_raw.exists():
        return
    for ext in RAW_DATA_EXTENSIONS:
        for source in bundled_raw.rglob(f"*{ext}"):
            destination = raw_dir / source.relative_to(bundled_raw)
            if not destination.exists():
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
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
    name: str   # path relative to data/raw/, POSIX-style (e.g. "EURUSD/EURUSD5.csv")
                # -- so instrument subfolders show up as "EURUSD/EURUSD5.csv"
                # rather than colliding on/hiding behind the bare filename.
    path: Path
    size_bytes: int


def list_stored_datasets() -> list[StoredDataset]:
    """
    Return every importable market-data file under data/raw/ (.csv, .tsv,
    .txt, .parquet -- see RAW_DATA_EXTENSIONS), newest first -- including
    files organized into subfolders (e.g. data/raw/EURUSD/EURUSD5.csv), not
    just ones directly in data/raw/ itself. `name` is the POSIX-style path
    relative to data/raw/, which both the desktop app's stored-dataset list
    and the web app's dropdown display as-is and (for the web app) submit
    back as the selection value -- `get_raw_data_dir() / name` resolves a
    subfolder entry correctly either way, since Path() accepts "/"
    separators on every platform including Windows.
    """
    raw_dir = get_raw_data_dir()
    files: list[Path] = []
    for ext in RAW_DATA_EXTENSIONS:
        files.extend(raw_dir.rglob(f"*{ext}"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        StoredDataset(name=f.relative_to(raw_dir).as_posix(), path=f, size_bytes=f.stat().st_size)
        for f in files
    ]


# A CSV under this many bytes cannot contain a header plus even one data
# row -- it's a placeholder/failed-export artifact, not real market data.
EMPTY_DATASET_BYTES = 32


def _quick_row_count(path: Path) -> int:
    """Cheap row count, without loading the file through pandas -- this
    runs once per file every time the dashboard loads, so it needs to stay
    fast even for multi-megabyte files.

    .parquet is a binary columnar format, not newline-delimited text --
    counting b"\\n" bytes in it (the old behavior here) produced a
    meaningless number instead of a row count. pyarrow can read a
    .parquet file's row count straight out of its footer metadata without
    decoding any actual column data, which is just as cheap as the
    line-count trick for text formats."""
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:
            return 0
    try:
        with open(path, "rb") as f:
            count = sum(1 for _ in f)
        return max(count - 1, 0)
    except OSError:
        return 0


def list_datasets_by_instrument() -> list[dict]:
    """Groups list_stored_datasets() by its top-level data/raw/ subfolder
    (the instrument), for the Dashboard's "Market Data Library" card --
    this is what actually makes the data Owen already has on disk visible
    in the app, independent of whether any backtest has been run yet."""
    raw_dir = get_raw_data_dir()
    groups: dict[str, list[dict]] = {}
    for ds in list_stored_datasets():
        parts = ds.name.split("/")
        instrument = parts[0] if len(parts) > 1 else "(ungrouped)"
        rows = _quick_row_count(ds.path)
        groups.setdefault(instrument, []).append({
            "name": parts[-1],
            "full_name": ds.name,
            "size_bytes": ds.size_bytes,
            "rows": rows,
            "empty": ds.size_bytes <= EMPTY_DATASET_BYTES or rows == 0,
        })

    result = []
    for instrument in sorted(groups.keys()):
        files = sorted(groups[instrument], key=lambda x: x["name"])
        result.append({
            "instrument": instrument,
            "files": files,
            "file_count": len(files),
            "empty_count": sum(1 for f in files if f["empty"]),
            "total_rows": sum(f["rows"] for f in files),
        })
    return result


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
