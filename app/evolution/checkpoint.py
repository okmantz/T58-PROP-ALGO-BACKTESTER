"""
Evolution Lab checkpointing.

Owen's ask: "if I start and stop the generation, it doesn't just start
over from scratch, but has a baseline to work off of." Before this
module, EVERYTHING an EvolutionRunner tracked (generation counter,
current elites used to seed the next generation's children, the
all-time leaderboard, the hypothesis journal) lived only in the
EvolutionRunner instance's memory -- clicking STOP then START again
built a brand new EvolutionRunner with generation=0, elites=[],
leaderboard=[], journal=[]. The knowledge graph (app.evolution.
knowledge_graph) was the only thing that survived a restart, and it
only informs the journal's confidence narrative, not what the next
generation actually is.

This module persists exactly the state needed to resume a run:

    generation           -- the next generation number to run
    elites               -- (spec, meta) pairs used to seed mutated
                             children next generation
    leaderboard          -- serialized EvolutionCandidateRecord summaries
                             (all-time best seen)
    journal              -- the numbered HYPOTHESIS text entries
    data_fingerprint      -- a cheap fingerprint of the market data the
                             run was using, so resuming against a
                             DIFFERENT dataset is detected and refused
                             (silently "resuming" a EURUSD run's elites
                             against freshly loaded XAUUSD data would
                             produce nonsense) rather than silently
                             mixing incompatible runs.

Storage: one JSON file at `path` (default data/evolution/checkpoint.json
under the app's writable base dir), overwritten after every generation.
This is deliberately NOT the knowledge graph (which is an append-only
cross-run log of every candidate ever evaluated, used for similarity
lookups) -- this is "resume exactly where THIS run left off."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.data.storage import get_app_base_dir


def default_checkpoint_path() -> Path:
    return get_app_base_dir() / "data" / "evolution" / "checkpoint.json"


def default_tested_log_path() -> Path:
    return get_app_base_dir() / "data" / "evolution" / "tested_candidates.jsonl"


def data_fingerprint(df: pd.DataFrame) -> dict:
    """A cheap, order-sensitive fingerprint of the market data used for a
    run -- enough to tell "same dataset" from "different dataset" without
    hashing the whole (often multi-million-row) frame."""
    if df is None or df.empty:
        return {"rows": 0}
    try:
        first_ts = str(df["timestamp"].iloc[0])
        last_ts = str(df["timestamp"].iloc[-1])
    except Exception:
        first_ts = last_ts = None
    return {
        "rows": int(len(df)),
        "columns": sorted(str(c) for c in df.columns),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


@dataclass
class EvolutionCheckpoint:
    generation: int = 0
    elites: list[dict] = field(default_factory=list)      # [{"spec": ..., "meta": ...}, ...]
    leaderboard: list[dict] = field(default_factory=list)  # serialized EvolutionCandidateRecord
    journal: list[str] = field(default_factory=list)
    data_fingerprint: dict = field(default_factory=dict)
    saved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "elites": self.elites,
            "leaderboard": self.leaderboard,
            "journal": self.journal,
            "data_fingerprint": self.data_fingerprint,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionCheckpoint":
        return cls(
            generation=int(d.get("generation", 0)),
            elites=list(d.get("elites", [])),
            leaderboard=list(d.get("leaderboard", [])),
            journal=list(d.get("journal", [])),
            data_fingerprint=dict(d.get("data_fingerprint", {})),
            saved_at=str(d.get("saved_at", "")),
        )


def save_checkpoint(checkpoint: EvolutionCheckpoint, path: Optional[Path] = None) -> None:
    path = Path(path) if path else default_checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, default=str)
    # Atomic-ish replace so a crash mid-write can't corrupt the checkpoint
    # a later run would otherwise try (and fail) to resume from.
    tmp_path.replace(path)


def load_checkpoint(path: Optional[Path] = None) -> Optional[EvolutionCheckpoint]:
    path = Path(path) if path else default_checkpoint_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return EvolutionCheckpoint.from_dict(json.load(f))
    except Exception:
        return None


def clear_checkpoint(path: Optional[Path] = None) -> None:
    path = Path(path) if path else default_checkpoint_path()
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def append_tested_rows(rows: list[dict], path: Optional[Path] = None) -> None:
    """Append one JSON-lines row per candidate tested this generation --
    this is the "what was actually tested" record the Evolution Lab tab
    can list/filter/export, independent of whether a candidate made the
    leaderboard."""
    if not rows:
        return
    path = Path(path) if path else default_tested_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_tested_rows(path: Optional[Path] = None, limit: int = 500) -> list[dict]:
    """Return up to the most recent `limit` tested-candidate rows, newest
    last (same order as a log file). Reads the whole file -- this log is
    plain-text JSONL and expected to be at most tens of thousands of
    lines for a multi-day run, so this is simpler and fast enough rather
    than maintaining a separate index."""
    path = Path(path) if path else default_tested_log_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return rows[-limit:]


def clear_tested_log(path: Optional[Path] = None) -> None:
    path = Path(path) if path else default_tested_log_path()
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
