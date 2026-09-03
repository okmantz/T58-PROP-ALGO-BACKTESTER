"""
SQLite-backed results store for the Search Lab.

Thousands of one-off HTML reports (the app's current single-strategy output
format) is not a usable way to review a batch search -- there is no way to
ask "show me the top 20 that passed every validation gate, sorted by
deflated Sharpe" against a pile of HTML files. This module gives every
candidate evaluated in a search run a durable row, queryable as a
leaderboard, and resumable (a crashed or interrupted run's completed stages
are still on disk).

Uses the Python standard library's sqlite3 only -- no new dependency.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_runs (
    run_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    mode TEXT NOT NULL,
    family TEXT,
    instrument TEXT,
    timeframe TEXT,
    total_candidates INTEGER,
    config_json TEXT,
    status TEXT DEFAULT 'running',
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS candidates (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    family TEXT,
    source_type TEXT,
    config_json TEXT,
    code_text TEXT,
    code_extension TEXT,
    params_json TEXT,
    statistics_json TEXT,
    prop_summary_json TEXT,
    mc_summary_json TEXT,
    walk_forward_json TEXT,
    robustness_json TEXT,
    deflated_sharpe_json TEXT,
    lookahead_json TEXT,
    cost_ladder_json TEXT,
    quick_score REAL,
    fitness REAL,
    composite_score REAL,
    passed_stage1 INTEGER DEFAULT 0,
    passed_stage2 INTEGER DEFAULT 0,
    passed_stage3_gate INTEGER DEFAULT 0,
    gate_notes TEXT,
    error TEXT,
    created_at REAL,
    UNIQUE(run_id, candidate_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_candidates_run_stage ON candidates(run_id, stage);
CREATE INDEX IF NOT EXISTS idx_candidates_composite ON candidates(run_id, stage, composite_score);
"""

_JSON_FIELDS = {
    "config_json": "config",
    "params_json": "params",
    "statistics_json": "statistics",
    "prop_summary_json": "prop_summary",
    "mc_summary_json": "mc_summary",
    "walk_forward_json": "walk_forward",
    "robustness_json": "robustness",
    "deflated_sharpe_json": "deflated_sharpe",
    "lookahead_json": "lookahead",
    "cost_ladder_json": "cost_ladder",
}

_CANDIDATE_COLUMNS = [
    "candidate_id", "run_id", "stage", "family", "source_type", "config_json", "code_text",
    "code_extension", "params_json", "statistics_json", "prop_summary_json", "mc_summary_json",
    "walk_forward_json", "robustness_json", "deflated_sharpe_json", "lookahead_json",
    "cost_ladder_json", "quick_score", "fitness", "composite_score", "passed_stage1",
    "passed_stage2", "passed_stage3_gate", "gate_notes", "error", "created_at",
]


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    cols = [d[0] for d in cursor.description]
    record = dict(zip(cols, row))
    for json_col, plain_key in _JSON_FIELDS.items():
        raw = record.get(json_col)
        record[plain_key] = json.loads(raw) if raw else None
    return record


class ResultsDB:
    """
    One SQLite file per Search Lab run (or shared across runs -- run_id
    scopes every query). Not safe for concurrent writers from multiple
    processes; by design, only the main orchestrating process
    (app.search.batch_runner.run_search) ever writes -- worker processes
    return plain dicts over the process pool and the main process persists
    them, so there is never a second writer to coordinate with.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self, run_id: str, mode: str, family: str | None, instrument: str,
        timeframe: str, total_candidates: int, config: dict,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO search_runs "
            "(run_id, created_at, mode, family, instrument, timeframe, total_candidates, config_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')",
            (run_id, time.time(), mode, family, instrument, timeframe, total_candidates,
             json.dumps(config, default=str)),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self._conn.execute(
            "UPDATE search_runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, time.time(), run_id),
        )
        self._conn.commit()

    def run_summary(self, run_id: str) -> dict | None:
        cur = self._conn.execute("SELECT * FROM search_runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        record = dict(zip(cols, row))
        if record.get("config_json"):
            record["config"] = json.loads(record["config_json"])
        record["stage_counts"] = {
            stage: self._conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE run_id = ? AND stage = ?", (run_id, stage),
            ).fetchone()[0]
            for stage in ("stage1", "stage2", "stage3")
        }
        return record

    def list_runs(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM search_runs ORDER BY created_at DESC LIMIT ?", (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def insert_candidate(self, run_id: str, candidate_id: str, stage: str, record: dict) -> None:
        values: dict = {
            "candidate_id": candidate_id, "run_id": run_id, "stage": stage,
            "created_at": time.time(),
        }
        for key in (
            "family", "source_type", "quick_score", "fitness", "composite_score",
            "passed_stage1", "passed_stage2", "passed_stage3_gate", "gate_notes", "error",
            "code_text", "code_extension",
        ):
            values[key] = record.get(key)
        for json_col, plain_key in _JSON_FIELDS.items():
            v = record.get(plain_key)
            values[json_col] = json.dumps(v, default=str) if v is not None else None

        placeholders = ", ".join("?" for _ in _CANDIDATE_COLUMNS)
        col_list = ", ".join(_CANDIDATE_COLUMNS)
        # INSERT OR REPLACE + the UNIQUE(run_id, candidate_id, stage) constraint in the
        # schema means re-inserting the same candidate at the same stage (e.g. re-running
        # Stage 3 on a candidate) overwrites its previous row instead of duplicating it.
        self._conn.execute(
            f"INSERT OR REPLACE INTO candidates ({col_list}) VALUES ({placeholders})",
            tuple(values.get(c) for c in _CANDIDATE_COLUMNS),
        )
        self._conn.commit()

    def get_candidate(self, candidate_id: str, run_id: str | None = None, stage: str | None = None) -> dict | None:
        q = "SELECT * FROM candidates WHERE candidate_id = ?"
        params: list = [candidate_id]
        if run_id is not None:
            q += " AND run_id = ?"
            params.append(run_id)
        if stage is not None:
            q += " AND stage = ?"
            params.append(stage)
        q += " ORDER BY row_id DESC LIMIT 1"
        cur = self._conn.execute(q, params)
        row = cur.fetchone()
        return _row_to_dict(cur, row) if row else None

    def leaderboard(self, run_id: str, stage: str = "stage3", top_n: int = 25, only_passed: bool = False) -> list[dict]:
        q = "SELECT * FROM candidates WHERE run_id = ? AND stage = ?"
        params: list = [run_id, stage]
        if only_passed:
            q += " AND passed_stage3_gate = 1"
        q += " ORDER BY composite_score DESC LIMIT ?"
        params.append(top_n)
        cur = self._conn.execute(q, params)
        return [_row_to_dict(cur, row) for row in cur.fetchall()]

    def live_leaderboard(self, run_id: str, top_n: int = 20) -> list[dict]:
        """Best-known row per candidate_id across EVERY stage reached SO
        FAR, ranked by whichever score that stage actually has (composite_score
        for stage3, fitness for stage2, quick_score for stage1) -- meant to
        be polled WHILE a run is still in progress (see poll_live_leaderboard
        below for polling from a separate thread/connection), not only
        after leaderboard()'s stage3-only view is available. Each
        candidate_id appears once, at whichever stage it's currently
        furthest into."""
        cur = self._conn.execute(
            """
            SELECT c.* FROM candidates c
            INNER JOIN (
                SELECT candidate_id,
                       MAX(CASE stage WHEN 'stage3' THEN 3 WHEN 'stage2' THEN 2 WHEN 'stage1' THEN 1 ELSE 0 END) AS stage_rank
                FROM candidates WHERE run_id = ?
                GROUP BY candidate_id
            ) latest
            ON c.candidate_id = latest.candidate_id
               AND (CASE c.stage WHEN 'stage3' THEN 3 WHEN 'stage2' THEN 2 WHEN 'stage1' THEN 1 ELSE 0 END) = latest.stage_rank
            WHERE c.run_id = ?
            ORDER BY COALESCE(c.composite_score, c.fitness, c.quick_score, -1e18) DESC
            LIMIT ?
            """,
            (run_id, run_id, top_n),
        )
        return [_row_to_dict(cur, row) for row in cur.fetchall()]

    def count_stage(self, run_id: str, stage: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id = ? AND stage = ?", (run_id, stage),
        )
        return int(cur.fetchone()[0])


def poll_live_leaderboard(db_path: str | Path, run_id: str, top_n: int = 20) -> list[dict]:
    """Live-leaderboard polling for a caller that does NOT own the
    ResultsDB instance the search itself is writing through -- e.g. a UI
    refresh timer polling a Search Lab / multi-instrument run still in
    progress in a background thread. Opens a short-lived, read-only
    connection (SQLite's WAL mode -- already enabled in ResultsDB.__init__
    -- lets a read-only reader see already-committed rows without
    blocking, or being blocked by, the writer) and closes it again before
    returning, so this is safe to call on a timer without accumulating
    open connections.

    Returns an empty list (never raises) if the database file doesn't
    exist yet -- e.g. polled in the brief window before a run's ResultsDB
    has created it -- since \"no leaderboard yet\" is a normal state for a
    just-started run, not an error worth surfacing to the UI.
    """
    path = Path(db_path)
    if not path.exists():
        return []
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return []
    try:
        shell = ResultsDB.__new__(ResultsDB)
        shell.path = str(path)
        shell._conn = conn
        return shell.live_leaderboard(run_id, top_n=top_n)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
