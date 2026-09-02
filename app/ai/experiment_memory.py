"""T58 Research Memory -- a durable, searchable record of every strategy
test this app has ever run, so the AI Research Agent (and you) can ask
"has something like this been tried before, and what happened?" instead
of every experiment starting from zero.

This is the "Level 3 -- self-improving research loop" piece: every run of
Full Pipeline / Quick Optimize / a Batch Test / the Research Agent itself
gets one row here (best-effort -- recording a completed experiment must
never be able to fail the run that produced it). Over time this becomes
exactly the "10,000 strategies tested / 2,000 rejected / 500 robust..."
memory described in the research-engine plan: a plain SQL table you can
already query directly, PLUS semantic search over a short free-text
summary of each experiment (via app.ai.vector_store, using local Ollama
embeddings) so "strategies similar to this one" doesn't depend on exact
name matches.

Uses the standard library's sqlite3 only for storage (same pattern as
app.search.results_db) -- the only new dependency is the *optional*
embedding step, which degrades to a plain keyword/SQL search over the
`summary` column when Ollama isn't enabled or reachable. Nothing here
ever blocks or fails a caller: every public function catches its own
exceptions and returns an empty/default result rather than propagating.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.ai.ollama_settings import OllamaSettings, load_settings
from app.ai.vector_store import OllamaEmbedder, VectorStore

VECTOR_COLLECTION = "experiments"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    origin TEXT NOT NULL,
    strategy_name TEXT,
    source_type TEXT,
    instrument TEXT,
    verdict TEXT,
    trades INTEGER,
    net_profit REAL,
    win_rate REAL,
    profit_factor REAL,
    max_drawdown_pct REAL,
    eval_pass_probability REAL,
    first_payout_probability REAL,
    risk_of_ruin_pct REAL,
    summary TEXT,
    lesson TEXT,
    config_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_strategy_name ON experiments(strategy_name);
CREATE INDEX IF NOT EXISTS idx_experiments_verdict ON experiments(verdict);
"""


def _db_path() -> Path:
    from app.data.storage import get_app_base_dir

    d = get_app_base_dir() / "data" / "ai_memory"
    d.mkdir(parents=True, exist_ok=True)
    return d / "experiments.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@dataclass
class Experiment:
    id: str
    created_at: float
    origin: str
    strategy_name: str
    source_type: str
    instrument: str
    verdict: str
    trades: int
    net_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    eval_pass_probability: float
    first_payout_probability: float
    risk_of_ruin_pct: float
    summary: str
    lesson: str
    config: dict = field(default_factory=dict)
    similarity: float | None = None  # only set by search_similar_experiments

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("config", None)
        d["config"] = self.config
        return d


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    try:
        config = json.loads(row["config_json"] or "{}")
    except Exception:
        config = {}
    return Experiment(
        id=row["id"], created_at=row["created_at"], origin=row["origin"] or "",
        strategy_name=row["strategy_name"] or "", source_type=row["source_type"] or "",
        instrument=row["instrument"] or "", verdict=row["verdict"] or "UNKNOWN",
        trades=row["trades"] or 0, net_profit=row["net_profit"] or 0.0,
        win_rate=row["win_rate"] or 0.0, profit_factor=row["profit_factor"] or 0.0,
        max_drawdown_pct=row["max_drawdown_pct"] or 0.0,
        eval_pass_probability=row["eval_pass_probability"] or 0.0,
        first_payout_probability=row["first_payout_probability"] or 0.0,
        risk_of_ruin_pct=row["risk_of_ruin_pct"] or 0.0,
        summary=row["summary"] or "", lesson=row["lesson"] or "", config=config,
    )


def _build_summary(
    strategy_name: str, source_type: str, instrument: str, verdict: str,
    trades: int, net_profit: float, win_rate: float, profit_factor: float,
    max_drawdown_pct: float, eval_pass_probability: float, lesson: str,
) -> str:
    """One short natural-language paragraph per experiment -- this is
    the text that actually gets embedded/keyword-matched, so it's written
    to read like a research-notebook entry rather than a raw stats dump."""
    parts = [
        f"Strategy '{strategy_name}' ({source_type or 'unknown'}) on {instrument or 'unknown instrument'}: "
        f"{trades} trades, net profit {net_profit:,.2f}, win rate {win_rate:.1f}%, "
        f"profit factor {profit_factor:.2f}, max drawdown {max_drawdown_pct:.1f}%, "
        f"eval pass probability {eval_pass_probability:.1f}%. Verdict: {verdict}."
    ]
    if lesson:
        parts.append(f"Lesson: {lesson}")
    return " ".join(parts)


def record_experiment(
    origin: str,
    strategy_name: str,
    source_type: str = "",
    instrument: str = "",
    verdict: str = "UNKNOWN",
    trades: int = 0,
    net_profit: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    max_drawdown_pct: float = 0.0,
    eval_pass_probability: float = 0.0,
    first_payout_probability: float = 0.0,
    risk_of_ruin_pct: float = 0.0,
    lesson: str = "",
    config: dict | None = None,
    settings: OllamaSettings | None = None,
) -> str | None:
    """Records one completed experiment. Returns the new experiment's id,
    or None if recording failed for any reason (never raises -- this is
    called from the tail end of Full Pipeline / Quick Optimize / Batch
    Test runs and must never be able to turn a successful backtest run
    into a crash).

    `origin`: which feature produced this, e.g. "full_pipeline",
    "quick_optimize", "batch_test", "research_agent" -- free text, used
    only for filtering/display.

    `settings`: optional OllamaSettings. If usable, the summary is also
    embedded into the semantic experiment-memory index (best-effort --
    a failure here still leaves the SQL row recorded; only the semantic
    index entry is skipped). Defaults to whatever's saved via
    app.ai.ollama_settings if not passed explicitly.
    """
    try:
        exp_id = uuid.uuid4().hex
        summary = _build_summary(
            strategy_name, source_type, instrument, verdict, trades, net_profit,
            win_rate, profit_factor, max_drawdown_pct, eval_pass_probability, lesson,
        )
        conn = _connect()
        with conn:
            conn.execute(
                """INSERT INTO experiments (
                    id, created_at, origin, strategy_name, source_type, instrument, verdict,
                    trades, net_profit, win_rate, profit_factor, max_drawdown_pct,
                    eval_pass_probability, first_payout_probability, risk_of_ruin_pct,
                    summary, lesson, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp_id, time.time(), origin, strategy_name, source_type, instrument, verdict,
                    trades, net_profit, win_rate, profit_factor, max_drawdown_pct,
                    eval_pass_probability, first_payout_probability, risk_of_ruin_pct,
                    summary, lesson, json.dumps(config or {}),
                ),
            )
        conn.close()
    except Exception:
        return None

    try:
        eff_settings = settings if settings is not None else load_settings()
        if eff_settings.is_usable:
            embedder = OllamaEmbedder(eff_settings)
            vector, err = embedder.embed_one(summary)
            if vector is not None:
                store = VectorStore(VECTOR_COLLECTION)
                store.upsert(exp_id, summary, vector, metadata={"strategy_name": strategy_name})
    except Exception:
        pass  # semantic indexing is a bonus -- the SQL row above is already safely recorded

    return exp_id


def get_recent_experiments(limit: int = 20, origin: str | None = None) -> list[Experiment]:
    try:
        conn = _connect()
        if origin:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE origin = ? ORDER BY created_at DESC LIMIT ?", (origin, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        conn.close()
        return [_row_to_experiment(r) for r in rows]
    except Exception:
        return []


def get_experiment_by_id(experiment_id: str) -> Experiment | None:
    try:
        conn = _connect()
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        conn.close()
        return _row_to_experiment(row) if row is not None else None
    except Exception:
        return None


_LEADERBOARD_SORT_COLUMNS = {
    "eval_pass_probability", "first_payout_probability", "net_profit",
    "profit_factor", "win_rate", "created_at",
}


def get_leaderboard(
    limit: int = 20, origin: str | None = None, sort_by: str = "eval_pass_probability",
    verdict: str | None = None,
) -> list[Experiment]:
    """The "every experiment permanently recorded, ranked" view: every
    recorded experiment (optionally filtered to one `origin`, e.g.
    'research_loop', or one `verdict`, e.g. 'KEEP'), ranked by `sort_by`
    (must be one of _LEADERBOARD_SORT_COLUMNS -- an unrecognized value
    falls back to eval_pass_probability rather than risking a SQL
    injection via a caller-supplied column name). Highest first.
    """
    col = sort_by if sort_by in _LEADERBOARD_SORT_COLUMNS else "eval_pass_probability"
    try:
        conn = _connect()
        clauses, params = [], []
        if origin:
            clauses.append("origin = ?")
            params.append(origin)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM experiments {where} ORDER BY {col} DESC LIMIT ?", (*params, limit),
        ).fetchall()
        conn.close()
        return [_row_to_experiment(r) for r in rows]
    except Exception:
        return []


def find_similar_discarded_dna(active_tags: list[str], origin: str | None = None, min_jaccard: float = 0.99) -> bool:
    """Thin convenience wrapper over is_dna_tagset_previously_discarded
    for callers that just want a yes/no answer at (near-)exact match --
    prefer calling is_dna_tagset_previously_discarded directly when a
    looser similarity threshold or the actual similarity score is
    useful (e.g. app.ai.research_loop's cross-session dedup check)."""
    is_repeat, _similarity = is_dna_tagset_previously_discarded(active_tags, origin=origin, min_jaccard=min_jaccard)
    return is_repeat


def render_leaderboard_card(exp: Experiment, parent: Experiment | None = None) -> str:
    """Plain-text experiment card in the 'Experiment #N / Strategy /
    Changes / Results / Compared with parent / Verdict' shape -- the
    format a research log actually reads in. `parent` (if given) must be
    the Experiment referenced by exp.config.get("parent_experiment");
    passing the wrong one produces a misleading diff, so callers should
    resolve it via get_experiment_by_id(exp.config.get("parent_experiment"))
    themselves rather than guessing.

    Reads Strategy DNA tags from config["dna"] as a plain list of tag
    strings (the same shape app.ai.research_loop records and
    get_dna_tagsets_by_verdict/is_dna_tagset_previously_discarded read),
    not the full app.strategy.dna.StrategyDNA.to_dict() shape.
    """
    lines = [f"Experiment  {exp.id[:8]}", "", "Strategy:", f"  {exp.strategy_name}"]

    exp_tags = set(exp.config.get("dna") or []) if isinstance(exp.config, dict) else set()
    parent_tags = set(parent.config.get("dna") or []) if (parent and isinstance(parent.config, dict)) else set()
    if exp_tags and parent_tags:
        added = sorted(exp_tags - parent_tags)
        removed = sorted(parent_tags - exp_tags)
        if added or removed:
            lines += ["", "Changes:"]
            lines += [f"  + {t.split('.', 1)[-1].replace('_', ' ')}" for t in added]
            lines += [f"  - {t.split('.', 1)[-1].replace('_', ' ')}" for t in removed]

    lines += [
        "", "Results:",
        f"  Pass: {exp.eval_pass_probability:.1f}%",
        f"  Payout: {exp.first_payout_probability:.1f}%",
        f"  DD: {exp.max_drawdown_pct:.1f}%",
    ]

    if parent is not None:
        lines += [
            "", "Compared with parent:",
            f"  Pass: {exp.eval_pass_probability - parent.eval_pass_probability:+.1f}%",
            f"  Payout: {exp.first_payout_probability - parent.first_payout_probability:+.1f}%",
            f"  DD: {exp.max_drawdown_pct - parent.max_drawdown_pct:+.1f}%",
        ]

    lines += ["", "Verdict:", f"  {exp.verdict}"]
    if exp.lesson:
        lines += ["", "Lesson:", f"  {exp.lesson}"]
    return "\n".join(lines)


def get_summary_counts() -> dict:
    """Returns the "T58 Research Memory" dashboard numbers: total
    experiments tested, and a breakdown by verdict. Never raises --
    returns all-zero counts on any error (e.g. the DB not existing yet,
    which is the normal state before the first experiment is recorded)."""
    try:
        conn = _connect()
        total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        by_verdict_rows = conn.execute(
            "SELECT verdict, COUNT(*) as n FROM experiments GROUP BY verdict"
        ).fetchall()
        by_strategy_rows = conn.execute(
            "SELECT strategy_name, COUNT(*) as n FROM experiments GROUP BY strategy_name "
            "ORDER BY n DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return {
            "total": total,
            "by_verdict": {r["verdict"]: r["n"] for r in by_verdict_rows},
            "top_strategies": {r["strategy_name"]: r["n"] for r in by_strategy_rows},
        }
    except Exception:
        return {"total": 0, "by_verdict": {}, "top_strategies": {}}


def _keyword_search(query: str, max_results: int) -> list[Experiment]:
    """Plain SQL LIKE fallback over the summary/lesson/strategy_name
    columns -- used whenever semantic search isn't available, so
    "similar past experiments" always returns SOMETHING rather than
    silently requiring Ollama to be enabled."""
    terms = [t for t in query.lower().split() if len(t) > 2][:8]
    if not terms:
        return get_recent_experiments(limit=max_results)
    try:
        conn = _connect()
        clauses = " OR ".join(["LOWER(summary) LIKE ? OR LOWER(lesson) LIKE ? OR LOWER(strategy_name) LIKE ?"] * len(terms))
        params: list = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like, like])
        rows = conn.execute(
            f"SELECT * FROM experiments WHERE {clauses} ORDER BY created_at DESC LIMIT ?",
            (*params, max_results),
        ).fetchall()
        conn.close()
        return [_row_to_experiment(r) for r in rows]
    except Exception:
        return []


def search_similar_experiments(
    query: str, settings: OllamaSettings | None = None, max_results: int = 5,
) -> list[Experiment]:
    """Finds past experiments most similar to `query` -- pass either a
    plain description ("RSI mean reversion on gold, tight stops") or an
    experiment's own summary text to find its nearest neighbors.

    Uses semantic (embedding) search when `settings` is usable and the
    experiment vector index isn't empty; falls back to a plain keyword
    LIKE search over stored summaries/lessons otherwise. Every returned
    Experiment has `.similarity` set (cosine similarity in semantic mode,
    or 1.0 for every keyword-mode hit since there's no comparable score).
    """
    eff_settings = settings if settings is not None else load_settings()
    if eff_settings.is_usable:
        store = VectorStore(VECTOR_COLLECTION)
        if len(store):
            embedder = OllamaEmbedder(eff_settings)
            vector, err = embedder.embed_one(query)
            if vector is not None:
                hits = store.search(vector, top_k=max_results)
                if hits:
                    try:
                        conn = _connect()
                        results = []
                        for item_id, similarity, _meta in hits:
                            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (item_id,)).fetchone()
                            if row is not None:
                                exp = _row_to_experiment(row)
                                exp.similarity = similarity
                                results.append(exp)
                        conn.close()
                        if results:
                            return results
                    except Exception:
                        pass  # fall through to keyword search below

    results = _keyword_search(query, max_results)
    for r in results:
        r.similarity = 1.0
    return results


def get_dna_tagsets_by_verdict(
    origin: str | None = None, verdict: str = "DISCARD", limit: int = 200,
) -> list[list[str]]:
    """Every recorded experiment's Strategy DNA active-tag set (see
    app.strategy.dna) for a given verdict, newest first -- the piece the
    Ollama Research Loop uses to avoid re-testing a structurally
    identical strategy it (or a prior run) already tried and discarded.
    Only returns experiments whose config_json actually contains a "dna"
    list (older experiments recorded before Strategy DNA existed simply
    don't contribute -- never raises or requires a migration).
    """
    try:
        conn = _connect()
        if origin:
            rows = conn.execute(
                "SELECT config_json FROM experiments WHERE verdict = ? AND origin = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (verdict, origin, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT config_json FROM experiments WHERE verdict = ? ORDER BY created_at DESC LIMIT ?",
                (verdict, limit),
            ).fetchall()
        conn.close()
    except Exception:
        return []

    tagsets = []
    for row in rows:
        try:
            config = json.loads(row["config_json"] or "{}")
            dna_tags = config.get("dna")
            if dna_tags:
                tagsets.append(list(dna_tags))
        except Exception:
            continue
    return tagsets


def is_dna_tagset_previously_discarded(
    tags: list[str], origin: str | None = None, min_jaccard: float = 0.85, limit: int = 200,
) -> tuple[bool, float]:
    """Whether `tags` (a strategy's active DNA tags) is close enough to a
    PAST DISCARD-verdict experiment's tag set to count as a repeat --
    measured by Jaccard similarity (intersection / union) rather than
    exact equality, since two strategies with the same core idea rarely
    produce byte-identical tag sets. Returns (is_repeat, best_similarity)
    -- best_similarity is 0.0 if there's nothing to compare against yet.
    """
    if not tags:
        return False, 0.0
    tag_set = set(tags)
    best = 0.0
    for past_tags in get_dna_tagsets_by_verdict(origin=origin, verdict="DISCARD", limit=limit):
        past_set = set(past_tags)
        union = tag_set | past_set
        if not union:
            continue
        jaccard = len(tag_set & past_set) / len(union)
        best = max(best, jaccard)
    return best >= min_jaccard, best
