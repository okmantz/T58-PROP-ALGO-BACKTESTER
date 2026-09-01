"""
Knowledge graph -- "what tends to work, where, and under what conditions."

This is deliberately NOT a graph database. A defensible v1 of what Owen
described (T58 recognizing "liquidity sweep + high volatility + NY session
+ trend alignment -> historically strong") doesn't need one: it needs a
flat, append-only log of (feature vector -> outcome) for every candidate
the Evolution Lab has ever fully evaluated, plus a similarity query over
that log. That's what this module is. If it later needs to become an
actual graph (nodes for indicators/sessions/regimes with typed edges to
outcome nodes), this log is exactly the training data that would be
built from -- nothing here is wasted by staying simple now.

Storage: JSON Lines at `path` (default data/evolution/knowledge_graph.jsonl),
one record per line, append-only. No database dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.data.storage import get_app_base_dir


def _default_kg_path() -> Path:
    # Routed through get_app_base_dir() (same helper app/data/storage.py,
    # app/reports/run_history.py, app/strategy/library.py, and the theme/
    # settings files all use) instead of a bare relative path -- a plain
    # "data/evolution/..." path resolves against the current working
    # directory, which for a double-clicked packaged .exe is NOT
    # guaranteed to be the exe's own folder (or even writable), so the
    # knowledge graph could silently fail to persist, or write somewhere
    # the user never finds, across restarts.
    return get_app_base_dir() / "data" / "evolution" / "knowledge_graph.jsonl"


DEFAULT_KG_PATH = _default_kg_path()


def feature_vector_for_spec(spec: dict, meta: dict | None = None) -> dict:
    """Builds a structural feature vector for a candidate spec -- the thing
    similarity is computed over. Deliberately coarse/categorical (family,
    presence of a session/volatility/trend filter, indicator kinds used,
    direction bias) rather than exact parameter values, since two
    strategies with slightly different EMA periods but the same mechanism
    SHOULD look similar to this -- that's the whole point of "similarity"
    being about the *idea*, not the exact numbers.
    """
    meta = meta or {}
    family = meta.get("family") or spec.get("family") or "unknown"
    source_type = spec.get("source_type", "unknown")

    text = ""
    if source_type == "manual":
        text = json.dumps(spec.get("config", {}), default=str).lower()
    else:
        text = (spec.get("code_text") or "").lower()

    def _has(*needles: str) -> bool:
        return any(n in text for n in needles)

    features = {
        "family": family,
        "source_type": source_type,
        "uses_rsi": _has("rsi"),
        "uses_ema": _has("ema"),
        "uses_sma": _has("sma", "moving average", "mean"),
        "uses_atr": _has("atr", "volatility"),
        "uses_breakout": _has("breakout", "donchian", "sweep", "liquidity"),
        "uses_session_filter": _has("session", "ny_open", "london", "asia", "hour"),
        "uses_volume": _has("volume", "imbalance"),
        "mean_reversion": _has("mean_reversion", "mean reversion", "fade", "oversold", "overbought"),
        "trend_following": _has("trend", "momentum", "pullback"),
    }
    return features


def _jaccard(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    agree = sum(1 for k in keys if a.get(k) == b.get(k))
    return agree / len(keys)


@dataclass
class KnowledgeGraph:
    path: Path = field(default_factory=lambda: DEFAULT_KG_PATH)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, feature_vector: dict, outcome: dict) -> None:
        """outcome: free-form dict -- expected to include at least
        `final_score` (PROP FITNESS), `passed` (bool, did it reach the
        generation's KEEP TOP N), and whatever display fields are useful
        (strategy name, instrument, generation)."""
        row = {"features": feature_vector, "outcome": outcome}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _iter_records(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def query_similar(self, feature_vector: dict, top_k: int = 10) -> list[tuple[float, dict]]:
        scored = [(_jaccard(feature_vector, r["features"]), r) for r in self._iter_records()]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:top_k]

    def feature_success_rates(self) -> dict[str, dict]:
        """For every (feature_name, feature_value) pair seen so far, the
        fraction of records with that value that were marked `passed`.
        This is the "known edge" signal -- e.g. {'uses_session_filter':
        {True: 0.71, False: 0.22}} means a session filter correlates with
        surviving the funnel, across everything tested so far."""
        buckets: dict[tuple, list[bool]] = {}
        for r in self._iter_records():
            passed = bool(r.get("outcome", {}).get("passed"))
            for k, v in r.get("features", {}).items():
                buckets.setdefault((k, v), []).append(passed)
        out: dict[str, dict] = {}
        for (k, v), passes in buckets.items():
            out.setdefault(k, {})[v] = sum(passes) / len(passes)
        return out

    def describe(self, feature_vector: dict, top_k: int = 10) -> str:
        """The narrative format Owen asked for:
            Similarity: 82% to 417 previously tested strategies
            Known edge: NY liquidity continuation
            Novel component: volatility filter
            Historical success rate of similar strategies: 71%
        Best-effort from whatever's actually in the log -- says so plainly
        when there isn't enough history yet rather than inventing numbers.
        """
        all_records = list(self._iter_records())
        n_total = len(all_records)
        if n_total == 0:
            return "No prior strategies recorded yet -- this is the first one in the knowledge graph."

        similar = self.query_similar(feature_vector, top_k=top_k)
        n_similar = sum(1 for score, _ in similar if score >= 0.6)
        avg_similarity = (sum(s for s, _ in similar) / len(similar)) if similar else 0.0

        success_rates = self.feature_success_rates()
        known_edges = []
        novel = []
        for k, v in feature_vector.items():
            if v in (False, "unknown", None):
                continue
            rate = success_rates.get(k, {}).get(v)
            label = str(v) if k in ("family", "source_type") else k.replace("uses_", "").replace("_", " ")
            if rate is None:
                novel.append(label)
            elif rate >= 0.6:
                known_edges.append(f"{label} ({rate:.0%} historical success)")

        similar_passed = [r for _, r in similar if r.get("outcome", {}).get("passed")]
        historical_success_rate = (len(similar_passed) / len(similar)) if similar else None

        lines = [
            f"Similarity: {avg_similarity:.0%} to {n_similar} previously tested strategies (of {n_total} total logged).",
            f"Known edge(s): {', '.join(known_edges) if known_edges else 'none identified yet at >=60% historical success.'}",
            f"Novel component(s): {', '.join(novel) if novel else 'none -- every feature here has been tried before.'}",
        ]
        if historical_success_rate is not None:
            lines.append(f"Historical success rate of similar strategies: {historical_success_rate:.0%}")
        return "\n".join(lines)
