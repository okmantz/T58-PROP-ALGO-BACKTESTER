"""Local vector store for the T58 AI Research Engine.

This is the "real RAG" upgrade over app.ai.research_library's original
plain-keyword retrieval: text is embedded into vectors using a LOCAL
Ollama embedding model (e.g. `ollama pull nomic-embed-text`), stored on
disk, and retrieved by cosine similarity. No cloud API, no new heavy
dependency (no sentence-transformers/faiss/chromadb) -- embeddings are
just another local Ollama call, using the exact same OllamaSettings
(host/model/api-key) already used everywhere else in app.ai, and numpy
(already a hard dependency of this app) for the similarity math.

Design goals, same order of priorities as research_library.py:
  1. Off/degraded by default: every function here fails SAFE. If Ollama
     isn't reachable, the configured embedding model isn't pulled, or the
     store is simply empty, callers get an empty result / a clear
     `.error`, never an exception. Everything upstream (research_library,
     experiment_memory, research_agent) is written to fall back to
     keyword-only matching when this returns nothing -- semantic search
     is a strict improvement when available, never a hard requirement.
  2. Cheap to keep current: each stored item is keyed by a content hash,
     so re-indexing unchanged text costs nothing (no re-embedding, no
     Ollama call).
  3. Plain files, no server: persisted as one JSON file per named
     "collection" (e.g. "research", "experiments") under this app's own
     data directory -- inspectable, deletable, and portable exactly like
     every other persistent store in this codebase (see
     app.data.storage / app.strategy.library / app.search.results_db).

This module intentionally does NOT decide what to embed or how to chunk
text -- that stays the job of the caller (research_library for papers,
experiment_memory for past test results). It only stores vectors and
answers "what's most similar to this query vector".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ai.ollama_settings import OllamaSettings
from app.data.storage import get_app_base_dir

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT_SECONDS = 30


def _store_dir() -> Path:
    d = get_app_base_dir() / "data" / "ai_memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collection_path(collection: str) -> Path:
    safe = "".join(c for c in collection if c.isalnum() or c in ("-", "_")) or "default"
    return _store_dir() / f"vectors_{safe}.json"


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class EmbeddingResult:
    vectors: dict[str, list[float]] = field(default_factory=dict)  # id -> vector
    error: str | None = None


class OllamaEmbedder:
    """Thin IO wrapper around Ollama's `/api/embeddings` endpoint. Mirrors
    app.ai.ollama_client.OllamaClient's fail-safe posture: any network
    error, timeout, or malformed response results in `.error` being set
    and an empty vector list -- never a raised exception."""

    def __init__(self, settings: OllamaSettings, model: str | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.settings = settings
        self.model = model or DEFAULT_EMBED_MODEL
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def embed_one(self, text: str) -> tuple[list[float] | None, str | None]:
        import requests

        host = (self.settings.host or "").rstrip("/")
        if not host:
            return None, "No Ollama host configured."
        try:
            resp = requests.post(
                f"{host}/api/embeddings",
                headers=self._headers(),
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return None, f"Couldn't reach Ollama at {host} (is it running?)."
        except requests.exceptions.Timeout:
            return None, f"Ollama at {host} didn't respond in time."
        except Exception as exc:
            return None, f"Ollama embedding request failed: {exc}"

        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            return None, (
                f"Ollama responded, but returned no embedding -- is '{self.model}' pulled? "
                f"(Try: ollama pull {self.model})"
            )
        return [float(v) for v in vector], None

    def embed_many(self, texts: list[str]) -> EmbeddingResult:
        """Embeds each text one at a time (Ollama's embeddings endpoint has
        no true batch mode across versions). Stops at the first failure
        and reports it, rather than silently returning a partial index --
        callers should treat a partial EmbeddingResult.error as "semantic
        indexing is currently unavailable" and keep whatever was
        previously indexed."""
        vectors: dict[str, list[float]] = {}
        for i, text in enumerate(texts):
            vec, err = self.embed_one(text)
            if err is not None:
                return EmbeddingResult(vectors=vectors, error=err)
            vectors[str(i)] = vec
        return EmbeddingResult(vectors=vectors)

    def test_connection(self) -> tuple[bool, str]:
        vec, err = self.embed_one("connection test")
        if err:
            return False, err
        return True, f"Connected -- embedding model '{self.model}' returned a {len(vec)}-dimensional vector."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class VectorItem:
    item_id: str
    text_hash: str
    vector: list[float]
    metadata: dict


class VectorStore:
    """A named collection of (id -> vector + metadata), persisted as one
    JSON file. Every method is safe to call even when the collection file
    doesn't exist yet (starts empty) or is corrupted (treated as empty
    rather than raising -- a damaged cache file should never break the
    app, it should just cause a one-time silent re-embed)."""

    def __init__(self, collection: str):
        self.collection = collection
        self._path = _collection_path(collection)
        self._items: dict[str, VectorItem] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in raw.get("items", []):
                item = VectorItem(
                    item_id=entry["id"], text_hash=entry["hash"],
                    vector=entry["vector"], metadata=entry.get("metadata", {}),
                )
                self._items[item.item_id] = item
        except Exception:
            self._items = {}

    def _save(self) -> None:
        payload = {
            "items": [
                {"id": it.item_id, "hash": it.text_hash, "vector": it.vector, "metadata": it.metadata}
                for it in self._items.values()
            ]
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass  # best-effort persistence -- an in-memory-only session degrades gracefully next run

    def __len__(self) -> int:
        return len(self._items)

    def has_current(self, item_id: str, text: str) -> bool:
        """True if this id is already stored with a vector for this EXACT
        text (so callers can skip re-embedding unchanged content)."""
        existing = self._items.get(item_id)
        return existing is not None and existing.text_hash == content_hash(text)

    def upsert(self, item_id: str, text: str, vector: list[float], metadata: dict | None = None) -> None:
        self._items[item_id] = VectorItem(
            item_id=item_id, text_hash=content_hash(text), vector=vector, metadata=metadata or {},
        )
        self._save()

    def delete(self, item_id: str) -> None:
        if item_id in self._items:
            del self._items[item_id]
            self._save()

    def prune_to(self, keep_ids: set[str]) -> None:
        """Drops any stored item whose id is not in keep_ids -- used to
        clear out vectors for research files that have been removed."""
        stale = [k for k in self._items if k not in keep_ids]
        if not stale:
            return
        for k in stale:
            del self._items[k]
        self._save()

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float, dict]]:
        """Returns up to top_k (item_id, cosine_similarity, metadata)
        tuples, highest similarity first. Returns [] on an empty store --
        never raises."""
        if not self._items:
            return []
        q = np.array(query_vector, dtype=float)
        scored = []
        for it in self._items.values():
            try:
                score = cosine_similarity(q, np.array(it.vector, dtype=float))
            except Exception:
                continue
            scored.append((it.item_id, score, it.metadata))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]
