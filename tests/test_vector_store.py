"""Tests for app.ai.vector_store -- covers the pure VectorStore persistence
layer without any real Ollama connection, plus OllamaEmbedder's
fail-safe error handling via a stubbed `requests` module."""
from __future__ import annotations

import types

import numpy as np
import pytest

from app.ai import vector_store
from app.ai.ollama_settings import OllamaSettings


@pytest.fixture(autouse=True)
def _isolated_store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "_store_dir", lambda: tmp_path)
    yield


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert vector_store.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert vector_store.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_safe():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert vector_store.cosine_similarity(a, b) == 0.0


def test_empty_store_search_returns_empty_list():
    store = vector_store.VectorStore("test_collection")
    assert store.search([1.0, 0.0]) == []
    assert len(store) == 0


def test_upsert_and_search_ranks_by_similarity():
    store = vector_store.VectorStore("test_collection")
    store.upsert("a", "text a", [1.0, 0.0])
    store.upsert("b", "text b", [0.0, 1.0])
    store.upsert("c", "text c", [0.9, 0.1])

    results = store.search([1.0, 0.0], top_k=2)
    ids = [r[0] for r in results]
    assert ids[0] == "a"
    assert "b" not in ids  # least similar, excluded by top_k=2


def test_store_persists_across_instances(tmp_path):
    store1 = vector_store.VectorStore("persisted")
    store1.upsert("x", "hello world", [1.0, 2.0], metadata={"source": "doc1"})

    store2 = vector_store.VectorStore("persisted")
    assert len(store2) == 1
    results = store2.search([1.0, 2.0], top_k=1)
    assert results[0][0] == "x"
    assert results[0][2] == {"source": "doc1"}


def test_has_current_true_only_for_unchanged_text():
    store = vector_store.VectorStore("test_collection")
    store.upsert("a", "original text", [1.0])
    assert store.has_current("a", "original text") is True
    assert store.has_current("a", "changed text") is False
    assert store.has_current("nonexistent", "original text") is False


def test_delete_removes_item():
    store = vector_store.VectorStore("test_collection")
    store.upsert("a", "text", [1.0])
    store.delete("a")
    assert len(store) == 0


def test_prune_to_drops_items_not_in_keep_set():
    store = vector_store.VectorStore("test_collection")
    store.upsert("a", "text a", [1.0])
    store.upsert("b", "text b", [2.0])
    store.prune_to({"a"})
    assert len(store) == 1
    assert store.has_current("a", "text a")


def test_corrupted_store_file_treated_as_empty(tmp_path):
    path = vector_store._collection_path("broken")
    path.write_text("not valid json{{{", encoding="utf-8")
    store = vector_store.VectorStore("broken")
    assert len(store) == 0


# ---------------------------------------------------------------------------
# OllamaEmbedder -- fail-safe network behavior, mocked requests
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("bad status")

    def json(self):
        return self._payload


def test_embed_one_returns_none_and_error_when_host_unset():
    embedder = vector_store.OllamaEmbedder(OllamaSettings(enabled=True, host=""))
    vector, err = embedder.embed_one("hello")
    assert vector is None
    assert err is not None


def test_embed_one_success(monkeypatch):
    fake_requests = types.SimpleNamespace(
        post=lambda *a, **k: _FakeResponse({"embedding": [0.1, 0.2, 0.3]}),
        exceptions=types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    embedder = vector_store.OllamaEmbedder(OllamaSettings(enabled=True, host="http://localhost:11434"))
    vector, err = embedder.embed_one("hello")
    assert err is None
    assert vector == [0.1, 0.2, 0.3]


def test_embed_one_missing_embedding_key_is_an_error(monkeypatch):
    fake_requests = types.SimpleNamespace(
        post=lambda *a, **k: _FakeResponse({}),
        exceptions=types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    embedder = vector_store.OllamaEmbedder(OllamaSettings(enabled=True, host="http://localhost:11434"))
    vector, err = embedder.embed_one("hello")
    assert vector is None
    assert "pulled" in err


def test_embed_one_connection_error_is_fail_safe(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("nope")

    fake_requests = types.SimpleNamespace(
        post=_raise,
        exceptions=types.SimpleNamespace(ConnectionError=ConnectionError, Timeout=TimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    embedder = vector_store.OllamaEmbedder(OllamaSettings(enabled=True, host="http://localhost:11434"))
    vector, err = embedder.embed_one("hello")
    assert vector is None
    assert "reach Ollama" in err
