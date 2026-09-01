"""
Research library -- lets the optional local-Ollama AI assistant draw on
trading/quant research papers you drop into the top-level `research/`
folder, instead of only ever guessing from its own training data.

Design goals, in order:
  1. Zero setup: drop a PDF/text/markdown file in `research/`, nothing
     else to configure. Works with NO embedding model and NO Ollama
     connection at all via the plain keyword-overlap scoring below.
  2. Cheap and deterministic by default: finding which excerpts are
     relevant to a given query is a plain keyword-overlap score over
     paragraph-sized chunks -- the same "systematic first, AI only where
     it must be" philosophy as app.optimize.gene_fitness_analysis and
     app.validation.icir.
  3. Real semantic retrieval when available: pass an `OllamaSettings`
     into `find_relevant_excerpts` / call `embed_index` first, and this
     module blends in cosine-similarity search over local Ollama
     embeddings (see app.ai.vector_store) -- catching excerpts that are
     conceptually relevant but share few exact words with the query
     ("volatility regime" query surfacing an "ATR-normalized breakout"
     passage, for instance). This is the actual RAG layer: documents stay
     on disk as plain text, only their vectors are stored, and retrieval
     never sends anything to a cloud service -- the embedding model is
     just another local Ollama pull, same as the chat model.
  4. Cheap to keep current: papers are re-parsed (and re-embedded, if
     embeddings are enabled) only when a file is added, removed, or its
     modification time changes -- not on every single call.

Supported file types: .pdf (via the optional `pypdf` package), .txt, .md.
A PDF that fails to parse (scanned-image-only, corrupted, password-
protected) is skipped with a warning recorded in IndexStats, never
raised -- a bad paper should never break a GA run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ai.ollama_settings import OllamaSettings
from app.ai.vector_store import OllamaEmbedder, VectorStore, content_hash

VECTOR_COLLECTION = "research"

RESEARCH_DIR_NAME = "research"
CHUNK_TARGET_CHARS = 900   # roughly a paragraph or two -- small enough to be a focused excerpt,
                           # large enough to keep the idea's context intact
MIN_CHUNK_CHARS = 120      # drop stray short fragments (headers, page numbers, running footers)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "been", "this", "that", "these", "those", "with", "as", "by", "at", "from", "it",
    "its", "we", "our", "their", "they", "he", "she", "you", "your", "which", "can", "will",
    "not", "but", "if", "than", "then", "such", "into", "also", "may", "one", "two",
}


def _project_root() -> Path:
    # app/ai/research_library.py -> app/ai -> app -> project root
    return Path(__file__).resolve().parents[2]


def research_dir() -> Path:
    d = _project_root() / RESEARCH_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def _chunk_text(text: str) -> list[str]:
    """Splits on blank lines (paragraph breaks) first, then greedily packs
    consecutive paragraphs together up to roughly CHUNK_TARGET_CHARS so a
    chunk is neither a single sentence fragment nor an entire page. A
    single paragraph that alone exceeds the target (common in PDFs with no
    real paragraph breaks) is further split on sentence boundaries rather
    than left as one oversized chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_TARGET_CHARS * 1.5:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > CHUNK_TARGET_CHARS and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) > CHUNK_TARGET_CHARS and current:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


@dataclass
class ResearchChunk:
    source: str    # filename, relative to research/
    text: str


@dataclass
class IndexStats:
    files_indexed: int = 0
    chunks_indexed: int = 0
    files_skipped: list[str] = field(default_factory=list)  # files that failed to extract any text


_cache: dict[str, tuple[float, list[ResearchChunk]]] = {}  # path -> (mtime, chunks)


def list_research_files() -> list[Path]:
    d = research_dir()
    return sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")
    )


def build_index() -> tuple[list[ResearchChunk], IndexStats]:
    """Re-parses only files that are new or changed since the last call
    (tracked by mtime) -- cheap to call before every Ollama suggestion
    request rather than needing an explicit "reindex" step."""
    stats = IndexStats()
    all_chunks: list[ResearchChunk] = []
    seen_paths = set()

    for path in list_research_files():
        key = str(path)
        seen_paths.add(key)
        mtime = path.stat().st_mtime
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            chunks = cached[1]
        else:
            text = _extract_text(path)
            if not text.strip():
                stats.files_skipped.append(path.name)
                _cache[key] = (mtime, [])
                continue
            rel_name = path.relative_to(research_dir()).as_posix()
            chunks = [ResearchChunk(source=rel_name, text=c) for c in _chunk_text(text)]
            _cache[key] = (mtime, chunks)
        if chunks:
            stats.files_indexed += 1
        all_chunks.extend(chunks)

    # Drop cache entries for files that were removed from the folder.
    for stale_key in list(_cache.keys()):
        if stale_key not in seen_paths:
            del _cache[stale_key]

    stats.chunks_indexed = len(all_chunks)
    return all_chunks, stats


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _keyword_scores(query: str, chunks: list[ResearchChunk]) -> dict[int, float]:
    """Returns {chunk_index: score} for every chunk with nonzero
    keyword overlap. Split out from find_relevant_excerpts so the hybrid
    path below can call it directly against the SAME chunk list/indices
    without re-tokenizing twice."""
    query_terms = set(_tokenize(query))
    if not query_terms:
        return {}
    doc_freq: dict[str, int] = {}
    chunk_tokens = []
    for chunk in chunks:
        tokens = set(_tokenize(chunk.text))
        chunk_tokens.append(tokens)
        for t in tokens & query_terms:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    scores: dict[int, float] = {}
    for i, tokens in enumerate(chunk_tokens):
        overlap = tokens & query_terms
        if overlap:
            scores[i] = sum(1.0 / doc_freq[t] for t in overlap)
    return scores


def _chunk_item_id(chunk: ResearchChunk, index: int) -> str:
    # Stable per (source file, position) so re-embedding an unchanged
    # chunk is a no-op (VectorStore.has_current checks the text hash too,
    # so an edited-but-same-position chunk is still correctly re-embedded).
    return f"{chunk.source}::{index}"


@dataclass
class EmbedIndexStats:
    chunks_embedded: int = 0     # newly embedded this call
    chunks_already_current: int = 0
    total_chunks: int = 0
    error: str | None = None     # set only if embedding stopped early


def embed_index(settings: OllamaSettings, embed_model: str | None = None) -> EmbedIndexStats:
    """Embeds every research chunk not already stored with a current
    vector, using a local Ollama embedding model. Safe to call before
    every retrieval (like build_index()) -- chunks already embedded with
    unchanged text cost nothing. Stops and reports `.error` at the first
    embedding failure (Ollama unreachable, model not pulled) rather than
    hammering an unreachable host once per remaining chunk; whatever was
    already embedded in prior calls stays usable regardless."""
    chunks, _stats = build_index()
    store = VectorStore(VECTOR_COLLECTION)
    result = EmbedIndexStats(total_chunks=len(chunks))
    if not settings.is_usable:
        result.error = "AI Assist is not enabled -- semantic search stays off, keyword search still works."
        return result

    embedder = OllamaEmbedder(settings, model=embed_model)
    live_ids = set()
    for i, chunk in enumerate(chunks):
        item_id = _chunk_item_id(chunk, i)
        live_ids.add(item_id)
        if store.has_current(item_id, chunk.text):
            result.chunks_already_current += 1
            continue
        vector, err = embedder.embed_one(chunk.text)
        if err is not None:
            result.error = err
            return result  # stop early; leave the store as-is -- don't prune on a failed/partial pass
        store.upsert(item_id, chunk.text, vector, metadata={"source": chunk.source})
        result.chunks_embedded += 1

    store.prune_to(live_ids)  # drop vectors for chunks from files that were edited/removed since
    return result


def find_relevant_excerpts(
    query: str,
    max_excerpts: int = 3,
    max_chars_per_excerpt: int = 700,
    settings: OllamaSettings | None = None,
    embed_model: str | None = None,
) -> list[dict]:
    """Finds the excerpts most relevant to `query`.

    With no `settings` (or AI Assist disabled/unreachable), behaves
    exactly as before: plain TF-style keyword-overlap scoring, no
    embeddings/ML dependency -- scores each indexed chunk by how many of
    the query's (non-stopword) terms it contains, weighted slightly
    toward rarer terms.

    With a usable `settings`, blends in cosine-similarity search over
    this app's local Ollama embeddings (see app.ai.vector_store) -- run
    embed_index(settings) at least once first so there's something to
    search; a chunk that has no stored vector yet (never embedded, or
    embedding failed) simply falls back to its keyword-only score, so a
    partially-embedded library still degrades gracefully rather than
    silently hiding un-embedded papers.

    Returns [] with an empty research/ folder or a query with no usable
    terms and no semantic match either -- callers should treat that
    exactly like having no research library at all.
    """
    chunks, _stats = build_index()
    if not chunks:
        return []

    keyword_scores = _keyword_scores(query, chunks)
    semantic_scores: dict[int, float] = {}

    if settings is not None and settings.is_usable:
        store = VectorStore(VECTOR_COLLECTION)
        if len(store):
            embedder = OllamaEmbedder(settings, model=embed_model)
            query_vector, err = embedder.embed_one(query)
            if query_vector is not None:
                id_to_index = {_chunk_item_id(chunk, i): i for i, chunk in enumerate(chunks)}
                for item_id, similarity, _meta in store.search(query_vector, top_k=max(20, max_excerpts * 5)):
                    idx = id_to_index.get(item_id)
                    if idx is not None and similarity > 0:
                        semantic_scores[idx] = similarity

    if not keyword_scores and not semantic_scores:
        return []

    # Normalize each score set to [0, 1] independently before blending, so
    # neither scale (unbounded keyword-overlap sum vs. cosine similarity
    # in [-1, 1]) dominates the other just because of its raw magnitude.
    def _normalized(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        top = max(scores.values()) or 1.0
        return {k: v / top for k, v in scores.items()}

    kw_norm = _normalized(keyword_scores)
    sem_norm = _normalized(semantic_scores)
    # Weighted toward semantic when both are present -- it's the whole
    # point of embedding the library in the first place -- but keyword
    # score still contributes so an exact rare-term hit isn't buried
    # under a vaguely-similar-sounding passage.
    combined: dict[int, float] = {}
    for idx in set(kw_norm) | set(sem_norm):
        combined[idx] = 0.4 * kw_norm.get(idx, 0.0) + 0.6 * sem_norm.get(idx, 0.0)

    ranked = sorted(combined.items(), key=lambda t: t[1], reverse=True)[:max_excerpts]
    results = []
    for idx, score in ranked:
        chunk = chunks[idx]
        text = chunk.text if len(chunk.text) <= max_chars_per_excerpt else chunk.text[:max_chars_per_excerpt] + "..."
        results.append({"source": chunk.source, "text": text, "score": round(score, 3)})
    return results
