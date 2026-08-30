"""
Research library -- lets the optional local-Ollama AI assistant draw on
trading/quant research papers you drop into the top-level `research/`
folder, instead of only ever guessing from its own training data.

Design goals, in order:
  1. Zero setup: drop a PDF/text/markdown file in `research/`, nothing
     else to configure. No vector database, no embedding model, no
     internet call.
  2. Cheap and deterministic: finding which excerpts are relevant to a
     given strategy/gene context is a plain keyword-overlap score over
     paragraph-sized chunks -- the same "systematic first, AI only where
     it must be" philosophy as app.optimize.gene_fitness_analysis and
     app.validation.icir. This never calls Ollama itself; it only
     prepares better CONTEXT for the one Ollama call that already exists
     (see app.ai.ollama_client.suggest_parameter_adjustments).
  3. Cheap to keep current: papers are re-parsed only when a file is
     added, removed, or its modification time changes -- not on every
     single call.

Supported file types: .pdf (via the optional `pypdf` package), .txt, .md.
A PDF that fails to parse (scanned-image-only, corrupted, password-
protected) is skipped with a warning recorded in IndexStats, never
raised -- a bad paper should never break a GA run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

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


def find_relevant_excerpts(query: str, max_excerpts: int = 3, max_chars_per_excerpt: int = 700) -> list[dict]:
    """Plain TF-style keyword-overlap scoring, no embeddings/ML dependency
    -- scores each indexed chunk by how many of the query's (non-stopword)
    terms it contains, weighted slightly toward rarer terms so a chunk
    matching on "moving average" doesn't outscore one matching on a
    specific, less-common term the query cares about. Returns [] with an
    empty research/ folder or a query with no usable terms -- callers
    should treat that exactly like having no research library at all.
    """
    chunks, _stats = build_index()
    if not chunks:
        return []
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    # Document frequency per term, for the rarity weighting mentioned above.
    doc_freq: dict[str, int] = {}
    chunk_tokens = []
    for chunk in chunks:
        tokens = set(_tokenize(chunk.text))
        chunk_tokens.append(tokens)
        for t in tokens & query_terms:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    scored = []
    for chunk, tokens in zip(chunks, chunk_tokens):
        overlap = tokens & query_terms
        if not overlap:
            continue
        score = sum(1.0 / doc_freq[t] for t in overlap)
        scored.append((score, chunk))

    scored.sort(key=lambda sc: sc[0], reverse=True)
    results = []
    for score, chunk in scored[:max_excerpts]:
        text = chunk.text if len(chunk.text) <= max_chars_per_excerpt else chunk.text[:max_chars_per_excerpt] + "..."
        results.append({"source": chunk.source, "text": text, "score": round(score, 3)})
    return results
