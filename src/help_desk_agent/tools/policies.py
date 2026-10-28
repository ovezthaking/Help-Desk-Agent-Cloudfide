"""Policy search tool (Markdown chunking + BM25 & Gemini Embeddings RAG)."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from operator import mul
from pathlib import Path
from typing import Any

from google import genai

from help_desk_agent.config import (
    EMBEDDINGS_CACHE_PATH,
    POLICIES_DIR,
    STAFF_EMAIL,
    STAFF_PHONE,
    get_gemini_api_key,
    get_gemini_embedding_model,
)
from help_desk_agent.models import PolicyChunk

EmbedFn = Callable[[list[str]], list[list[float]]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SECTION_SPLIT_RE = re.compile(r"(?m)^(?=## )")
_EBIKE_VARIANTS = frozenset({"e-bike", "ebikes", "e-bikes"})

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "get",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "will",
        "with",
        "you",
        "your",
    }
)

SYNONYM_MAP: dict[str, list[str]] = {
    "saturday": ["open", "monday", "sunday", "april", "october", "november"],
    "sunday": ["open", "closed", "november", "march"],
    "weekend": ["open", "saturday", "sunday"],
    "hours": ["open", "closed", "08", "20", "09", "17"],
    "schedule": ["open", "closed"],
    "cost": ["price", "eur", "rental", "hour", "day"],
    "rate": ["price", "eur", "rental", "hour", "day"],
    "much": ["price", "eur", "refund"],
    "ebike": ["electric", "bike"],
    "electric": ["ebike", "bike"],
    "child": ["kids", "seat", "age", "children", "under"],
    "teenager": ["age", "limit", "years", "old", "under"],
    "year": ["age", "limit", "years", "old", "16", "13", "18"],
    "old": ["age", "limit", "years", "16", "13", "18"],
    "pet": ["dog", "basket"],
    "puppy": ["dog", "basket"],
    "animal": ["dog", "basket"],
    "delayed": ["late", "return", "delay", "fee"],
    "overdue": ["late", "return", "delay", "fee"],
    "rain": ["weather", "storm", "warning"],
    "weather": ["storm", "warning", "refund"],
    "crash": ["breaks", "accident", "repair", "damage"],
    "broken": ["breaks", "accident", "repair"],
    "theft": ["stolen", "lost", "police"],
    "buy": ["sell", "rent"],
    "purchase": ["sell", "rent"],
    "location": ["where", "park", "entrance", "tram"],
    "address": ["where", "park", "entrance", "tram"],
}


@lru_cache(maxsize=1024)
def _normalize_token(token: str) -> str:
    """Apply lightweight normalization/stemming to a lowercase word token."""
    if token in _EBIKE_VARIANTS:
        return "ebike"
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, expand_synonyms: bool = False) -> list[str]:
    """Tokenize text into normalized keywords."""
    cleaned = text.lower().replace("e-bike", "ebike")
    raw_tokens = _TOKEN_RE.findall(cleaned)
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        norm = _normalize_token(tok)
        if norm and norm not in STOPWORDS:
            tokens.append(norm)
            if expand_synonyms and (synonyms := SYNONYM_MAP.get(norm)):
                tokens.extend(synonyms)
    return tokens


def _slugify(title: str) -> str:
    """Convert a section or document title into a URL-friendly slug."""
    return _SLUG_RE.sub("-", title.lower()).strip("-")


def _parse_markdown_chunks(md_file: Path, raw_text: str) -> list[PolicyChunk]:
    """Split a single markdown document into section PolicyChunks."""
    parts = _SECTION_SPLIT_RE.split(raw_text)
    header_block, *section_blocks = (
        ["", *parts] if raw_text.startswith("## ") else parts
    )

    doc_title = md_file.stem.capitalize()
    preamble_lines: list[str] = []
    for line in header_block.splitlines():
        stripped = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            doc_title = line[2:].strip()
        elif stripped:
            preamble_lines.append(stripped)

    if not section_blocks:
        return [
            PolicyChunk(
                chunk_id=f"{md_file.name}#{_slugify(doc_title)}",
                source_file=md_file.name,
                document_title=doc_title,
                section_title=doc_title,
                content=raw_text,
            )
        ]

    preamble = "\n".join(preamble_lines)
    prefix = (
        f"# {doc_title}\n{preamble}\n\n" if preamble else f"# {doc_title}\n\n"
    )

    return [
        PolicyChunk(
            chunk_id=f"{md_file.name}#{_slugify(sec_title)}",
            source_file=md_file.name,
            document_title=doc_title,
            section_title=sec_title,
            content=f"{prefix}{body}",
        )
        for block in section_blocks
        if (body := block.strip())
        and (sec_title := body.partition("\n")[0][3:].strip())
    ]


def load_policy_chunks(policies_dir: Path | None = None) -> list[PolicyChunk]:
    """Parse markdown files in data/policies/ into logical section chunks."""
    directory = policies_dir or POLICIES_DIR
    if not directory.exists():
        return []

    chunks: list[PolicyChunk] = []
    for md_file in sorted(directory.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8").strip()
        if raw_text:
            chunks.extend(_parse_markdown_chunks(md_file, raw_text))
    return chunks


@lru_cache(maxsize=256)
def _chunk_token_stats(
    document_title: str,
    section_title: str,
    content: str,
) -> tuple[Counter[str], int]:
    """Return cached (term-frequency Counter, token count) for a chunk."""
    title_tokens = tokenize(
        f"{document_title} {section_title}",
        expand_synonyms=False,
    )
    body_tokens = tokenize(content, expand_synonyms=False)
    all_tokens = title_tokens * 2 + body_tokens
    return Counter(all_tokens), len(all_tokens)


def compute_bm25_scores(
    query: str,
    chunks: list[PolicyChunk],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute BM25 relevance scores with title boosting for policy chunks."""
    if not chunks:
        return []

    query_counts = Counter(tokenize(query, expand_synonyms=True))
    if not query_counts:
        return [0.0] * len(chunks)

    # Boost section title and document title tokens (cached per chunk content)
    doc_stats = [
        _chunk_token_stats(c.document_title, c.section_title, c.content)
        for c in chunks
    ]

    n_docs = len(chunks)
    avgdl = max(
        sum(doc_len for _, doc_len in doc_stats) / max(n_docs, 1),
        1.0,
    )

    # Precompute weighted IDF only for unique query terms
    idf_map: dict[str, float] = {}
    for q_tok, q_weight in query_counts.items():
        n_q = sum(1 for tf_map, _ in doc_stats if q_tok in tf_map)
        idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        idf_map[q_tok] = idf * q_weight

    query_term_set = idf_map.keys()
    k1_plus_1 = k1 + 1.0

    scores: list[float] = []
    for tf_map, doc_len in doc_stats:
        matching_terms = tf_map.keys() & query_term_set
        if not matching_terms:
            scores.append(0.0)
            continue

        norm_denom = k1 * (1.0 - b + b * (doc_len / avgdl))
        score = sum(
            idf_map[tok]
            * ((tf_map[tok] * k1_plus_1) / (tf_map[tok] + norm_denom))
            for tok in matching_terms
        )
        scores.append(score)

    return scores


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    norm_product = math.hypot(*vec_a) * math.hypot(*vec_b)
    if norm_product == 0.0:
        return 0.0
    return sum(map(mul, vec_a, vec_b)) / norm_product


def _default_gemini_embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using the Google GenAI SDK with model fallback."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY configured for embeddings.")

    client = genai.Client(api_key=api_key)
    preferred_model = get_gemini_embedding_model()
    candidate_models = list(
        dict.fromkeys(
            (preferred_model, "gemini-embedding-001", "gemini-embedding-2")
        )
    )

    last_err: Exception | None = None
    for model_name in candidate_models:
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=texts,
            )
            if response.embeddings:
                return [list(emb.values or []) for emb in response.embeddings]
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Embedding generation failed: {last_err}")


_MEMORY_EMBEDDING_CACHE: dict[str, list[float]] = {}


def _chunk_cache_key(chunk: PolicyChunk) -> str:
    """Build a deterministic cache key for a policy chunk."""
    digest = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
    return f"{chunk.chunk_id}:{digest}"


def _get_chunk_embeddings(
    chunks: list[PolicyChunk],
    embed_fn: EmbedFn,
    cache_path: Path | None = EMBEDDINGS_CACHE_PATH,
) -> list[list[float]]:
    """Retrieve or compute embeddings for policy chunks with local caching."""
    if not _MEMORY_EMBEDDING_CACHE and cache_path and cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                _MEMORY_EMBEDDING_CACHE.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass

    chunk_keys = [_chunk_cache_key(c) for c in chunks]
    missing_pairs = [
        (key, c.content)
        for key, c in zip(chunk_keys, chunks, strict=False)
        if key not in _MEMORY_EMBEDDING_CACHE
    ]

    if missing_pairs:
        missing_keys, missing_texts = zip(*missing_pairs, strict=False)
        new_vectors = embed_fn(list(missing_texts))
        _MEMORY_EMBEDDING_CACHE.update(
            zip(missing_keys, new_vectors, strict=False)
        )
        if cache_path and embed_fn is _default_gemini_embed:
            try:
                cache_path.write_text(
                    json.dumps(_MEMORY_EMBEDDING_CACHE),
                    encoding="utf-8",
                )
            except OSError:
                pass

    return [_MEMORY_EMBEDDING_CACHE[k] for k in chunk_keys]


def _resolve_embed_fn(
    mode: str,
    embed_fn: EmbedFn | None,
) -> EmbedFn | None:
    """Resolve the active embedding function for the requested search mode."""
    if mode in {"semantic", "hybrid"}:
        return embed_fn or _default_gemini_embed
    if mode == "auto" and (embed_fn is not None or get_gemini_api_key()):
        return embed_fn or _default_gemini_embed
    return None


def _combine_hybrid_score(
    b_score: float,
    s_score: float,
    max_bm25: float,
) -> float:
    """Combine normalized BM25 and semantic similarity scores."""
    norm_s = max(0.0, s_score)
    if b_score > 0.0:
        return round(0.55 * (b_score / max_bm25) + 0.45 * norm_s, 4)
    return round(norm_s, 4) if norm_s >= 0.45 else 0.0


def _calculate_final_scores(
    bm25_scores: list[float],
    semantic_scores: list[float] | None,
    mode_used: str,
) -> list[float]:
    """Calculate combined or single-mode relevance scores for chunks."""
    if mode_used == "semantic" and semantic_scores is not None:
        return semantic_scores
    if mode_used == "hybrid" and semantic_scores is not None:
        max_bm25 = max(bm25_scores, default=0.0) or 1.0
        return [
            _combine_hybrid_score(b_score, s_score, max_bm25)
            for b_score, s_score in zip(
                bm25_scores, semantic_scores, strict=False
            )
        ]
    return [round(s, 4) for s in bm25_scores]


def search_policies(
    query: str,
    top_k: int = 4,
    mode: str = "auto",
    policies_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Find the parts of the documents in data/policies/ relevant to a query.

    Supports:
    - 'bm25': lexical BM25 with domain synonym expansion (100% offline).
    - 'semantic': vector cosine similarity using Gemini embeddings.
    - 'hybrid' / 'auto': Reciprocal Rank Fusion (RRF) of BM25 + embeddings
      when an API key or embed_fn is available, falling back to BM25 offline.
    """
    staff_contact = {"email": STAFF_EMAIL, "phone": STAFF_PHONE}
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "query": str(query or ""),
            "mode_used": "none",
            "results": [],
            "staff_contact": staff_contact,
        }

    chunks = load_policy_chunks(policies_dir=policies_dir)
    if not chunks:
        return {
            "query": query,
            "mode_used": "none",
            "results": [],
            "staff_contact": staff_contact,
        }

    bm25_scores = compute_bm25_scores(query, chunks)
    active_embed_fn = _resolve_embed_fn(mode, embed_fn)

    semantic_scores: list[float] | None = None
    mode_used = "bm25"

    if active_embed_fn is not None:
        try:
            chunk_vecs = _get_chunk_embeddings(
                chunks,
                embed_fn=active_embed_fn,
                cache_path=(
                    EMBEDDINGS_CACHE_PATH if policies_dir is None else None
                ),
            )
            query_vec = active_embed_fn([query])[0]
            semantic_scores = [
                cosine_similarity(query_vec, c_vec) for c_vec in chunk_vecs
            ]
            mode_used = "semantic" if mode == "semantic" else "hybrid"
        except Exception:
            semantic_scores = None
            mode_used = "bm25"

    final_scores = _calculate_final_scores(
        bm25_scores, semantic_scores, mode_used
    )

    scored_chunks = [
        chunk.model_copy(update={"score": score})
        for chunk, score in zip(chunks, final_scores, strict=False)
        if score > 0.0
    ]
    top_chunks = heapq.nlargest(
        max(1, top_k),
        scored_chunks,
        key=lambda c: c.score,
    )

    return {
        "query": query,
        "mode_used": mode_used,
        "results": [
            {
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "document_title": c.document_title,
                "section_title": c.section_title,
                "content": c.content,
                "score": c.score,
            }
            for c in top_chunks
        ],
        "staff_contact": staff_contact,
    }
