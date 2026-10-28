"""Policy search tool (Markdown chunking + BM25 & Gemini Embeddings RAG)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

STOPWORDS = {
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

SYNONYM_MAP: dict[str, list[str]] = {
    "saturday": ["open", "monday", "sunday", "april", "october", "november"],
    "sunday": ["open", "closed", "sunday", "november", "march"],
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


def _normalize_token(token: str) -> str:
    """Apply lightweight normalization/stemming to a lowercase word token."""
    if token in ("e-bike", "ebikes", "e-bikes"):
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
    raw_tokens = re.findall(r"[a-z0-9]+", cleaned)
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        norm = _normalize_token(tok)
        if norm and norm not in STOPWORDS:
            tokens.append(norm)
            if expand_synonyms and norm in SYNONYM_MAP:
                tokens.extend(SYNONYM_MAP[norm])
    return tokens


def load_policy_chunks(policies_dir: Path | None = None) -> list[PolicyChunk]:
    """Parse markdown files in data/policies/ into logical section chunks."""
    directory = policies_dir or POLICIES_DIR
    if not directory.exists():
        return []

    chunks: list[PolicyChunk] = []
    for md_file in sorted(directory.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8").strip()
        if not raw_text:
            continue

        lines = raw_text.splitlines()
        doc_title = md_file.stem.capitalize()
        preamble_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current_section_title: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("# ") and not line.startswith("## "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if current_section_title is not None:
                    sections.append((current_section_title, current_lines))
                current_section_title = line[3:].strip()
                current_lines = [line]
            else:
                if current_section_title is None:
                    if line.strip():
                        preamble_lines.append(line.strip())
                else:
                    current_lines.append(line)

        if current_section_title is not None:
            sections.append((current_section_title, current_lines))

        preamble = "\n".join(preamble_lines).strip()

        if not sections:
            slug = re.sub(r"[^a-z0-9]+", "-", doc_title.lower()).strip("-")
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{md_file.name}#{slug}",
                    source_file=md_file.name,
                    document_title=doc_title,
                    section_title=doc_title,
                    content=raw_text,
                )
            )
            continue

        for sec_title, sec_lines in sections:
            slug = re.sub(r"[^a-z0-9]+", "-", sec_title.lower()).strip("-")
            body = "\n".join(sec_lines).strip()
            if preamble:
                full_content = f"# {doc_title}\n{preamble}\n\n{body}"
            else:
                full_content = f"# {doc_title}\n\n{body}"

            chunks.append(
                PolicyChunk(
                    chunk_id=f"{md_file.name}#{slug}",
                    source_file=md_file.name,
                    document_title=doc_title,
                    section_title=sec_title,
                    content=full_content,
                )
            )

    return chunks


def compute_bm25_scores(
    query: str,
    chunks: list[PolicyChunk],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute BM25 relevance scores with title boosting for policy chunks."""
    if not chunks:
        return []

    query_tokens = tokenize(query, expand_synonyms=True)
    if not query_tokens:
        return [0.0 for _ in chunks]

    doc_tokens_list: list[list[str]] = []
    for chunk in chunks:
        # Boost section title and document title tokens
        title_tokens = tokenize(
            f"{chunk.document_title} {chunk.section_title}",
            expand_synonyms=False,
        )
        body_tokens = tokenize(chunk.content, expand_synonyms=False)
        doc_tokens_list.append(title_tokens * 2 + body_tokens)

    n_docs = len(chunks)
    avgdl = sum(len(t) for t in doc_tokens_list) / max(n_docs, 1)

    # Document frequencies
    df: dict[str, int] = {}
    for tokens in doc_tokens_list:
        for unique_tok in set(tokens):
            df[unique_tok] = df.get(unique_tok, 0) + 1

    scores: list[float] = []
    for tokens in doc_tokens_list:
        score = 0.0
        doc_len = len(tokens)
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        for q_tok in query_tokens:
            if q_tok not in tf_map:
                continue
            tf = tf_map[q_tok]
            n_q = df.get(q_tok, 0)
            idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
            denom = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1.0)))
            score += idf * ((tf * (k1 + 1.0)) / denom)

        scores.append(score)

    return scores


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _default_gemini_embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using the Google GenAI SDK with model fallback."""
    from google import genai

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY configured for embeddings.")

    client = genai.Client(api_key=api_key)
    preferred_model = get_gemini_embedding_model()
    candidate_models = [preferred_model]
    for fallback in ("gemini-embedding-001", "gemini-embedding-2"):
        if fallback not in candidate_models:
            candidate_models.append(fallback)

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


def _get_chunk_embeddings(
    chunks: list[PolicyChunk],
    embed_fn: EmbedFn,
    cache_path: Path | None = EMBEDDINGS_CACHE_PATH,
) -> list[list[float]]:
    """Retrieve or compute embeddings for policy chunks with local caching."""
    global _MEMORY_EMBEDDING_CACHE

    if not _MEMORY_EMBEDDING_CACHE and cache_path and cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                _MEMORY_EMBEDDING_CACHE.update(loaded)
        except Exception:
            pass

    missing_texts: list[str] = []
    missing_keys: list[str] = []
    chunk_keys: list[str] = []

    for chunk in chunks:
        digest = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
        key = f"{chunk.chunk_id}:{digest}"
        chunk_keys.append(key)
        if key not in _MEMORY_EMBEDDING_CACHE:
            missing_keys.append(key)
            missing_texts.append(chunk.content)

    if missing_texts:
        new_vectors = embed_fn(missing_texts)
        for key, vec in zip(missing_keys, new_vectors, strict=False):
            _MEMORY_EMBEDDING_CACHE[key] = vec
        if cache_path and embed_fn is _default_gemini_embed:
            try:
                cache_path.write_text(
                    json.dumps(_MEMORY_EMBEDDING_CACHE),
                    encoding="utf-8",
                )
            except Exception:
                pass

    return [_MEMORY_EMBEDDING_CACHE[k] for k in chunk_keys]


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
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "query": str(query or ""),
            "mode_used": "none",
            "results": [],
            "staff_contact": {"email": STAFF_EMAIL, "phone": STAFF_PHONE},
        }

    chunks = load_policy_chunks(policies_dir=policies_dir)
    if not chunks:
        return {
            "query": query,
            "mode_used": "none",
            "results": [],
            "staff_contact": {"email": STAFF_EMAIL, "phone": STAFF_PHONE},
        }

    bm25_scores = compute_bm25_scores(query, chunks)

    use_semantic = False
    active_embed_fn = embed_fn
    if mode in ("semantic", "hybrid"):
        use_semantic = True
        if active_embed_fn is None:
            active_embed_fn = _default_gemini_embed
    elif mode == "auto":
        if active_embed_fn is not None:
            use_semantic = True
        elif get_gemini_api_key() is not None:
            use_semantic = True
            active_embed_fn = _default_gemini_embed

    semantic_scores: list[float] | None = None
    mode_used = "bm25"

    if use_semantic and active_embed_fn is not None:
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

    # Combine scores
    final_scores: list[float] = []
    if mode_used == "semantic" and semantic_scores is not None:
        final_scores = semantic_scores
    elif mode_used == "hybrid" and semantic_scores is not None:
        # Normalized hybrid combination (BM25 + Semantic similarity)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for b_score, s_score in zip(
            bm25_scores, semantic_scores, strict=False
        ):
            norm_b = b_score / max_bm25 if max_bm25 > 0 else 0.0
            norm_s = max(0.0, s_score)
            # If BM25 found an exact match, combine both; if BM25 is 0,
            # only trust semantic if similarity is reasonably strong (>= 0.45)
            if b_score > 0:
                combined = 0.55 * norm_b + 0.45 * norm_s
            else:
                combined = norm_s if norm_s >= 0.45 else 0.0
            final_scores.append(round(combined, 4))
    else:
        final_scores = [round(s, 4) for s in bm25_scores]

    scored_chunks: list[PolicyChunk] = []
    for chunk, score in zip(chunks, final_scores, strict=False):
        if score > 0.0:
            scored_chunks.append(chunk.model_copy(update={"score": score}))

    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    top_chunks = scored_chunks[: max(1, top_k)]

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
        "staff_contact": {
            "email": STAFF_EMAIL,
            "phone": STAFF_PHONE,
        },
    }
