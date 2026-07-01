"""
tools/rag_search.py
====================
RAG (Retrieval-Augmented Generation) search tool — Phase 3.

Phase implementation status:
  3.1  rag_search tool   ✅
  3.2  Deduplication     ✅

Public API
----------
  rag_search(query, top_k)          — @tool: semantic search over ChromaDB
  deduplicate_and_rank(raw, top_k)  — keep best chunk per candidate, re-rank
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME: str = "resumes"

# ── Lazy singletons (reuse same model/collection as the indexer if already
#    loaded, otherwise create fresh ones — each module load is independent) ─────
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("RAG search: loading embedding model %s", BGE_MODEL_NAME)
        _model = SentenceTransformer(BGE_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from pathlib import Path
        persist_dir = Path(CHROMA_PERSIST_DIR).resolve()
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            _collection = client.get_collection(name=COLLECTION_NAME)
            logger.info(
                "RAG search: connected to collection %r (%d doc(s)).",
                COLLECTION_NAME,
                _collection.count(),
            )
        except Exception:
            # Collection doesn't exist yet — return None so rag_search can
            # return an empty list instead of crashing.
            logger.warning(
                "RAG search: collection %r not found. Run `python -m ingestion` first.",
                COLLECTION_NAME,
            )
            _collection = None
    return _collection


# ── Phase 3.2 — Deduplication & ranking ───────────────────────────────────────

def deduplicate_and_rank(
    raw_results: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Collapse multiple chunks from the same candidate, keep the best score.

    ChromaDB ``collection.query`` may return several chunks for a single
    candidate when that candidate's resume was split into multiple segments.
    This function:

    1. Iterates all returned (document, metadata, distance) triples.
    2. Converts ChromaDB cosine *distance* → *relevance score* (1 − distance),
       clamped to [0, 1].
    3. Groups by ``candidate_id``; for each candidate, keeps the chunk with
       the highest relevance score and uses its snippet.
    4. Sorts the deduplicated list by relevance score descending.
    5. Truncates to ``top_k``.

    Parameters
    ----------
    raw_results:
        The dict returned by ``collection.query(...)``.  Expected keys:
        ``"documents"``, ``"metadatas"``, ``"distances"``, ``"ids"``.
    top_k:
        Maximum number of deduplicated candidates to return.

    Returns
    -------
    list[dict]
        Each element has keys:
        ``candidate_id``, ``file_path``, ``relevance_score``, ``snippet``.
    """
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    # Build a mapping: candidate_id → best result so far
    best: dict[str, dict[str, Any]] = {}

    for doc, meta, dist in zip(documents, metadatas, distances):
        cid = meta.get("candidate_id", "unknown")
        # ChromaDB cosine distance ∈ [0, 2]; relevance = 1 − distance (clamped)
        score = float(max(0.0, min(1.0, 1.0 - dist)))

        if cid not in best or score > best[cid]["relevance_score"]:
            best[cid] = {
                "candidate_id": cid,
                "file_path": meta.get("file_path", ""),
                "relevance_score": round(score, 4),
                "snippet": doc[:500] if doc else "",   # first 500 chars of chunk
            }

    # Sort by relevance descending, truncate
    ranked = sorted(best.values(), key=lambda x: x["relevance_score"], reverse=True)
    return ranked[:top_k]


# ── Phase 3.1 — rag_search @tool ──────────────────────────────────────────────

@tool
def rag_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic search over indexed resumes in ChromaDB.

    Embeds ``query`` using the BGE model, retrieves the ``top_k * 3`` most
    similar resume chunks from ChromaDB, deduplicates by candidate (keeping
    the highest-scoring chunk per person), and returns the top ``top_k``
    unique candidates ranked by relevance.

    Args:
        query: Free-text search query describing the desired candidate profile.
            Examples: ``"React developer 3 years"``,
            ``"machine learning Python TensorFlow"``.
        top_k: Maximum number of unique candidates to return (default 10).

    Returns:
        List of dicts, each containing:
        - ``candidate_id`` (str): Filename stem of the resume.
        - ``file_path``    (str): Absolute path to the source file.
        - ``relevance_score`` (float): Cosine similarity in [0, 1]; higher = better.
        - ``snippet``      (str): Most relevant text excerpt from the resume.

        Returns an empty list if the index is empty or the query finds nothing.

    Raises:
        RuntimeError: If the embedding model fails to encode the query.
    """
    if not query or not query.strip():
        logger.warning("rag_search called with empty query — returning [].")
        return []

    collection = _get_collection()
    if collection is None or collection.count() == 0:
        logger.warning("rag_search: ChromaDB collection is empty or missing.")
        return []

    # Clamp top_k to a sensible range
    top_k = max(1, min(top_k, 50))
    n_retrieve = top_k * 3          # over-fetch so dedup still yields top_k

    try:
        model = _get_model()
        query_embedding = model.encode(query, show_progress_bar=False).tolist()
    except Exception as exc:
        logger.error("rag_search: failed to encode query — %s", exc)
        raise RuntimeError(f"Embedding model failed for query {query!r}: {exc}") from exc

    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_retrieve, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("rag_search: ChromaDB query failed — %s", exc)
        return []

    results = deduplicate_and_rank(raw, top_k)

    logger.info(
        "rag_search(%r, top_k=%d) → %d unique candidate(s) returned.",
        query[:60],
        top_k,
        len(results),
    )
    return results
