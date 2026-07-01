"""
ingestion/indexer.py
====================
Resume chunker & ChromaDB indexer — Phase 2.4.

Core responsibilities:
  1. Chunk raw resume text into ~500-token segments with 50-token overlap.
  2. Heuristically detect which resume section each chunk belongs to.
  3. Generate BGE embeddings via sentence-transformers.
  4. Upsert chunks + embeddings into a ChromaDB persistent collection.

Public API
----------
  index_resume(candidate_id, text, file_path)   — index a single resume
  chunk_text(text, chunk_size, overlap)          — split text into chunks
  detect_section(chunk)                          — classify a chunk's section
  get_collection()                               — return the ChromaDB collection
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME: str = "resumes"

# ── Lazy singletons ────────────────────────────────────────────────────────────
# Deferred so that importing this module doesn't immediately download the model
# or connect to ChromaDB (important for unit tests that mock these).
_model = None
_collection = None


def _get_model():
    """Return the SentenceTransformer singleton (lazy-loaded)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", BGE_MODEL_NAME)
        _model = SentenceTransformer(BGE_MODEL_NAME)
    return _model


def get_collection():
    """Return the ChromaDB collection singleton (lazy-loaded).

    Creates the collection if it does not already exist.  The ChromaDB client
    persists data to ``CHROMA_PERSIST_DIR`` so the index survives restarts.
    """
    global _collection
    if _collection is None:
        import chromadb
        persist_dir = Path(CHROMA_PERSIST_DIR).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Connecting to ChromaDB at: %s", persist_dir)
        client = chromadb.PersistentClient(path=str(persist_dir))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection %r ready (%d existing doc(s)).",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ── Section detection ──────────────────────────────────────────────────────────

# Ordered list of (pattern, section_label) pairs.  First match wins.
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(summary|objective|profile|about me)\b", re.I), "summary"),
    (re.compile(r"\b(skills?|technical skills?|competenc|technologies)\b", re.I), "skills"),
    (re.compile(r"\b(experience|employment|work history|career)\b", re.I), "experience"),
    (re.compile(r"\b(education|academic|degree|university|college|school)\b", re.I), "education"),
    (re.compile(r"\b(projects?|portfolio|open.?source)\b", re.I), "projects"),
    (re.compile(r"\b(certifications?|licenses?|awards?|achievements?)\b", re.I), "certifications"),
    (re.compile(r"\b(publications?|papers?|research)\b", re.I), "publications"),
    (re.compile(r"\b(contact|email|phone|linkedin|github)\b", re.I), "contact"),
]


def detect_section(chunk: str) -> str:
    """Heuristically classify a text chunk into a resume section label.

    Scans for section-header keywords in order of priority.  Falls back to
    ``"general"`` if no known section header is detected.

    Parameters
    ----------
    chunk:
        A single text chunk (part of a resume).

    Returns
    -------
    str
        One of: ``"summary"``, ``"skills"``, ``"experience"``,
        ``"education"``, ``"projects"``, ``"certifications"``,
        ``"publications"``, ``"contact"``, or ``"general"``.
    """
    for pattern, label in _SECTION_PATTERNS:
        if pattern.search(chunk):
            return label
    return "general"


# ── Text chunker ───────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split ``text`` into overlapping token-approximate chunks.

    Uses whitespace-split word count as a proxy for tokens (good enough for
    BGE's context window of 512 tokens).  Each chunk is ``chunk_size`` words
    long with an ``overlap``-word overlap with its predecessor.

    Parameters
    ----------
    text:
        The raw resume text to split.
    chunk_size:
        Target number of words per chunk (default 500).
    overlap:
        Number of words to repeat at the start of each successive chunk
        (default 50).

    Returns
    -------
    list[str]
        Non-empty list of text chunks.  Returns ``[""]`` for empty input so
        the caller always has at least one chunk to embed.
    """
    if not text or not text.strip():
        return [""]

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


# ── Core indexer ───────────────────────────────────────────────────────────────

def index_resume(
    candidate_id: str,
    text: str,
    file_path: str,
    collection=None,
    model=None,
) -> int:
    """Chunk, embed, and upsert a single resume into ChromaDB.

    Parameters
    ----------
    candidate_id:
        Unique identifier for the candidate (typically the filename stem,
        e.g. ``"alice_chen_frontend"``).
    text:
        Full raw text of the resume.
    file_path:
        Original path of the source file (stored as metadata).
    collection:
        Optional ChromaDB collection to use (defaults to the singleton).
        Inject a mock during testing.
    model:
        Optional SentenceTransformer instance (defaults to the singleton).
        Inject a mock during testing.

    Returns
    -------
    int
        The number of chunks that were upserted.
    """
    if not text or not text.strip():
        logger.warning("index_resume: empty text for candidate %r — skipping.", candidate_id)
        return 0

    if collection is None:
        collection = get_collection()
    if model is None:
        model = _get_model()

    chunks = chunk_text(text)
    logger.debug(
        "index_resume: %r → %d chunk(s) from %s", candidate_id, len(chunks), file_path
    )

    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    ids = [f"{candidate_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "candidate_id": candidate_id,
            "file_path": str(file_path),
            "section": detect_section(chunk),
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "index_resume: upserted %d chunk(s) for candidate %r.", len(chunks), candidate_id
    )
    return len(chunks)
