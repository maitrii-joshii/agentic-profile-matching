"""
tools/comparison.py
====================
compare_candidates @tool — Phase 4.4.

Accepts a list of candidate IDs, fetches their resume text from ChromaDB,
sends a head-to-head comparison request to the DeepSeek LLM, and returns a
structured comparison matrix plus summary.

Error handling (Phase 4.6): uses ``call_llm_with_retry`` (max 3 attempts).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from prompts.system import call_llm_with_retry, get_llm
from prompts.templates import (
    CANDIDATE_COMPARISON_SYSTEM,
    CANDIDATE_COMPARISON_USER,
)

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME: str = "resumes"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _fetch_candidate_chunks(candidate_ids: list[str]) -> dict[str, str]:
    """Retrieve all indexed chunks for each candidate from ChromaDB and join them."""
    import chromadb

    persist_dir = Path(CHROMA_PERSIST_DIR).resolve()
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.warning("compare_candidates: ChromaDB collection not found.")
        return {}

    profiles: dict[str, str] = {}
    for cid in candidate_ids:
        try:
            results = collection.get(
                where={"candidate_id": cid},
                include=["documents"],
            )
            docs = results.get("documents", [])
            profiles[cid] = "\n\n".join(docs) if docs else ""
        except Exception as exc:
            logger.warning("compare_candidates: failed to fetch %r — %s", cid, exc)
            profiles[cid] = ""

    return profiles


def _format_candidates_text(profiles: dict[str, str]) -> str:
    parts = []
    for cid, text in profiles.items():
        snippet = text[:1500] if text else "(no resume text available)"
        parts.append(f"--- Candidate: {cid} ---\n{snippet}")
    return "\n\n".join(parts)


# ── Phase 4.4 — compare_candidates tool ───────────────────────────────────────

@tool
def compare_candidates(candidate_ids: list[str], jd_text: str = "") -> dict:
    """Head-to-head comparison of multiple candidates using the DeepSeek LLM.

    Fetches each candidate's indexed resume text from ChromaDB, then asks the
    LLM to score and rank them against the job description.

    Args:
        candidate_ids: List of candidate ID strings (filename stems, e.g.
            ``["alice_chen_frontend", "bob_lee_backend"]``).
        jd_text: Optional job description to compare against. If empty, the
            comparison is purely based on candidate profiles.

    Returns:
        Dict with keys:
        - ``summary``       — 2-3 sentence overall comparison.
        - ``ranking``       — candidate IDs ordered best → worst.
        - ``matrix``        — per-candidate score breakdown.
        - ``head_to_head``  — paragraph comparing the top 2 directly.

    Raises:
        ValueError: If fewer than 2 candidate IDs are provided.
        RuntimeError: If the DeepSeek API fails after retries or returns invalid JSON.
    """
    if len(candidate_ids) < 2:
        raise ValueError("compare_candidates requires at least 2 candidate IDs.")

    profiles = _fetch_candidate_chunks(candidate_ids)
    candidates_text = _format_candidates_text(profiles)
    jd_section = jd_text.strip() or "Not provided — compare candidates on their profiles alone."

    llm = get_llm()
    messages = [
        SystemMessage(content=CANDIDATE_COMPARISON_SYSTEM),
        HumanMessage(content=CANDIDATE_COMPARISON_USER.format(
            jd_text=jd_section,
            candidates_text=candidates_text,
        )),
    ]

    logger.info("compare_candidates: comparing %d candidates via DeepSeek …", len(candidate_ids))
    response = call_llm_with_retry(llm, messages)

    cleaned = _extract_json(response.content)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"compare_candidates: LLM returned invalid JSON: {exc}\n"
            f"Raw (first 500 chars): {response.content[:500]}"
        ) from exc

    # Ensure expected keys
    result.setdefault("summary", "")
    result.setdefault("ranking", candidate_ids)
    result.setdefault("matrix", {})
    result.setdefault("head_to_head", "")

    logger.info("compare_candidates: comparison complete. Ranking: %s", result.get("ranking"))
    return result
