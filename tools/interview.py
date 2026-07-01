"""
tools/interview.py
===================
generate_interview_questions @tool — Phase 4.5.

Fetches a candidate's resume text from ChromaDB, then asks the DeepSeek LLM to
generate targeted technical, behavioural, and gap-probing interview questions.

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
    INTERVIEW_QUESTIONS_SYSTEM,
    INTERVIEW_QUESTIONS_USER,
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


def _fetch_resume_text(candidate_id: str) -> str:
    """Retrieve all indexed chunks for a candidate and join into a single string."""
    import chromadb

    persist_dir = Path(CHROMA_PERSIST_DIR).resolve()
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(
            where={"candidate_id": candidate_id},
            include=["documents"],
        )
        docs = results.get("documents", [])
        return "\n\n".join(docs) if docs else ""
    except Exception as exc:
        logger.warning(
            "generate_interview_questions: failed to fetch resume for %r — %s",
            candidate_id, exc,
        )
        return ""


# ── Phase 4.5 — generate_interview_questions tool ─────────────────────────────

@tool
def generate_interview_questions(candidate_id: str, jd_text: str = "") -> dict:
    """Generate categorised interview questions for a specific candidate.

    Fetches the candidate's resume from ChromaDB and asks the DeepSeek LLM to
    produce targeted questions across three categories:

    - **technical**   — skill-specific questions drawn from the JD requirements.
    - **behavioural** — STAR-method questions probing soft skills and past behaviour.
    - **gap_probing** — questions targeting skills the candidate appears to lack.

    Args:
        candidate_id: The candidate's unique identifier (filename stem, e.g.
            ``"alice_chen_frontend"``).
        jd_text: Job description text to compare the resume against. If empty,
            the LLM generates generic questions from the resume alone.

    Returns:
        Dict with keys:
        - ``candidate_id``  — echoed back for traceability.
        - ``technical``     — list of ``{question, rationale}`` dicts.
        - ``behavioural``   — list of ``{question, rationale}`` dicts.
        - ``gap_probing``   — list of ``{question, gap}`` dicts.

    Raises:
        ValueError: If ``candidate_id`` is empty.
        RuntimeError: If the DeepSeek API fails after retries or returns invalid JSON.
    """
    if not candidate_id or not candidate_id.strip():
        raise ValueError("candidate_id must not be empty.")

    resume_text = _fetch_resume_text(candidate_id)
    if not resume_text:
        logger.warning(
            "generate_interview_questions: no resume text found for %r. "
            "Generating generic questions.",
            candidate_id,
        )
        resume_text = "(Resume not available — generate general questions for this role.)"

    jd_section = jd_text.strip() or "Not provided — generate questions based on the resume."

    llm = get_llm()
    messages = [
        SystemMessage(content=INTERVIEW_QUESTIONS_SYSTEM),
        HumanMessage(content=INTERVIEW_QUESTIONS_USER.format(
            jd_text=jd_section,
            candidate_id=candidate_id,
            resume_text=resume_text[:2000],   # cap to avoid token overflow
        )),
    ]

    logger.info(
        "generate_interview_questions: generating questions for %r via DeepSeek …",
        candidate_id,
    )
    response = call_llm_with_retry(llm, messages)

    cleaned = _extract_json(response.content)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"generate_interview_questions: LLM returned invalid JSON: {exc}\n"
            f"Raw (first 500 chars): {response.content[:500]}"
        ) from exc

    # Normalise output — ensure all expected keys exist
    result.setdefault("candidate_id", candidate_id)
    result.setdefault("technical", [])
    result.setdefault("behavioural", [])
    result.setdefault("gap_probing", [])

    logger.info(
        "generate_interview_questions: %d technical, %d behavioural, %d gap-probing questions.",
        len(result["technical"]),
        len(result["behavioural"]),
        len(result["gap_probing"]),
    )
    return result
