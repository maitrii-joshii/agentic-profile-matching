"""
tools/requirements.py
======================
extract_requirements @tool — Phase 4.3.

Sends a job description to the DeepSeek LLM and parses the response into a
structured dict with must_have / nice_to_have requirements, experience range,
and education requirements.

Error handling (Phase 4.6):
  - DeepSeek API calls go through ``call_llm_with_retry`` (max 3 attempts,
    exponential back-off).
  - JSON parse failures are retried once with a corrective prompt; if still
    invalid, a ``RuntimeError`` is raised.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from prompts.system import call_llm_with_retry, get_llm
from prompts.templates import (
    REQUIREMENT_EXTRACTION_SYSTEM,
    REQUIREMENT_EXTRACTION_USER,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the first JSON object from text."""
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    # Find the outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _parse_requirements_response(raw_text: str) -> dict:
    """Parse the LLM's JSON response into a requirements dict.

    Raises
    ------
    ValueError
        If the response cannot be parsed as valid JSON after stripping fences.
    """
    cleaned = _extract_json(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON for requirements: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]}"
        ) from exc

    # Normalise — ensure all expected keys exist with sensible defaults
    return {
        "must_have": data.get("must_have", []),
        "nice_to_have": data.get("nice_to_have", []),
        "experience_range": data.get("experience_range", {"min_years": None, "max_years": None}),
        "education": data.get("education", "Not specified"),
    }


# ── Phase 4.3 — extract_requirements tool ─────────────────────────────────────

@tool
def extract_requirements(jd_text: str) -> dict:
    """Extract structured hiring requirements from a job description.

    Sends the job description to the DeepSeek LLM (deepseek-chat or deepseek-coder) with
    a structured-output prompt and parses the JSON response into a dict with
    four keys:

    - ``must_have``       — list of ``{skill, category, min_years}`` dicts
    - ``nice_to_have``    — list of ``{skill, category, min_years}`` dicts
    - ``experience_range``— ``{min_years, max_years}`` dict (values may be null)
    - ``education``       — string describing education requirements

    Args:
        jd_text: Raw job description text (pasted or extracted from a file).

    Returns:
        Structured requirements dict ready to be stored in ``AgentState``.

    Raises:
        ValueError: If ``jd_text`` is empty.
        RuntimeError: If the DeepSeek API fails after 3 retries, or if the LLM
            returns unparseable JSON after a correction attempt.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text must not be empty.")

    llm = get_llm()
    messages = [
        SystemMessage(content=REQUIREMENT_EXTRACTION_SYSTEM),
        HumanMessage(content=REQUIREMENT_EXTRACTION_USER.format(jd_text=jd_text)),
    ]

    logger.info("extract_requirements: calling DeepSeek LLM …")
    response = call_llm_with_retry(llm, messages)
    raw_text: str = response.content

    try:
        result = _parse_requirements_response(raw_text)
    except ValueError as first_err:
        # One correction attempt: ask the LLM to fix its output
        logger.warning(
            "extract_requirements: JSON parse failed — asking LLM to correct. Error: %s",
            first_err,
        )
        correction_messages = messages + [
            response,
            HumanMessage(
                content=(
                    "Your response was not valid JSON. "
                    "Please return ONLY the JSON object, with no markdown, "
                    "no backticks, and no extra text."
                )
            ),
        ]
        response2 = call_llm_with_retry(llm, correction_messages)
        try:
            result = _parse_requirements_response(response2.content)
        except ValueError as second_err:
            raise RuntimeError(
                f"extract_requirements: LLM returned invalid JSON after correction attempt. "
                f"Error: {second_err}"
            ) from second_err

    logger.info(
        "extract_requirements: extracted %d must-have and %d nice-to-have requirements.",
        len(result["must_have"]),
        len(result["nice_to_have"]),
    )
    return result

@tool
def refine_requirements(current_reqs: dict, instruction: str) -> dict:
    """Refine existing hiring requirements based on a recruiter's instruction.
    
    Args:
        current_reqs: The current parsed requirements dict.
        instruction: The recruiter's refinement message (e.g., "Add TypeScript as must-have").
    """
    llm = get_llm()
    from prompts.templates import REQUIREMENT_REFINEMENT_SYSTEM, REQUIREMENT_REFINEMENT_USER
    
    messages = [
        SystemMessage(content=REQUIREMENT_REFINEMENT_SYSTEM),
        HumanMessage(content=REQUIREMENT_REFINEMENT_USER.format(
            current_reqs_json=json.dumps(current_reqs, indent=2),
            instruction=instruction
        )),
    ]

    logger.info("refine_requirements: calling DeepSeek LLM with instruction %r …", instruction[:50])
    response = call_llm_with_retry(llm, messages)
    
    try:
        result = _parse_requirements_response(response.content)
    except ValueError as first_err:
        logger.warning("refine_requirements: JSON parse failed — asking LLM to correct. Error: %s", first_err)
        correction_messages = messages + [
            response,
            HumanMessage(
                content="Your response was not valid JSON. Please return ONLY the JSON object, with no extra text."
            ),
        ]
        response2 = call_llm_with_retry(llm, correction_messages)
        try:
            result = _parse_requirements_response(response2.content)
        except ValueError as second_err:
            raise RuntimeError(f"refine_requirements: JSON invalid after correction. {second_err}") from second_err

    logger.info(
        "refine_requirements: refined to %d must-have and %d nice-to-have requirements.",
        len(result["must_have"]),
        len(result["nice_to_have"]),
    )
    return result
