"""
nodes/parse_jd.py
==================
Parse Job Description node — Phase 5.1.

Reads the most recent human message from ``state["messages"]``, extracts the
job description text (inline paste or file-path reference), normalises it, and
writes it to ``state["raw_jd"]``.

State writes: raw_jd
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from state import AgentState

logger = logging.getLogger(__name__)

# Patterns that suggest the message contains a file path rather than inline JD text
_FILE_PATH_PATTERN = re.compile(
    r"(?:^|\s)((?:[A-Za-z]:[\\/]|\.{0,2}[\\/]|\\/)[^\s]+\.(?:pdf|txt|docx|json))",
    re.IGNORECASE,
)


def _extract_jd_from_message(text: str) -> str:
    """Try to load JD from a file path embedded in the message, else return text as-is."""
    match = _FILE_PATH_PATTERN.search(text)
    if match:
        file_path = Path(match.group(1).strip())
        if file_path.exists():
            logger.info("parse_jd: loading JD from file %s", file_path)
            try:
                from tools.file_system import read_resume
                return read_resume.invoke(str(file_path))
            except Exception as exc:
                logger.warning("parse_jd: failed to read JD file %s — %s. Using raw text.", file_path, exc)

    # No valid file path — treat the whole message as the JD
    return text


def _normalise(text: str) -> str:
    """Collapse excessive blank lines and strip leading/trailing whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run(state: AgentState) -> dict:
    """Extract and normalise the job description from the latest user message.

    Looks at the last ``HumanMessage`` in ``state["messages"]``. If it
    contains a file path, the file is loaded; otherwise the message content
    itself is treated as the JD.

    Returns a partial state update with ``raw_jd`` populated.
    """
    messages = state.get("messages", [])

    # Find the most recent human message
    jd_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            jd_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not jd_text:
        logger.warning("parse_jd: no human message found — raw_jd will be empty.")
        return {"raw_jd": ""}

    jd_text = _extract_jd_from_message(jd_text)
    jd_text = _normalise(jd_text)

    logger.info("parse_jd: raw_jd set (%d chars).", len(jd_text))
    return {"raw_jd": jd_text}
