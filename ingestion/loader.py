"""
ingestion/loader.py
====================
Unified resume loader — Phase 2.3.

``load_resume(path)`` is the single entry-point used by the indexer and the
CLI.  It delegates format detection to ``tools.file_system`` helpers so there
is only one place where format-specific logic lives.

Corrupted or unreadable files are handled gracefully: a warning is logged and
an empty string is returned so the caller can decide whether to skip the file.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Re-use the private format readers from the file_system tool module so we
# don't duplicate PDF / JSON / TXT / DOCX parsing logic.
from tools.file_system import (
    _read_docx,
    _read_json,
    _read_pdf,
    _read_txt,
    _SUPPORTED_EXTENSIONS,
)


def load_resume(path: str | Path) -> str:
    """Load a resume from disk and return its raw text content.

    This is the unified entry-point for all resume parsing inside the
    ingestion pipeline.  Unlike the ``read_resume`` LangChain tool, this
    function does **not** raise on recoverable errors — instead it logs a
    warning and returns an empty string so the batch indexer can skip the
    file without crashing the whole run.

    Supported formats
    -----------------
    - ``.pdf``  — extracted via *pdfplumber*
    - ``.json`` — loaded and flattened to a readable text block
    - ``.txt``  — read directly with UTF-8 (with error replacement)
    - ``.docx`` — extracted via *python-docx*

    Parameters
    ----------
    path:
        Absolute or relative path to the resume file.

    Returns
    -------
    str
        Raw text of the resume.  May be an empty string if the file is
        corrupted, empty, or of an unsupported format.
    """
    p = Path(path).resolve()

    if not p.exists():
        logger.warning("load_resume: file not found — %s", p)
        return ""

    ext = p.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        logger.warning(
            "load_resume: unsupported extension %r — skipping %s", ext, p
        )
        return ""

    try:
        if ext == ".pdf":
            text = _read_pdf(p)
        elif ext == ".json":
            text = _read_json(p)
        elif ext == ".txt":
            text = _read_txt(p)
        elif ext == ".docx":
            text = _read_docx(p)
        else:
            logger.warning("load_resume: unhandled extension %r — skipping %s", ext, p)
            return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_resume: failed to parse %s — %s", p, exc)
        return ""

    if not text or not text.strip():
        logger.warning("load_resume: parsed to empty text — %s", p)

    return text
