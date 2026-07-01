"""
tools/file_system.py
====================
File-system tools for the Agentic Profile Matching agent.

Phase implementation status:
  2.1  read_resume   ✅
  2.2  list_resumes  ✅

Both functions are wrapped with LangChain's ``@tool`` decorator so they can
be bound to a LangGraph ``ToolNode`` and called by the agent's LLM.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Supported extensions ───────────────────────────────────────────────────────
_SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".json", ".txt", ".docx"}


# ── Private helpers ────────────────────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    import pdfplumber  # deferred import — only needed for PDF files

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _read_json(path: Path) -> str:
    """
    Load a JSON resume and convert to a readable text block.

    If the JSON is a dict, each top-level key is rendered as a labelled
    section.  If the JSON is a list (or any other structure) it is serialised
    with indentation.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            header = key.replace("_", " ").title()
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = str(value)
            lines.append(f"{header}: {value_str}")
        return "\n".join(lines)

    # Fallback: pretty-print the whole structure
    return json.dumps(data, indent=2, ensure_ascii=False)


def _read_txt(path: Path) -> str:
    """Read a plain-text resume."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # deferred import — python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


# ── Phase 2.1 — read_resume tool ──────────────────────────────────────────────

@tool
def read_resume(file_path: str) -> str:
    """Read a resume from disk and return its full text content.

    Supports PDF (.pdf via pdfplumber), JSON (.json), plain text (.txt), and
    Word documents (.docx via python-docx).  The raw text is returned as-is,
    without any normalisation, so downstream tools and LLM prompts receive the
    original wording.

    Args:
        file_path: Absolute or relative path to the resume file.

    Returns:
        The full text content of the resume as a single string.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file extension is not supported.
        RuntimeError: If the file exists but cannot be parsed (e.g. corrupted
            PDF, malformed JSON).
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path!r}")

    ext = path.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension {ext!r} for file {file_path!r}. "
            f"Supported extensions: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext == ".pdf":
            text = _read_pdf(path)
        elif ext == ".json":
            text = _read_json(path)
        elif ext == ".txt":
            text = _read_txt(path)
        elif ext == ".docx":
            text = _read_docx(path)
        else:
            # Should never reach here given the check above
            raise ValueError(f"Unhandled extension: {ext!r}")
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("Failed to parse resume %r: %s", file_path, exc)
        raise RuntimeError(
            f"Could not parse resume {file_path!r}: {exc}"
        ) from exc

    if not text or not text.strip():
        logger.warning("Resume %r parsed to empty text.", file_path)

    return text


# ── Phase 2.2 — list_resumes tool ─────────────────────────────────────────────

@tool
def list_resumes(directory: str) -> list[str]:
    """Recursively list all resume file paths inside a directory.

    Only files with supported extensions (.pdf, .json, .txt, .docx) are
    returned.  Hidden files (those whose names start with a dot) and any file
    whose name contains ``~$`` (temporary Word lock files) are skipped.

    Args:
        directory: Path to the directory to search.  May be relative (resolved
            against the current working directory) or absolute.

    Returns:
        Sorted list of **absolute** path strings, one per discovered resume.
        Returns an empty list if the directory contains no matching files.

    Raises:
        NotADirectoryError: If ``directory`` does not point to a directory.
    """
    dir_path = Path(directory).resolve()

    if not dir_path.is_dir():
        raise NotADirectoryError(
            f"Not a directory (or does not exist): {directory!r}"
        )

    found: list[str] = []
    for root, _dirs, files in os.walk(dir_path):
        for filename in files:
            # Skip hidden files and Word temporary lock files
            if filename.startswith(".") or "~$" in filename:
                continue
            file_path = Path(root) / filename
            if file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                found.append(str(file_path))

    found.sort()
    logger.debug("list_resumes(%r) → %d file(s) found", directory, len(found))
    return found
