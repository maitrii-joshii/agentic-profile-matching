"""
ingestion/__init__.py
======================
Ingestion package — Phase 2.5: CLI entry point.

Run bulk indexing with:
    python -m ingestion

This will:
  1. Discover all supported resume files under RESUME_DIR.
  2. Load each file via ``ingestion.loader.load_resume``.
  3. Index each resume into ChromaDB via ``ingestion.indexer.index_resume``.
  4. Print a summary of successes / skips.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Package-level logger ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESUME_DIR = Path(os.getenv("RESUME_DIR", "./data/resumes"))


# ── CLI main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Bulk-index all resumes found in RESUME_DIR into ChromaDB."""
    from tools.file_system import list_resumes
    from ingestion.loader import load_resume
    from ingestion.indexer import index_resume

    if not RESUME_DIR.exists():
        logger.error("RESUME_DIR does not exist: %s", RESUME_DIR.resolve())
        sys.exit(1)

    logger.info("Scanning for resumes in: %s", RESUME_DIR.resolve())
    resume_paths: list[str] = list_resumes.invoke(str(RESUME_DIR))

    if not resume_paths:
        logger.warning("No supported resume files found in %s. Nothing to index.", RESUME_DIR)
        sys.exit(0)

    logger.info("Found %d resume(s). Starting indexing…", len(resume_paths))

    total_chunks = 0
    skipped = 0

    for file_path in resume_paths:
        candidate_id = Path(file_path).stem          # e.g. "alice_chen_frontend"
        logger.info("Processing: %s", Path(file_path).name)

        text = load_resume(file_path)

        if not text or not text.strip():
            logger.warning("  ↳ Skipped (empty or unreadable): %s", file_path)
            skipped += 1
            continue

        chunks_added = index_resume(
            candidate_id=candidate_id,
            text=text,
            file_path=file_path,
        )
        total_chunks += chunks_added
        logger.info("  ↳ Indexed %d chunk(s) as %r", chunks_added, candidate_id)

    logger.info(
        "\n✅  Indexing complete — %d resume(s) indexed, %d skipped, %d total chunk(s) in ChromaDB.",
        len(resume_paths) - skipped,
        skipped,
        total_chunks,
    )


if __name__ == "__main__":
    main()
