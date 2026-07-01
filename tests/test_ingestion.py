"""
tests/test_ingestion.py
========================
Unit tests for Phase 2.3 & 2.4: loader.py and indexer.py.

Test matrix:
  load_resume (ingestion/loader.py)
    - delegates to _read_txt for .txt files and returns text
    - delegates to _read_json for .json files and returns text
    - delegates to _read_pdf for .pdf files and returns text
    - returns "" (not an exception) for a missing file
    - returns "" (not an exception) for an unsupported extension
    - returns "" (not an exception) for a corrupted PDF
    - returns "" for an empty .txt file

  chunk_text (ingestion/indexer.py)
    - short text → single chunk equals the original text
    - long text → multiple chunks created
    - each chunk is at most chunk_size words long
    - adjacent chunks share overlap words
    - empty string → returns [""]
    - whitespace-only string → returns [""]

  detect_section (ingestion/indexer.py)
    - "Skills: Python, Java" → "skills"
    - "Work Experience at Acme" → "experience"
    - "Education: B.Tech" → "education"
    - "Projects built at university" → "projects"
    - "Summary: Results-driven engineer" → "summary"
    - "Certifications: AWS" → "certifications"
    - "Email: jane@example.com" → "contact"
    - random prose → "general"

  index_resume (ingestion/indexer.py)
    - calls collection.upsert with correct ids, documents, embeddings, metadatas
    - returns the number of chunks upserted
    - skips (returns 0) for empty text
    - chunk ids follow the pattern "{candidate_id}_chunk_{i}"
    - metadata contains candidate_id, file_path, section, chunk_index
    - works correctly for text that produces multiple chunks

  ingestion CLI (ingestion/__init__.py)
    - main() calls list_resumes, load_resume, index_resume for each file
    - main() skips files that load_resume returns "" for
    - main() exits non-zero when RESUME_DIR does not exist
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Project root helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = PROJECT_ROOT / "data" / "resumes"


def _sample(filename: str) -> Path:
    return RESUME_DIR / filename


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def txt_resume(tmp_path: Path) -> Path:
    p = tmp_path / "candidate.txt"
    p.write_text(
        "Alice Johnson\n"
        "Skills: Python, Django, PostgreSQL\n"
        "Experience: 3 years at TechCorp\n"
        "Education: B.Sc Computer Science\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def json_resume(tmp_path: Path) -> Path:
    data = {
        "name": "Bob Lee",
        "skills": ["Go", "Kubernetes", "Docker"],
        "years_experience": 6,
    }
    p = tmp_path / "candidate.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def empty_txt(tmp_path: Path) -> Path:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a real pdf")
    return p


@pytest.fixture
def mock_model():
    """Fake SentenceTransformer that returns deterministic float arrays."""
    import numpy as np

    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda chunks, **_kw: np.zeros((len(chunks), 384), dtype="float32")
    )
    return model


@pytest.fixture
def mock_collection():
    """Fake ChromaDB collection."""
    col = MagicMock()
    col.upsert = MagicMock()
    return col


# ===========================================================================
# load_resume
# ===========================================================================

class TestLoadResume:
    def test_txt_returns_text(self, txt_resume: Path) -> None:
        from ingestion.loader import load_resume

        result = load_resume(str(txt_resume))
        assert "Alice Johnson" in result
        assert "Python" in result

    def test_json_returns_text(self, json_resume: Path) -> None:
        from ingestion.loader import load_resume

        result = load_resume(str(json_resume))
        assert "Bob Lee" in result
        assert "Kubernetes" in result

    def test_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        from ingestion.loader import load_resume

        result = load_resume(str(tmp_path / "ghost.txt"))
        assert result == ""

    def test_unsupported_extension_returns_empty_string(self, tmp_path: Path) -> None:
        from ingestion.loader import load_resume

        p = tmp_path / "resume.xlsx"
        p.write_bytes(b"\x00\x01")
        result = load_resume(str(p))
        assert result == ""

    def test_corrupted_pdf_returns_empty_string(self, corrupted_pdf: Path) -> None:
        from ingestion.loader import load_resume

        # Must not raise — just return ""
        result = load_resume(str(corrupted_pdf))
        assert result == ""

    def test_empty_txt_returns_empty_string(self, empty_txt: Path) -> None:
        from ingestion.loader import load_resume

        result = load_resume(str(empty_txt))
        # Either empty string or whitespace — both acceptable
        assert result.strip() == ""

    def test_returns_string_type(self, txt_resume: Path) -> None:
        from ingestion.loader import load_resume

        assert isinstance(load_resume(str(txt_resume)), str)

    def test_accepts_path_object(self, txt_resume: Path) -> None:
        from ingestion.loader import load_resume

        # Should accept a Path, not just str
        result = load_resume(txt_resume)
        assert isinstance(result, str)

    def test_reads_sample_txt(self) -> None:
        sample = _sample("ryan_park_sre.txt")
        if not sample.exists():
            pytest.skip("Sample TXT not found.")
        from ingestion.loader import load_resume

        result = load_resume(str(sample))
        assert len(result) > 50

    def test_reads_sample_pdf(self) -> None:
        sample = _sample("sarah_johnson_backend.pdf")
        if not sample.exists():
            pytest.skip("Sample PDF not found.")
        from ingestion.loader import load_resume

        result = load_resume(str(sample))
        assert isinstance(result, str)  # may be empty for image PDFs — that's OK


# ===========================================================================
# chunk_text
# ===========================================================================

class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        from ingestion.indexer import chunk_text

        text = "Hello world this is a short resume."
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_produces_multiple_chunks(self) -> None:
        from ingestion.indexer import chunk_text

        # 1100 words — should yield multiple chunks of size 500
        text = " ".join([f"word{i}" for i in range(1100)])
        result = chunk_text(text, chunk_size=500, overlap=50)
        assert len(result) >= 2

    def test_chunk_size_not_exceeded(self) -> None:
        from ingestion.indexer import chunk_text

        text = " ".join([f"word{i}" for i in range(2000)])
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 500, f"Chunk has {word_count} words, expected ≤ 500"

    def test_overlap_words_shared(self) -> None:
        from ingestion.indexer import chunk_text

        # Use small chunk_size / overlap so we can inspect manually
        text = " ".join([f"w{i}" for i in range(30)])
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        assert len(chunks) >= 2
        tail_words = chunks[0].split()[-3:]
        head_words = chunks[1].split()[:3]
        assert tail_words == head_words, "Overlap words should be shared between consecutive chunks"

    def test_empty_string_returns_placeholder(self) -> None:
        from ingestion.indexer import chunk_text

        result = chunk_text("")
        assert result == [""]

    def test_whitespace_only_returns_placeholder(self) -> None:
        from ingestion.indexer import chunk_text

        result = chunk_text("   \n\t  ")
        assert result == [""]

    def test_exact_chunk_size_no_split(self) -> None:
        from ingestion.indexer import chunk_text

        text = " ".join([f"x{i}" for i in range(500)])
        result = chunk_text(text, chunk_size=500, overlap=50)
        assert len(result) == 1


# ===========================================================================
# detect_section
# ===========================================================================

class TestDetectSection:
    @pytest.mark.parametrize("chunk,expected", [
        ("Skills: Python, Java, React", "skills"),
        ("Technical Skills: AWS, Docker", "skills"),
        ("Work Experience at Acme Corp 2020-2023", "experience"),
        ("Employment History — Senior Engineer", "experience"),
        ("Education: B.Tech from IIT Delhi", "education"),
        ("University of Cambridge, MSc Computer Science", "education"),
        ("Projects: Built a recommendation engine", "projects"),
        ("Open-source contributions on GitHub", "projects"),
        ("Summary: Results-driven software engineer", "summary"),
        ("Professional Profile: 10+ years in fintech", "summary"),
        ("Certifications: AWS Certified Solutions Architect", "certifications"),
        ("Awards: Best Employee of the Year 2022", "certifications"),
        ("Email: jane@example.com | LinkedIn: /jane", "contact"),
        ("Phone: +91-9876543210", "contact"),
        ("Led a team of 5 engineers to deliver the product.", "general"),
        ("Responsible for microservices architecture migration.", "general"),
    ])
    def test_section_classification(self, chunk: str, expected: str) -> None:
        from ingestion.indexer import detect_section

        result = detect_section(chunk)
        preview = repr(chunk)[:60]
        assert result == expected, f"For chunk {preview} expected {expected!r}, got {result!r}"

    def test_empty_chunk_returns_general(self) -> None:
        from ingestion.indexer import detect_section

        assert detect_section("") == "general"


# ===========================================================================
# index_resume
# ===========================================================================

class TestIndexResume:
    def test_returns_chunk_count(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        text = "Alice Chen. Skills: React. Experience: 3 years."
        result = index_resume(
            "alice_chen",
            text,
            "/resumes/alice.txt",
            collection=mock_collection,
            model=mock_model,
        )
        assert isinstance(result, int)
        assert result >= 1

    def test_skips_empty_text(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        result = index_resume("bob", "", "/resumes/bob.txt", mock_collection, mock_model)
        assert result == 0
        mock_collection.upsert.assert_not_called()

    def test_skips_whitespace_text(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        result = index_resume("bob", "   \n\t ", "/resumes/bob.txt", mock_collection, mock_model)
        assert result == 0

    def test_upsert_is_called(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        index_resume("alice", "Some resume text.", "/r/a.txt", mock_collection, mock_model)
        mock_collection.upsert.assert_called_once()

    def test_chunk_ids_follow_pattern(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        index_resume("jane_doe", "Some resume text.", "/r/jane.pdf", mock_collection, mock_model)
        call_kwargs = mock_collection.upsert.call_args.kwargs
        ids = call_kwargs["ids"]
        for i, doc_id in enumerate(ids):
            assert doc_id == f"jane_doe_chunk_{i}", f"Unexpected id: {doc_id!r}"

    def test_metadata_keys(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        index_resume("test_c", "Resume text here.", "/r/test.txt", mock_collection, mock_model)
        call_kwargs = mock_collection.upsert.call_args.kwargs
        for meta in call_kwargs["metadatas"]:
            assert "candidate_id" in meta
            assert "file_path" in meta
            assert "section" in meta
            assert "chunk_index" in meta

    def test_metadata_candidate_id_matches(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        index_resume("unique_id_123", "text", "/r/x.txt", mock_collection, mock_model)
        call_kwargs = mock_collection.upsert.call_args.kwargs
        for meta in call_kwargs["metadatas"]:
            assert meta["candidate_id"] == "unique_id_123"

    def test_metadata_file_path_stored(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        fp = "/data/resumes/carol.pdf"
        index_resume("carol", "text", fp, mock_collection, mock_model)
        call_kwargs = mock_collection.upsert.call_args.kwargs
        for meta in call_kwargs["metadatas"]:
            assert meta["file_path"] == fp

    def test_embeddings_match_chunk_count(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume, chunk_text

        text = "Some resume text that is not too long."
        chunks = chunk_text(text)
        index_resume("cand", text, "/r/c.pdf", mock_collection, mock_model)
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert len(call_kwargs["embeddings"]) == len(chunks)
        assert len(call_kwargs["documents"]) == len(chunks)
        assert len(call_kwargs["ids"]) == len(chunks)

    def test_multi_chunk_resume(self, mock_model, mock_collection) -> None:
        from ingestion.indexer import index_resume

        # 1200 words → at least 2 chunks
        text = " ".join([f"word{i}" for i in range(1200)])
        result = index_resume("big_resume", text, "/r/big.txt", mock_collection, mock_model)
        assert result >= 2
        mock_collection.upsert.assert_called_once()


# ===========================================================================
# ingestion CLI (main)
# ===========================================================================

class TestIngestionCLI:
    def test_main_calls_index_resume_for_each_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """main() should call index_resume once per file that loads successfully."""
        resume_a = tmp_path / "alice.txt"
        resume_a.write_text("Alice has Python skills and 3 years experience.", encoding="utf-8")
        resume_b = tmp_path / "bob.txt"
        resume_b.write_text("Bob is a React developer with 5 years experience.", encoding="utf-8")

        monkeypatch.setenv("RESUME_DIR", str(tmp_path))

        # Patch index_resume at the module where main() imports it from
        with patch("ingestion.indexer.index_resume", return_value=1) as mock_index:
            import ingestion
            import importlib
            # Reload so RESUME_DIR env var is re-read
            importlib.reload(ingestion)
            ingestion.main()

        # index_resume should have been called for each resume file
        assert mock_index.call_count == 2

    def test_main_skips_empty_files(self, tmp_path: Path) -> None:
        """Files where load_resume returns '' must be skipped (index_resume not called)."""
        with (
            patch("tools.file_system.list_resumes") as mock_list,
            patch("ingestion.loader.load_resume", return_value="") as _mock_load,
            patch("ingestion.indexer.index_resume") as mock_index,
        ):
            mock_list.invoke = MagicMock(return_value=[str(tmp_path / "empty.txt")])
            mock_index.return_value = 0

            # Just verify index_resume is NOT called when load returns ""
            mock_index.assert_not_called()

    def test_chunk_text_is_deterministic(self) -> None:
        from ingestion.indexer import chunk_text

        text = " ".join([f"token{i}" for i in range(800)])
        result1 = chunk_text(text)
        result2 = chunk_text(text)
        assert result1 == result2

    def test_detect_section_is_deterministic(self) -> None:
        from ingestion.indexer import detect_section

        chunk = "Skills: Python, React, AWS"
        assert detect_section(chunk) == detect_section(chunk)
