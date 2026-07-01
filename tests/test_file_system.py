"""
tests/test_file_system.py
==========================
Unit tests for Phase 2.1 & 2.2: read_resume and list_resumes tools.

Test matrix:
  read_resume
    - reads a real .txt resume from the sample data folder
    - reads a real .pdf resume from the sample data folder
    - reads a real .json resume (created in tmp fixture)
    - reads a real .docx resume if python-docx is installed
    - raises FileNotFoundError for a non-existent path
    - raises ValueError for an unsupported extension
    - raises RuntimeError for a corrupted PDF
    - returns non-empty string for all valid formats

  list_resumes
    - returns all supported files in a flat directory
    - walks sub-directories recursively
    - skips hidden files and Word lock files (~$)
    - skips files with unsupported extensions
    - returns paths as absolute strings
    - returns an empty list for an empty directory
    - raises NotADirectoryError for a non-existent path
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate the project root & sample data
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = PROJECT_ROOT / "data" / "resumes"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(filename: str) -> Path:
    """Return path to a file in data/resumes/."""
    return RESUME_DIR / filename


def _has_python_docx() -> bool:
    try:
        import docx  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def txt_resume(tmp_path: Path) -> Path:
    """A minimal plain-text resume written to a temp file."""
    p = tmp_path / "test_candidate.txt"
    p.write_text(
        "Jane Doe\n"
        "Email: jane@example.com\n"
        "Skills: Python, FastAPI, PostgreSQL\n"
        "Experience: 4 years backend development at Acme Corp\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def json_resume(tmp_path: Path) -> Path:
    """A minimal JSON resume written to a temp file."""
    data = {
        "name": "John Smith",
        "email": "john@example.com",
        "skills": ["React", "TypeScript", "Node.js"],
        "experience_years": 5,
        "summary": "Full-stack engineer with 5 years experience.",
    }
    p = tmp_path / "test_candidate.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def json_list_resume(tmp_path: Path) -> Path:
    """A JSON resume where the top-level structure is a list (edge case)."""
    data = [
        {"name": "Alice"},
        {"skills": ["Go", "Kubernetes"]},
    ]
    p = tmp_path / "list_resume.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def malformed_json(tmp_path: Path) -> Path:
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    return p


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    p = tmp_path / "resume.xlsx"
    p.write_bytes(b"\x00\x01\x02")
    return p


@pytest.fixture
def resume_directory(tmp_path: Path) -> Path:
    """A structured directory with mixed files for list_resumes tests."""
    # Supported files
    (tmp_path / "alice.pdf").write_bytes(b"fake-pdf-content")
    (tmp_path / "bob.txt").write_text("Bob resume", encoding="utf-8")
    (tmp_path / "carol.json").write_text('{"name":"Carol"}', encoding="utf-8")

    # Unsupported extensions — should be ignored
    (tmp_path / "notes.md").write_text("some notes", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")

    # Hidden file — should be ignored
    (tmp_path / ".DS_Store").write_bytes(b"\x00")

    # Word temp lock file — should be ignored
    (tmp_path / "~$draft.docx").write_bytes(b"\x00")

    # Sub-directory with more resumes
    sub = tmp_path / "archive"
    sub.mkdir()
    (sub / "old_resume.txt").write_text("Old resume", encoding="utf-8")

    return tmp_path


# ===========================================================================
# read_resume — plain text
# ===========================================================================

class TestReadResumeTxt:
    def test_reads_fixture_txt(self, txt_resume: Path) -> None:
        from tools.file_system import read_resume

        result = read_resume.invoke(str(txt_resume))
        assert "Jane Doe" in result
        assert "Python" in result

    def test_reads_sample_txt(self) -> None:
        """Read one of the sample .txt resumes shipped with the project."""
        sample = _sample("lena_fischer_data_scientist.txt")
        if not sample.exists():
            pytest.skip("Sample TXT resume not found.")

        from tools.file_system import read_resume

        result = read_resume.invoke(str(sample))
        assert isinstance(result, str)
        assert len(result) > 50, "Expected non-trivial resume text"

    def test_returns_string(self, txt_resume: Path) -> None:
        from tools.file_system import read_resume

        result = read_resume.invoke(str(txt_resume))
        assert isinstance(result, str)

    def test_non_empty(self, txt_resume: Path) -> None:
        from tools.file_system import read_resume

        result = read_resume.invoke(str(txt_resume))
        assert result.strip() != ""


# ===========================================================================
# read_resume — JSON
# ===========================================================================

class TestReadResumeJson:
    def test_reads_dict_json(self, json_resume: Path) -> None:
        from tools.file_system import read_resume

        result = read_resume.invoke(str(json_resume))
        # Expect key names to appear as headers
        assert "John Smith" in result
        assert "React" in result

    def test_reads_list_json(self, json_list_resume: Path) -> None:
        from tools.file_system import read_resume

        result = read_resume.invoke(str(json_list_resume))
        assert isinstance(result, str)
        assert "Alice" in result

    def test_malformed_json_raises_runtime_error(self, malformed_json: Path) -> None:
        from tools.file_system import read_resume

        with pytest.raises(RuntimeError, match="Could not parse resume"):
            read_resume.invoke(str(malformed_json))


# ===========================================================================
# read_resume — PDF
# ===========================================================================

class TestReadResumePdf:
    def test_reads_sample_pdf(self) -> None:
        """Read one of the sample PDFs shipped with the project."""
        sample = _sample("alice_chen_frontend.pdf")
        if not sample.exists():
            pytest.skip("Sample PDF not found.")

        from tools.file_system import read_resume

        result = read_resume.invoke(str(sample))
        assert isinstance(result, str)
        # PDFs may be image-only, in which case text is empty — tolerate that
        # but verify that the call succeeds without raising.

    def test_all_sample_pdfs_readable(self) -> None:
        """Every .pdf in data/resumes/ must be readable without raising."""
        if not RESUME_DIR.exists():
            pytest.skip("Sample resume directory not found.")

        from tools.file_system import read_resume

        pdf_files = list(RESUME_DIR.glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF resumes found.")

        for pdf in pdf_files:
            result = read_resume.invoke(str(pdf))
            assert isinstance(result, str), f"Expected str for {pdf.name}"


# ===========================================================================
# read_resume — DOCX
# ===========================================================================

class TestReadResumeDocx:
    @pytest.mark.skipif(
        not _has_python_docx(),
        reason="python-docx not installed",
    )
    def test_reads_sample_docx(self) -> None:
        sample = _sample("carlos_mendez_devops.docx")
        if not sample.exists():
            pytest.skip("Sample DOCX not found.")

        from tools.file_system import read_resume

        result = read_resume.invoke(str(sample))
        assert isinstance(result, str)
        assert len(result) > 10


# ===========================================================================
# read_resume — error cases
# ===========================================================================

class TestReadResumeErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from tools.file_system import read_resume

        with pytest.raises(FileNotFoundError):
            read_resume.invoke(str(tmp_path / "ghost.pdf"))

    def test_unsupported_extension_raises_value_error(
        self, unsupported_file: Path
    ) -> None:
        from tools.file_system import read_resume

        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_resume.invoke(str(unsupported_file))

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        from tools.file_system import read_resume

        # Passing a directory (no extension match) should raise ValueError
        with pytest.raises((ValueError, RuntimeError)):
            read_resume.invoke(str(tmp_path))


# ===========================================================================
# list_resumes — core behaviour
# ===========================================================================

class TestListResumes:
    def test_finds_supported_files(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        filenames = [Path(p).name for p in result]
        assert "alice.pdf" in filenames
        assert "bob.txt" in filenames
        assert "carol.json" in filenames

    def test_returns_absolute_paths(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        for p in result:
            assert Path(p).is_absolute(), f"Expected absolute path, got: {p!r}"

    def test_skips_unsupported_extensions(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        filenames = [Path(p).name for p in result]
        assert "notes.md" not in filenames
        assert "data.csv" not in filenames

    def test_skips_hidden_files(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        filenames = [Path(p).name for p in result]
        assert ".DS_Store" not in filenames

    def test_skips_word_lock_files(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        filenames = [Path(p).name for p in result]
        assert "~$draft.docx" not in filenames

    def test_walks_subdirectories(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        filenames = [Path(p).name for p in result]
        assert "old_resume.txt" in filenames

    def test_returns_sorted_list(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        assert result == sorted(result)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(tmp_path))
        assert result == []

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        from tools.file_system import list_resumes

        with pytest.raises(NotADirectoryError):
            list_resumes.invoke(str(tmp_path / "nonexistent_dir"))

    def test_discovers_all_sample_resumes(self) -> None:
        """list_resumes must discover all sample resumes in data/resumes/."""
        if not RESUME_DIR.exists():
            pytest.skip("Sample resume directory not found.")

        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(RESUME_DIR))
        assert len(result) >= 1, "Expected at least one resume in data/resumes/"
        # All returned paths must point to actual files
        for p in result:
            assert Path(p).is_file(), f"Path is not a file: {p!r}"

    def test_returns_list_type(self, resume_directory: Path) -> None:
        from tools.file_system import list_resumes

        result = list_resumes.invoke(str(resume_directory))
        assert isinstance(result, list)
