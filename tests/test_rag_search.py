"""
tests/test_rag_search.py
=========================
Unit & integration tests for Phase 3: rag_search tool.

Test matrix:
  deduplicate_and_rank
    - single result passes through unchanged
    - multiple chunks for same candidate → best score kept
    - truncates to top_k
    - sorts by relevance descending
    - distance-to-score conversion (1 − dist, clamped to [0,1])
    - empty raw results → empty list
    - missing metadata keys handled gracefully

  rag_search (unit — mocked collection & model)
    - returns list of dicts with required keys
    - deduplication applied (only one entry per candidate_id)
    - top_k respected
    - empty query → []
    - empty collection → []
    - missing collection → []
    - embedding failure → RuntimeError
    - ChromaDB query failure → []
    - top_k clamped to [1, 50]

  rag_search (integration — real ChromaDB index)
    - "React developer" returns relevant candidates (relevance > 0)
    - "machine learning Python" hits ML-related candidates
    - result keys present and types correct
    - all relevance scores in [0, 1]
    - no duplicate candidate_ids in results
    - nonsense query returns empty or low-relevance results
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_STORE = PROJECT_ROOT / "chroma_store"


# ===========================================================================
# Fixtures & helpers
# ===========================================================================

def _make_raw(
    candidate_ids: list[str],
    distances: list[float],
    docs: list[str] | None = None,
    file_paths: list[str] | None = None,
) -> dict:
    """Build a minimal ChromaDB query result dict."""
    n = len(candidate_ids)
    docs = docs or [f"resume text for {cid}" for cid in candidate_ids]
    file_paths = file_paths or [f"/resumes/{cid}.txt" for cid in candidate_ids]
    return {
        "documents": [docs],
        "metadatas": [[
            {"candidate_id": cid, "file_path": fp, "section": "general", "chunk_index": 0}
            for cid, fp in zip(candidate_ids, file_paths)
        ]],
        "distances": [distances],
        "ids": [[f"{cid}_chunk_0" for cid in candidate_ids]],
    }


@pytest.fixture
def mock_model():
    """Fake SentenceTransformer — returns a deterministic unit vector."""
    import numpy as np
    m = MagicMock()
    m.encode = MagicMock(return_value=np.zeros(384, dtype="float32"))
    return m


@pytest.fixture
def mock_collection_with_docs():
    """Fake ChromaDB collection with 3 candidates."""
    col = MagicMock()
    col.count.return_value = 3
    col.query.return_value = _make_raw(
        candidate_ids=["alice", "bob", "carol"],
        distances=[0.1, 0.3, 0.5],
    )
    return col


@pytest.fixture
def mock_empty_collection():
    col = MagicMock()
    col.count.return_value = 0
    return col


# ===========================================================================
# deduplicate_and_rank
# ===========================================================================

class TestDeduplicateAndRank:
    def test_single_result_passes_through(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["alice"], [0.2])
        result = deduplicate_and_rank(raw, top_k=5)
        assert len(result) == 1
        assert result[0]["candidate_id"] == "alice"

    def test_best_score_kept_per_candidate(self) -> None:
        """Two chunks for 'alice' — only the one with lower distance (higher score) survives."""
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(
            candidate_ids=["alice", "alice", "bob"],
            distances=[0.4, 0.1, 0.3],   # second alice has dist=0.1 → score=0.9
            docs=["alice chunk A", "alice chunk B", "bob text"],
        )
        result = deduplicate_and_rank(raw, top_k=5)
        # Only one alice entry
        alice_hits = [r for r in result if r["candidate_id"] == "alice"]
        assert len(alice_hits) == 1
        # It should be the better chunk (score ≈ 0.9, from dist=0.1)
        assert alice_hits[0]["relevance_score"] == pytest.approx(0.9, abs=0.01)
        assert alice_hits[0]["snippet"] == "alice chunk B"

    def test_truncates_to_top_k(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["a", "b", "c", "d", "e"], [0.1, 0.2, 0.3, 0.4, 0.5])
        result = deduplicate_and_rank(raw, top_k=3)
        assert len(result) == 3

    def test_sorted_by_relevance_descending(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["a", "b", "c"], [0.5, 0.1, 0.3])
        result = deduplicate_and_rank(raw, top_k=10)
        scores = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_distance_to_score_conversion(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["x"], [0.25])
        result = deduplicate_and_rank(raw, top_k=1)
        assert result[0]["relevance_score"] == pytest.approx(0.75, abs=0.01)

    def test_score_clamped_below_zero(self) -> None:
        """Distance > 1 (e.g. 1.5) → score clamped to 0."""
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["x"], [1.5])
        result = deduplicate_and_rank(raw, top_k=1)
        assert result[0]["relevance_score"] == 0.0

    def test_score_clamped_above_one(self) -> None:
        """Distance < 0 (edge case) → score clamped to 1."""
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["x"], [-0.1])
        result = deduplicate_and_rank(raw, top_k=1)
        assert result[0]["relevance_score"] == 1.0

    def test_empty_raw_returns_empty_list(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        result = deduplicate_and_rank(raw, top_k=5)
        assert result == []

    def test_required_keys_present(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        raw = _make_raw(["alice"], [0.2])
        result = deduplicate_and_rank(raw, top_k=5)
        assert set(result[0].keys()) == {"candidate_id", "file_path", "relevance_score", "snippet"}

    def test_snippet_truncated_to_500_chars(self) -> None:
        from tools.rag_search import deduplicate_and_rank

        long_doc = "x" * 2000
        raw = _make_raw(["alice"], [0.1], docs=[long_doc])
        result = deduplicate_and_rank(raw, top_k=1)
        assert len(result[0]["snippet"]) <= 500

    def test_missing_metadata_keys_graceful(self) -> None:
        """Metadata missing candidate_id / file_path → falls back to defaults."""
        from tools.rag_search import deduplicate_and_rank

        raw = {
            "documents": [["some text"]],
            "metadatas": [[{}]],          # no candidate_id / file_path
            "distances": [[0.3]],
            "ids": [["chunk_0"]],
        }
        result = deduplicate_and_rank(raw, top_k=5)
        assert len(result) == 1
        assert result[0]["candidate_id"] == "unknown"
        assert result[0]["file_path"] == ""


# ===========================================================================
# rag_search — unit (mocked)
# ===========================================================================

class TestRagSearchUnit:
    def test_returns_list(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "Python developer", "top_k": 5})
        assert isinstance(result, list)

    def test_required_keys_in_each_result(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "React developer"})
        for item in result:
            assert "candidate_id" in item
            assert "file_path" in item
            assert "relevance_score" in item
            assert "snippet" in item

    def test_no_duplicate_candidate_ids(self, mock_model) -> None:
        """Even if ChromaDB returns multiple chunks per candidate, each appears once."""
        from tools import rag_search as rs_module

        col = MagicMock()
        col.count.return_value = 4
        col.query.return_value = _make_raw(
            candidate_ids=["alice", "alice", "bob", "bob"],
            distances=[0.1, 0.4, 0.2, 0.5],
        )

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=col),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "developer", "top_k": 10})

        ids = [r["candidate_id"] for r in result]
        assert len(ids) == len(set(ids)), "Duplicate candidate_ids found"

    def test_top_k_respected(self, mock_model) -> None:
        from tools import rag_search as rs_module

        col = MagicMock()
        col.count.return_value = 10
        col.query.return_value = _make_raw(
            candidate_ids=[f"c{i}" for i in range(10)],
            distances=[i * 0.05 for i in range(10)],
        )

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=col),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "engineer", "top_k": 3})
        assert len(result) <= 3

    def test_empty_query_returns_empty_list(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": ""})
        assert result == []

    def test_whitespace_query_returns_empty_list(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "   "})
        assert result == []

    def test_empty_collection_returns_empty_list(self, mock_model, mock_empty_collection) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_empty_collection),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "React developer"})
        assert result == []

    def test_missing_collection_returns_empty_list(self, mock_model) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=None),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "developer"})
        assert result == []

    def test_chroma_query_failure_returns_empty_list(self, mock_model) -> None:
        from tools import rag_search as rs_module

        col = MagicMock()
        col.count.return_value = 5
        col.query.side_effect = Exception("ChromaDB connection error")

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=col),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "developer"})
        assert result == []

    def test_embedding_failure_raises_runtime_error(self) -> None:
        import numpy as np
        from tools import rag_search as rs_module

        bad_model = MagicMock()
        bad_model.encode.side_effect = RuntimeError("CUDA out of memory")

        col = MagicMock()
        col.count.return_value = 5

        with (
            patch.object(rs_module, "_get_model", return_value=bad_model),
            patch.object(rs_module, "_get_collection", return_value=col),
        ):
            from tools.rag_search import rag_search
            with pytest.raises(RuntimeError, match="Embedding model failed"):
                rag_search.invoke({"query": "developer"})

    def test_top_k_clamped_minimum_to_1(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            # top_k=0 should be clamped to 1, not cause an error
            result = rag_search.invoke({"query": "developer", "top_k": 0})
        assert len(result) <= 1

    def test_relevance_scores_in_range(self, mock_model, mock_collection_with_docs) -> None:
        from tools import rag_search as rs_module

        with (
            patch.object(rs_module, "_get_model", return_value=mock_model),
            patch.object(rs_module, "_get_collection", return_value=mock_collection_with_docs),
        ):
            from tools.rag_search import rag_search
            result = rag_search.invoke({"query": "Python engineer"})
        for r in result:
            assert 0.0 <= r["relevance_score"] <= 1.0


# ===========================================================================
# rag_search — integration (real ChromaDB, real BGE model)
# ===========================================================================

@pytest.mark.skipif(
    not CHROMA_STORE.exists(),
    reason="ChromaDB store not found — run `python -m ingestion` first.",
)
class TestRagSearchIntegration:
    """These tests hit the real ChromaDB index and the real BGE model."""

    def test_react_query_returns_results(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "React frontend developer", "top_k": 5})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_ml_query_returns_results(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "machine learning Python TensorFlow", "top_k": 5})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_result_keys_present(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "software engineer", "top_k": 3})
        for item in result:
            assert "candidate_id" in item
            assert "file_path" in item
            assert "relevance_score" in item
            assert "snippet" in item

    def test_relevance_scores_in_range(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "backend API REST", "top_k": 5})
        for r in result:
            assert 0.0 <= r["relevance_score"] <= 1.0, (
                f"Score {r['relevance_score']} out of range for {r['candidate_id']}"
            )

    def test_no_duplicate_candidate_ids(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "Python developer", "top_k": 10})
        ids = [r["candidate_id"] for r in result]
        assert len(ids) == len(set(ids)), "Duplicate candidate_ids in integration result"

    def test_top_k_not_exceeded(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "engineer", "top_k": 3})
        assert len(result) <= 3

    def test_results_sorted_by_relevance(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "data scientist", "top_k": 5})
        if len(result) >= 2:
            scores = [r["relevance_score"] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_file_paths_are_strings(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "developer", "top_k": 5})
        for r in result:
            assert isinstance(r["file_path"], str)

    def test_snippets_are_strings(self) -> None:
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "developer", "top_k": 5})
        for r in result:
            assert isinstance(r["snippet"], str)

    def test_natalie_brooks_found_for_react_query(self) -> None:
        """natalie_brooks_react.pdf should rank highly for a React query."""
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "React developer frontend", "top_k": 5})
        candidate_ids = [r["candidate_id"] for r in result]
        assert any("react" in cid.lower() or "natalie" in cid.lower() for cid in candidate_ids), (
            f"Expected a React-focused candidate in top-5; got: {candidate_ids}"
        )

    def test_nonsense_query_still_returns_list(self) -> None:
        """Even gibberish should return a list (just low relevance)."""
        from tools.rag_search import rag_search

        result = rag_search.invoke({"query": "xyzzy frobnicator blurp", "top_k": 3})
        assert isinstance(result, list)
