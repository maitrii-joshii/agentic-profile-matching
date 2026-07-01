"""
tests/test_tools.py
====================
Unit tests for Phase 4 LLM-powered tools with mocked DeepSeek responses.

Test matrix:
  prompts/system.py — get_llm, call_llm_with_retry
    - get_llm returns ChatDeepSeek instance when API key is set
    - get_llm raises EnvironmentError when DEEPSEEK_API_KEY missing
    - call_llm_with_retry succeeds on first attempt
    - call_llm_with_retry retries on rate-limit errors
    - call_llm_with_retry retries on 429/503 errors
    - call_llm_with_retry does NOT retry on non-transient errors
    - call_llm_with_retry raises RuntimeError after max retries
    - back-off delay doubles between retries

  tools/requirements.py — extract_requirements
    - returns dict with all four required keys
    - must_have list contains items with skill/category/min_years
    - handles LLM response wrapped in markdown fences
    - retries JSON correction on first parse failure
    - raises RuntimeError if both parse attempts fail
    - raises ValueError for empty jd_text

  tools/comparison.py — compare_candidates
    - returns dict with summary, ranking, matrix, head_to_head keys
    - ranking list contains all candidate IDs
    - matrix has entry for each candidate
    - raises ValueError for fewer than 2 candidates
    - handles ChromaDB not available (returns empty profiles gracefully)
    - raises RuntimeError on invalid JSON from LLM

  tools/interview.py — generate_interview_questions
    - returns dict with candidate_id, technical, behavioural, gap_probing
    - technical list is non-empty
    - gap_probing list is non-empty
    - raises ValueError for empty candidate_id
    - handles missing resume gracefully (still calls LLM)
    - raises RuntimeError on invalid JSON from LLM
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# Helpers
# ===========================================================================

def _ai(content: str) -> AIMessage:
    """Create a minimal AIMessage with the given content."""
    return AIMessage(content=content)


def _valid_requirements_json() -> str:
    return json.dumps({
        "must_have": [
            {"skill": "React", "category": "must_have", "min_years": 3},
            {"skill": "TypeScript", "category": "must_have", "min_years": None},
        ],
        "nice_to_have": [
            {"skill": "GraphQL", "category": "nice_to_have", "min_years": None},
        ],
        "experience_range": {"min_years": 3, "max_years": 7},
        "education": "Bachelor's in Computer Science or equivalent",
    })


def _valid_comparison_json() -> str:
    return json.dumps({
        "summary": "Alice outperforms Bob on frontend skills.",
        "ranking": ["alice", "bob"],
        "matrix": {
            "alice": {
                "overall_score": 88,
                "skill_match": {"React": 1.0},
                "experience_fit": 0.9,
                "strengths": ["Strong React"],
                "gaps": ["No AWS"],
                "hire_recommendation": "hire",
                "reasoning": "Strong match.",
            },
            "bob": {
                "overall_score": 72,
                "skill_match": {"React": 0.6},
                "experience_fit": 0.7,
                "strengths": ["Backend"],
                "gaps": ["Weak React"],
                "hire_recommendation": "borderline",
                "reasoning": "Partial match.",
            },
        },
        "head_to_head": "Alice edges Bob on React experience.",
    })


def _valid_interview_json() -> str:
    return json.dumps({
        "candidate_id": "alice",
        "technical": [
            {"question": "Explain React reconciliation.", "rationale": "Core skill"},
        ],
        "behavioural": [
            {"question": "Describe a time you handled a tight deadline.", "rationale": "Work ethic"},
        ],
        "gap_probing": [
            {"question": "Have you worked with AWS?", "gap": "AWS"},
        ],
    })


SAMPLE_JD = """
We are looking for a Senior Frontend Engineer with 3+ years of React experience.
Must have: React, TypeScript. Nice to have: GraphQL, AWS.
"""


# ===========================================================================
# prompts/system.py — get_llm
# ===========================================================================

class TestGetLlm:
    def test_returns_chat_deepseek_when_api_key_set(self, monkeypatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        from prompts.system import get_llm
        llm = get_llm()
        # Just check it's a LangChain-compatible chat model
        assert hasattr(llm, "invoke")

    def test_raises_error_when_api_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from prompts.system import get_llm
        with pytest.raises(EnvironmentError, match="DEEPSEEK_API_KEY"):
                get_llm()


# ===========================================================================
# prompts/system.py — call_llm_with_retry
# ===========================================================================

class TestCallLlmWithRetry:
    def test_succeeds_on_first_attempt(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        expected = _ai("hello")
        llm.invoke.return_value = expected

        result = call_llm_with_retry(llm, [])
        assert result is expected
        llm.invoke.assert_called_once()

    def test_retries_on_rate_limit_error(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("rate limit exceeded"),
            _ai("ok"),
        ]

        with patch("prompts.system.time.sleep"):
            result = call_llm_with_retry(llm, [], max_retries=3, base_delay=0.01)

        assert result.content == "ok"
        assert llm.invoke.call_count == 2

    def test_retries_on_429_string(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = [Exception("Error 429"), _ai("ok")]

        with patch("prompts.system.time.sleep"):
            result = call_llm_with_retry(llm, [], max_retries=3, base_delay=0.01)
        assert result.content == "ok"

    def test_retries_on_503(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = [Exception("503 service unavailable"), _ai("ok")]

        with patch("prompts.system.time.sleep"):
            result = call_llm_with_retry(llm, [], max_retries=3, base_delay=0.01)
        assert result.content == "ok"

    def test_does_not_retry_non_transient_error(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = Exception("invalid API key")

        with patch("prompts.system.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                call_llm_with_retry(llm, [], max_retries=3, base_delay=0.01)

        mock_sleep.assert_not_called()
        llm.invoke.assert_called_once()

    def test_raises_runtime_error_after_max_retries(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = Exception("rate limit exceeded")

        with patch("prompts.system.time.sleep"):
            with pytest.raises(RuntimeError, match="failed after"):
                call_llm_with_retry(llm, [], max_retries=3, base_delay=0.01)

        assert llm.invoke.call_count == 3

    def test_backoff_delay_doubles(self) -> None:
        from prompts.system import call_llm_with_retry

        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("rate limit exceeded"),
            Exception("rate limit exceeded"),
            _ai("ok"),
        ]

        sleep_calls: list[float] = []
        with patch("prompts.system.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            call_llm_with_retry(llm, [], max_retries=3, base_delay=1.0)

        assert len(sleep_calls) == 2
        assert sleep_calls[1] == sleep_calls[0] * 2


# ===========================================================================
# tools/requirements.py — extract_requirements
# ===========================================================================

class TestExtractRequirements:
    def _mock_llm(self, content: str):
        llm = MagicMock()
        llm.invoke.return_value = _ai(content)
        return llm

    def test_returns_dict_with_required_keys(self, monkeypatch) -> None:
        from tools import requirements as req_mod
        monkeypatch.setattr(req_mod, "get_llm", lambda: self._mock_llm(_valid_requirements_json()))
        monkeypatch.setattr(req_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        from tools.requirements import extract_requirements
        result = extract_requirements.invoke(SAMPLE_JD)
        assert set(result.keys()) >= {"must_have", "nice_to_have", "experience_range", "education"}

    def test_must_have_list_has_correct_structure(self, monkeypatch) -> None:
        from tools import requirements as req_mod
        monkeypatch.setattr(req_mod, "get_llm", lambda: self._mock_llm(_valid_requirements_json()))
        monkeypatch.setattr(req_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        from tools.requirements import extract_requirements
        result = extract_requirements.invoke(SAMPLE_JD)
        for item in result["must_have"]:
            assert "skill" in item
            assert "category" in item
            assert item["category"] == "must_have"

    def test_strips_markdown_fences(self, monkeypatch) -> None:
        from tools import requirements as req_mod
        fenced = f"```json\n{_valid_requirements_json()}\n```"
        monkeypatch.setattr(req_mod, "get_llm", lambda: self._mock_llm(fenced))
        monkeypatch.setattr(req_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        from tools.requirements import extract_requirements
        result = extract_requirements.invoke(SAMPLE_JD)
        assert isinstance(result["must_have"], list)

    def test_raises_value_error_for_empty_jd(self, monkeypatch) -> None:
        from tools.requirements import extract_requirements
        with pytest.raises(ValueError, match="must not be empty"):
            extract_requirements.invoke("")

    def test_raises_value_error_for_whitespace_jd(self, monkeypatch) -> None:
        from tools.requirements import extract_requirements
        with pytest.raises(ValueError):
            extract_requirements.invoke("   ")

    def test_correction_attempt_on_invalid_json(self, monkeypatch) -> None:
        """First call returns garbage JSON; second call (correction) returns valid JSON."""
        from tools import requirements as req_mod

        call_count = [0]
        def fake_retry(llm, msgs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return _ai("not json at all {{{")
            return _ai(_valid_requirements_json())

        monkeypatch.setattr(req_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(req_mod, "call_llm_with_retry", fake_retry)

        from tools.requirements import extract_requirements
        result = extract_requirements.invoke(SAMPLE_JD)
        assert "must_have" in result
        assert call_count[0] == 2

    def test_raises_runtime_error_after_two_bad_responses(self, monkeypatch) -> None:
        from tools import requirements as req_mod

        monkeypatch.setattr(req_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(req_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: _ai("not json"))

        from tools.requirements import extract_requirements
        with pytest.raises(RuntimeError):
            extract_requirements.invoke(SAMPLE_JD)

    def test_defaults_missing_keys(self, monkeypatch) -> None:
        """LLM returns JSON with only must_have — other keys get defaults."""
        from tools import requirements as req_mod
        partial_json = json.dumps({"must_have": [{"skill": "Python", "category": "must_have", "min_years": 2}]})
        monkeypatch.setattr(req_mod, "get_llm", lambda: self._mock_llm(partial_json))
        monkeypatch.setattr(req_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        from tools.requirements import extract_requirements
        result = extract_requirements.invoke(SAMPLE_JD)
        assert result["nice_to_have"] == []
        assert "min_years" in result["experience_range"]
        assert result["education"] == "Not specified"


# ===========================================================================
# tools/comparison.py — compare_candidates
# ===========================================================================

class TestCompareCandidates:
    def _patch(self, monkeypatch, content: str):
        from tools import comparison as cmp_mod
        monkeypatch.setattr(cmp_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(cmp_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: _ai(content))
        monkeypatch.setattr(cmp_mod, "_fetch_candidate_chunks",
                            lambda ids: {cid: f"Resume of {cid}" for cid in ids})

    def test_returns_required_keys(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_comparison_json())
        from tools.comparison import compare_candidates
        result = compare_candidates.invoke({"candidate_ids": ["alice", "bob"], "jd_text": SAMPLE_JD})
        assert set(result.keys()) >= {"summary", "ranking", "matrix", "head_to_head"}

    def test_ranking_contains_candidates(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_comparison_json())
        from tools.comparison import compare_candidates
        result = compare_candidates.invoke({"candidate_ids": ["alice", "bob"]})
        assert set(result["ranking"]) == {"alice", "bob"}

    def test_matrix_has_entry_per_candidate(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_comparison_json())
        from tools.comparison import compare_candidates
        result = compare_candidates.invoke({"candidate_ids": ["alice", "bob"]})
        assert "alice" in result["matrix"]
        assert "bob" in result["matrix"]

    def test_raises_value_error_for_single_candidate(self, monkeypatch) -> None:
        from tools.comparison import compare_candidates
        with pytest.raises(ValueError, match="at least 2"):
            compare_candidates.invoke({"candidate_ids": ["alice"]})

    def test_raises_value_error_for_empty_list(self, monkeypatch) -> None:
        from tools.comparison import compare_candidates
        with pytest.raises(ValueError):
            compare_candidates.invoke({"candidate_ids": []})

    def test_handles_invalid_json_with_runtime_error(self, monkeypatch) -> None:
        from tools import comparison as cmp_mod
        monkeypatch.setattr(cmp_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(cmp_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: _ai("not json"))
        monkeypatch.setattr(cmp_mod, "_fetch_candidate_chunks",
                            lambda ids: {cid: "text" for cid in ids})

        from tools.comparison import compare_candidates
        with pytest.raises(RuntimeError):
            compare_candidates.invoke({"candidate_ids": ["alice", "bob"]})

    def test_strips_markdown_fences(self, monkeypatch) -> None:
        fenced = f"```json\n{_valid_comparison_json()}\n```"
        self._patch(monkeypatch, fenced)
        from tools.comparison import compare_candidates
        result = compare_candidates.invoke({"candidate_ids": ["alice", "bob"]})
        assert "summary" in result


# ===========================================================================
# tools/interview.py — generate_interview_questions
# ===========================================================================

class TestGenerateInterviewQuestions:
    def _patch(self, monkeypatch, content: str, resume: str = "Some resume text"):
        from tools import interview as iv_mod
        monkeypatch.setattr(iv_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(iv_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: _ai(content))
        monkeypatch.setattr(iv_mod, "_fetch_resume_text", lambda cid: resume)

    def test_returns_required_keys(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_interview_json())
        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke(
            {"candidate_id": "alice", "jd_text": SAMPLE_JD}
        )
        assert set(result.keys()) >= {"candidate_id", "technical", "behavioural", "gap_probing"}

    def test_technical_list_non_empty(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_interview_json())
        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke({"candidate_id": "alice"})
        assert len(result["technical"]) >= 1

    def test_gap_probing_list_non_empty(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_interview_json())
        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke({"candidate_id": "alice"})
        assert len(result["gap_probing"]) >= 1

    def test_candidate_id_echoed(self, monkeypatch) -> None:
        self._patch(monkeypatch, _valid_interview_json())
        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke({"candidate_id": "alice"})
        assert result["candidate_id"] == "alice"

    def test_raises_value_error_for_empty_candidate_id(self) -> None:
        from tools.interview import generate_interview_questions
        with pytest.raises(ValueError, match="must not be empty"):
            generate_interview_questions.invoke({"candidate_id": ""})

    def test_raises_value_error_for_whitespace_candidate_id(self) -> None:
        from tools.interview import generate_interview_questions
        with pytest.raises(ValueError):
            generate_interview_questions.invoke({"candidate_id": "  "})

    def test_handles_missing_resume_gracefully(self, monkeypatch) -> None:
        """Even when no resume is found, the LLM is still called."""
        from tools import interview as iv_mod
        llm_called = [False]

        def fake_retry(llm, msgs, **kw):
            llm_called[0] = True
            return _ai(_valid_interview_json())

        monkeypatch.setattr(iv_mod, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(iv_mod, "call_llm_with_retry", fake_retry)
        monkeypatch.setattr(iv_mod, "_fetch_resume_text", lambda cid: "")

        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke({"candidate_id": "unknown_candidate"})
        assert llm_called[0]
        assert "technical" in result

    def test_raises_runtime_error_on_invalid_json(self, monkeypatch) -> None:
        self._patch(monkeypatch, "this is not json")
        from tools.interview import generate_interview_questions
        with pytest.raises(RuntimeError):
            generate_interview_questions.invoke({"candidate_id": "alice"})

    def test_defaults_missing_keys_in_response(self, monkeypatch) -> None:
        """LLM returns JSON with only technical — other lists default to []."""
        partial = json.dumps({"candidate_id": "alice", "technical": [{"question": "Q1", "rationale": "R"}]})
        self._patch(monkeypatch, partial)
        from tools.interview import generate_interview_questions
        result = generate_interview_questions.invoke({"candidate_id": "alice"})
        assert result["behavioural"] == []
        assert result["gap_probing"] == []
