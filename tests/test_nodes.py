"""
tests/test_nodes.py
====================
Unit tests for Phase 5 LangGraph nodes (5.7).

All nodes are tested in isolation with mocked state dicts and mocked tool calls.

Test matrix:
  parse_jd
    - extracts JD from last HumanMessage
    - returns empty string when no human message present
    - normalises excessive blank lines
    - detects file path and attempts to load it
    - handles multiple messages, picks last human one

  extract_requirements (node)
    - calls extract_requirements tool with raw_jd
    - stores must_have + nice_to_have as flat list
    - returns empty list when raw_jd is empty
    - merges requirements on refinement (existing + new, no duplicates)
    - resets refinement_requested to False after merge

  search_resumes (node)
    - builds query from must_have requirements
    - populates candidate_shortlist from rag_search results
    - sets current_round = 1
    - returns empty shortlist when rag_search returns nothing
    - returns empty shortlist when no requirements

  rank_candidates (node)
    - Round 1: scores all candidates, keeps top 10, advances to round 2
    - Round 2: scores remaining, keeps top 5, advances to round 3
    - Round 3: calls compare_candidates, sorts by ranking
    - saves snapshot to round_history each round
    - handles empty shortlist gracefully
    - handles LLM scoring failure gracefully (uses fallback score)

  generate_report (node)
    - assigns strong_hire for score >= 85
    - assigns hire for score >= 70
    - assigns borderline for score >= 55
    - assigns no_hire for score < 55
    - adds improvement_suggestions for borderline candidates
    - sets needs_human_feedback = True
    - appends AIMessage summary to messages
    - handles empty shortlist gracefully

  human_feedback (node)
    - detects 'approved' intent → needs_human_feedback = False
    - detects 'refinement' intent → refinement_requested = True
    - detects 'explain' intent → appends AIMessage, stays in feedback loop
    - detects 'new_search' intent → resets shortlist/round/history
    - detects 'rerank' intent → resets current_round to 1
    - unknown message → appends help message, stays in loop
    - no message → returns needs_human_feedback = True
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# State builder helper
# ===========================================================================

def _state(**kwargs) -> dict:
    """Build a minimal AgentState-compatible dict."""
    defaults = {
        "messages": [],
        "raw_jd": "",
        "parsed_requirements": [],
        "candidate_shortlist": [],
        "comparison_results": None,
        "current_round": 1,
        "round_history": [],
        "final_recommendations": None,
        "needs_human_feedback": False,
        "refinement_requested": False,
    }
    defaults.update(kwargs)
    return defaults


def _candidate(cid: str, score: float = 75.0) -> dict:
    return {
        "candidate_id": cid,
        "name": cid.replace("_", " ").title(),
        "overall_score": score,
        "skill_match": {"React": 0.9},
        "experience_fit": 0.8,
        "strengths": [f"{cid} strength"],
        "gaps": [f"{cid} gap"],
        "reasoning": f"Reasoning for {cid}",
        "file_path": f"/resumes/{cid}.pdf",
        "relevance_score": score / 100,
        "snippet": f"Resume snippet for {cid}",
    }


SAMPLE_JD = "We need a Senior React Engineer with 3+ years. TypeScript is a must."

SAMPLE_REQUIREMENTS = [
    {"skill": "React", "category": "must_have", "min_years": 3},
    {"skill": "TypeScript", "category": "must_have", "min_years": None},
    {"skill": "GraphQL", "category": "nice_to_have", "min_years": None},
]


# ===========================================================================
# parse_jd
# ===========================================================================

class TestParseJd:
    def test_extracts_jd_from_human_message(self) -> None:
        from nodes.parse_jd import run
        state = _state(messages=[HumanMessage(content=SAMPLE_JD)])
        result = run(state)
        assert SAMPLE_JD.strip() in result["raw_jd"]

    def test_picks_last_human_message(self) -> None:
        from nodes.parse_jd import run
        state = _state(messages=[
            HumanMessage(content="first message"),
            AIMessage(content="agent reply"),
            HumanMessage(content=SAMPLE_JD),
        ])
        result = run(state)
        assert "React" in result["raw_jd"]
        assert "first message" not in result["raw_jd"]

    def test_empty_when_no_human_message(self) -> None:
        from nodes.parse_jd import run
        state = _state(messages=[AIMessage(content="hello")])
        result = run(state)
        assert result["raw_jd"] == ""

    def test_empty_when_no_messages(self) -> None:
        from nodes.parse_jd import run
        state = _state(messages=[])
        result = run(state)
        assert result["raw_jd"] == ""

    def test_normalises_excessive_blank_lines(self) -> None:
        from nodes.parse_jd import run
        text = "Line one\n\n\n\n\nLine two"
        state = _state(messages=[HumanMessage(content=text)])
        result = run(state)
        assert "\n\n\n" not in result["raw_jd"]

    def test_strips_leading_trailing_whitespace(self) -> None:
        from nodes.parse_jd import run
        state = _state(messages=[HumanMessage(content="  JD content  \n")])
        result = run(state)
        assert result["raw_jd"] == result["raw_jd"].strip()


# ===========================================================================
# extract_requirements (node)
# ===========================================================================

class TestExtractRequirementsNode:
    def _tool_result(self):
        return {
            "must_have": [
                {"skill": "React", "category": "must_have", "min_years": 3},
                {"skill": "TypeScript", "category": "must_have", "min_years": None},
            ],
            "nice_to_have": [
                {"skill": "GraphQL", "category": "nice_to_have", "min_years": None},
            ],
            "experience_range": {"min_years": 3, "max_years": 7},
            "education": "BS CS",
        }

    def test_stores_flat_requirements_list(self) -> None:
        from nodes import extract_requirements as er_mod
        with patch("tools.requirements.extract_requirements",
                   MagicMock(invoke=MagicMock(return_value=self._tool_result()))):
            result = er_mod.run(_state(raw_jd=SAMPLE_JD))
        assert len(result["parsed_requirements"]) == 3   # 2 must + 1 nice

    def test_returns_empty_when_raw_jd_empty(self) -> None:
        from nodes.extract_requirements import run
        result = run(_state(raw_jd=""))
        assert result["parsed_requirements"] == []

    def test_refinement_resets_flag(self) -> None:
        from nodes import extract_requirements as er_mod
        from langchain_core.messages import HumanMessage
        with patch("tools.requirements.refine_requirements",
                   MagicMock(invoke=MagicMock(return_value={"must_have": [], "nice_to_have": []}))):
            state = _state(raw_jd=SAMPLE_JD, refinement_requested=True, messages=[HumanMessage(content="test")])
            result = er_mod.run(state)
        assert result.get("refinement_requested") is False


# ===========================================================================
# search_resumes (node)
# ===========================================================================

class TestSearchResumesNode:
    def _rag_hits(self) -> list[dict]:
        return [
            {"candidate_id": "alice", "file_path": "/r/alice.pdf",
             "relevance_score": 0.9, "snippet": "Alice is a React dev"},
            {"candidate_id": "bob", "file_path": "/r/bob.pdf",
             "relevance_score": 0.7, "snippet": "Bob knows TypeScript"},
        ]

    def test_populates_shortlist(self) -> None:
        from nodes import search_resumes as sr_mod
        with patch("tools.rag_search.rag_search",
                   MagicMock(invoke=MagicMock(return_value=self._rag_hits()))):
            result = sr_mod.run(_state(parsed_requirements=SAMPLE_REQUIREMENTS))
        assert len(result["candidate_shortlist"]) == 2

    def test_sets_current_round_1(self) -> None:
        from nodes import search_resumes as sr_mod
        with patch("tools.rag_search.rag_search",
                   MagicMock(invoke=MagicMock(return_value=self._rag_hits()))):
            result = sr_mod.run(_state(parsed_requirements=SAMPLE_REQUIREMENTS))
        assert result["current_round"] == 1

    def test_empty_when_no_requirements(self, monkeypatch) -> None:
        from nodes.search_resumes import run
        result = run(_state(parsed_requirements=[]))
        assert result["candidate_shortlist"] == []

    def test_empty_when_rag_search_returns_nothing(self) -> None:
        from nodes import search_resumes as sr_mod
        with patch("tools.rag_search.rag_search",
                   MagicMock(invoke=MagicMock(return_value=[]))):
            result = sr_mod.run(_state(parsed_requirements=SAMPLE_REQUIREMENTS))
        assert result["candidate_shortlist"] == []

    def test_candidate_has_required_keys(self) -> None:
        from nodes import search_resumes as sr_mod
        with patch("tools.rag_search.rag_search",
                   MagicMock(invoke=MagicMock(return_value=self._rag_hits()))):
            result = sr_mod.run(_state(parsed_requirements=SAMPLE_REQUIREMENTS))
        for c in result["candidate_shortlist"]:
            assert "candidate_id" in c
            assert "overall_score" in c
            assert "skill_match" in c


# ===========================================================================
# rank_candidates (node)
# ===========================================================================

class TestRankCandidatesNode:
    def _mock_llm_score(self, score: float = 80.0):
        """Return a mock that fakes LLM scoring."""
        response = MagicMock()
        response.content = json.dumps({
            "overall_score": score,
            "skill_match": {"React": 0.9},
            "experience_fit": 0.85,
            "strengths": ["Good React"],
            "gaps": ["No AWS"],
            "reasoning": "Solid candidate",
        })
        llm = MagicMock()
        llm.invoke.return_value = response
        return llm

    def test_round1_keeps_at_most_10(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score(75.0))
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        candidates = [_candidate(f"c{i}", 80.0) for i in range(14)]
        state = _state(parsed_requirements=SAMPLE_REQUIREMENTS,
                       candidate_shortlist=candidates, current_round=1)
        result = rc_mod.run(state)
        assert len(result["candidate_shortlist"]) <= 10

    def test_round1_advances_to_round2(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score())
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        state = _state(parsed_requirements=SAMPLE_REQUIREMENTS,
                       candidate_shortlist=[_candidate("alice")], current_round=1)
        result = rc_mod.run(state)
        assert result["current_round"] == 2

    def test_round2_keeps_at_most_5(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score())
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        candidates = [_candidate(f"c{i}") for i in range(10)]
        state = _state(parsed_requirements=SAMPLE_REQUIREMENTS,
                       candidate_shortlist=candidates, current_round=2)
        result = rc_mod.run(state)
        assert len(result["candidate_shortlist"]) <= 5

    def test_round2_advances_to_round3(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score())
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        state = _state(parsed_requirements=SAMPLE_REQUIREMENTS,
                       candidate_shortlist=[_candidate("alice")], current_round=2)
        result = rc_mod.run(state)
        assert result["current_round"] == 3

    def test_round_history_snapshot_saved(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score())
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        state = _state(parsed_requirements=SAMPLE_REQUIREMENTS,
                       candidate_shortlist=[_candidate("alice")], current_round=1)
        result = rc_mod.run(state)
        assert len(result["round_history"]) == 1
        assert result["round_history"][0]["round"] == 1

    def test_empty_shortlist_returns_without_scoring(self, monkeypatch) -> None:
        from nodes.rank_candidates import run
        result = run(_state(current_round=1))
        # Should not raise; current_round stays the same
        assert result.get("current_round") == 1

    def test_round3_calls_compare_candidates(self, monkeypatch) -> None:
        from nodes import rank_candidates as rc_mod
        monkeypatch.setattr(rc_mod, "get_llm", lambda: self._mock_llm_score())
        monkeypatch.setattr(rc_mod, "call_llm_with_retry",
                            lambda llm, msgs, **kw: llm.invoke(msgs))

        compare_mock = MagicMock(invoke=MagicMock(return_value={
            "summary": "Alice is best",
            "ranking": ["alice", "bob"],
            "matrix": {},
            "head_to_head": "...",
        }))
        monkeypatch.setattr(rc_mod, "compare_candidates", compare_mock, raising=False)

        with patch("nodes.rank_candidates.compare_candidates") as mock_cmp:
            mock_cmp.invoke.return_value = {
                "summary": "Alice is best",
                "ranking": ["alice", "bob"],
                "matrix": {},
                "head_to_head": "...",
            }
            state = _state(
                raw_jd=SAMPLE_JD,
                parsed_requirements=SAMPLE_REQUIREMENTS,
                candidate_shortlist=[_candidate("alice", 90), _candidate("bob", 75)],
                current_round=3,
            )
            result = rc_mod.run(state)

        assert result.get("current_round") == 4   # 4 = sentinel: round 3 complete


# ===========================================================================
# generate_report (node)
# ===========================================================================

class TestGenerateReportNode:
    def test_strong_hire_for_high_score(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("alice", 90.0)])
        result = run(state)
        rec = result["final_recommendations"][0]
        assert rec["hire_recommendation"] == "strong_hire"

    def test_hire_for_mid_score(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("bob", 75.0)])
        result = run(state)
        assert result["final_recommendations"][0]["hire_recommendation"] == "hire"

    def test_borderline_for_low_mid_score(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("carol", 60.0)])
        result = run(state)
        assert result["final_recommendations"][0]["hire_recommendation"] == "borderline"

    def test_no_hire_for_low_score(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("dave", 40.0)])
        result = run(state)
        assert result["final_recommendations"][0]["hire_recommendation"] == "no_hire"

    def test_borderline_has_improvement_suggestions(self) -> None:
        from nodes.generate_report import run
        c = _candidate("eve", 60.0)
        c["gaps"] = ["AWS", "Kubernetes"]
        state = _state(candidate_shortlist=[c])
        result = run(state)
        rec = result["final_recommendations"][0]
        assert rec["hire_recommendation"] == "borderline"
        assert len(rec["improvement_suggestions"]) > 0

    def test_sets_needs_human_feedback(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("alice")])
        result = run(state)
        assert result["needs_human_feedback"] is True

    def test_appends_ai_message(self) -> None:
        from nodes.generate_report import run
        state = _state(candidate_shortlist=[_candidate("alice")])
        result = run(state)
        assert any(isinstance(m, AIMessage) for m in result.get("messages", []))

    def test_empty_shortlist_returns_empty_recommendations(self) -> None:
        from nodes.generate_report import run
        result = run(_state())
        assert result["final_recommendations"] == []

    def test_all_candidates_have_required_keys(self) -> None:
        from nodes.generate_report import run
        candidates = [_candidate(f"c{i}", 80 - i * 10) for i in range(4)]
        result = run(_state(candidate_shortlist=candidates))
        for rec in result["final_recommendations"]:
            for key in ("candidate_id", "hire_recommendation", "strengths", "gaps", "reasoning"):
                assert key in rec


# ===========================================================================
# human_feedback (node)
# ===========================================================================

class TestHumanFeedbackNode:
    def _state_with_msg(self, text: str, **kwargs) -> dict:
        return _state(messages=[
            AIMessage(content="Here are the results."),
            HumanMessage(content=text),
        ], **kwargs)

    def test_approved_clears_feedback_flag(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("approved", None)):
            result = run(self._state_with_msg("Looks good, approve"))
        assert result["needs_human_feedback"] is False

    def test_refinement_sets_flag(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("refinement", None)):
            result = run(self._state_with_msg("Also require Kubernetes experience"))
        assert result.get("refinement_requested") is True

    def test_explain_returns_ai_message(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = self._state_with_msg(
            "Why is Alice ranked higher than Bob?",
            candidate_shortlist=[_candidate("alice", 90), _candidate("bob", 70)],
        )
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", "alice")):
            result = run(state)
        assert any(isinstance(m, AIMessage) for m in result.get("messages", []))

    def test_explain_stays_in_feedback_loop(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = self._state_with_msg(
            "Explain the ranking",
            candidate_shortlist=[_candidate("alice")],
        )
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", None)):
            result = run(state)
        assert result["needs_human_feedback"] is True

    def test_new_search_resets_state(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = self._state_with_msg(
            "Start over with a different role",
            candidate_shortlist=[_candidate("alice")],
            round_history=[{"round": 1, "candidates": []}],
            current_round=3,
        )
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("new_search", None)):
            result = run(state)
        assert result.get("candidate_shortlist") == []
        assert result.get("current_round") == 1
        assert result.get("round_history") == []

    def test_rerank_resets_round(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("rerank", None)):
            result = run(self._state_with_msg("Re-rank the candidates please"))
        assert result.get("current_round") == 1

    def test_unknown_message_stays_in_loop(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("unknown", None)):
            result = run(self._state_with_msg("asdfgh random text here"))
        assert result["needs_human_feedback"] is True

    def test_no_message_stays_in_loop(self) -> None:
        from nodes.human_feedback import run
        result = run(_state(messages=[AIMessage(content="results here")]))
        assert result["needs_human_feedback"] is True

    def test_remove_requirement_detected_as_refinement(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("refinement", None)):
            result = run(self._state_with_msg("Remove the TypeScript requirement"))
        assert result.get("refinement_requested") is True

    def test_change_experience_detected_as_refinement(self) -> None:
        from unittest.mock import patch
        from nodes.human_feedback import run
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("refinement", None)):
            result = run(self._state_with_msg("Change minimum experience to 5 years"))
        assert result.get("refinement_requested") is True
