"""
tests/test_explainability.py
==============================
Test Scenario 4 — Phase 8.6 / 8.7:
  "Why X over Y?" returns valid reasoning from candidate_shortlist and round_history.

Validates:
  - Explain intent is detected from "why", "explain", "reason" phrases
  - Response includes candidate name, score, reasoning, strengths, gaps
  - Response pulls from round_history when available
  - Fallback to top-2 candidates when no specific candidate is named
  - Specific candidate name matching in the explain query
  - Borderline candidates receive improvement suggestions in the report
  - Human feedback stays in loop after explain (needs_human_feedback=True)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from state import AgentState


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_candidate(cid: str, name: str, score: float = 80.0) -> dict:
    return {
        "candidate_id": cid,
        "name": name,
        "overall_score": score,
        "skill_match": {"React": 0.9, "TypeScript": 0.6},
        "experience_fit": 0.75,
        "strengths": [f"Strong {name} React expertise", "Fast learner"],
        "gaps": ["No Kubernetes experience", "Limited TypeScript"],
        "reasoning": (
            f"{name} ranked this way because they demonstrate {score:.0f}/100 match. "
            f"Their React proficiency is excellent but TypeScript exposure is limited."
        ),
    }


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "raw_jd": "Senior React Developer",
        "parsed_requirements": [{"skill": "React", "category": "must_have", "min_years": 3}],
        "candidate_shortlist": [
            _make_candidate("alice_001", "Alice Johnson", 92.0),
            _make_candidate("bob_002", "Bob Smith", 74.0),
            _make_candidate("carol_003", "Carol Davis", 55.0),
        ],
        "comparison_results": None,
        "current_round": 4,
        "round_history": [
            {"round": 1, "candidates": [
                {"candidate_id": "alice_001", "overall_score": 88.0},
                {"candidate_id": "bob_002", "overall_score": 70.0},
                {"candidate_id": "carol_003", "overall_score": 52.0},
            ]},
            {"round": 2, "candidates": [
                {"candidate_id": "alice_001", "overall_score": 91.0},
                {"candidate_id": "bob_002", "overall_score": 74.0},
                {"candidate_id": "carol_003", "overall_score": 55.0},
            ]},
            {"round": 3, "candidates": [
                {"candidate_id": "alice_001", "overall_score": 92.0},
                {"candidate_id": "bob_002", "overall_score": 74.0},
                {"candidate_id": "carol_003", "overall_score": 55.0},
            ]},
        ],
        "final_recommendations": None,
        "needs_human_feedback": True,
        "refinement_requested": False,
    }
    base.update(overrides)
    return base


# ── Intent Detection ────────────────────────────────────────────────────

class TestExplainIntentDetection:
    """The explain intent is correctly routed in the human_feedback node."""

    def test_explain_intent_routes_correctly(self):
        """When LLM returns explain intent, node responds with explanation."""
        from unittest.mock import patch
        from nodes import human_feedback as hf

        state = _make_state(messages=[HumanMessage(content="Why is Alice ranked first?")])

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", "alice")):
            result = hf.run(state)

        msgs = result.get("messages", [])
        assert len(msgs) == 1
        assert "Alice Johnson" in msgs[0].content
        assert result.get("needs_human_feedback") is True

    def test_approve_intent_routes_correctly(self):
        """When LLM returns approved, refinement_requested is cleared."""
        from unittest.mock import patch
        from nodes import human_feedback as hf

        state = _make_state(messages=[HumanMessage(content="Looks good, approve")])

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("approved", None)):
            result = hf.run(state)

        assert result.get("needs_human_feedback") is False

    def test_refinement_intent_sets_flag(self):
        """When LLM returns refinement, refinement_requested flag is set."""
        from unittest.mock import patch
        from nodes import human_feedback as hf

        state = _make_state(messages=[HumanMessage(content="Also require TypeScript")])

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("refinement", None)):
            result = hf.run(state)

        assert result.get("refinement_requested") is True

    def test_unknown_intent_stays_in_loop(self):
        """When LLM returns unknown, feedback loop is maintained."""
        from unittest.mock import patch
        from nodes import human_feedback as hf

        state = _make_state(messages=[HumanMessage(content="hello")])

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("unknown", None)):
            result = hf.run(state)

        assert result.get("needs_human_feedback") is True


# ── Explain Response Content ────────────────────────────────────────────────────

class TestExplainResponseContent:
    """_build_explain_response should populate the response with candidate data."""

    def test_response_includes_candidate_name(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("why is alice ranked first", state)
        assert "Alice Johnson" in response

    def test_response_includes_score(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("why is alice ranked first", state)
        assert "92" in response

    def test_response_includes_reasoning(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("why is alice ranked first", state)
        assert "React proficiency" in response or "92/100" in response

    def test_response_includes_strengths(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("explain alice", state)
        assert "Strengths" in response or "React expertise" in response

    def test_response_includes_gaps(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("explain alice", state)
        assert "Gaps" in response or "Kubernetes" in response or "TypeScript" in response

    def test_response_includes_round_history_info(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("why is alice ranked first", state)
        assert "3 round" in response or "round" in response.lower()

    def test_response_falls_back_to_top_2_when_no_name_matched(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        # Query doesn't mention any specific candidate name
        response = _build_explain_response("why is the top candidate ranked first?", state)
        # Should include at least one of the top candidates
        assert "Alice Johnson" in response or "Bob Smith" in response

    def test_specific_candidate_name_match(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("why is Bob ranked second?", state)
        assert "Bob Smith" in response

    def test_candidate_id_also_matched(self):
        from nodes.human_feedback import _build_explain_response
        state = _make_state()
        response = _build_explain_response("explain alice 001", state)
        # alice_001 id should match "alice" in text
        assert "Alice Johnson" in response


# ── Human Feedback Node Explainability Integration ──────────────────────────────

class TestHumanFeedbackNodeExplainability:
    """human_feedback.run() should correctly handle explain intent end-to-end."""

    def test_explain_appends_ai_message(self):
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = _make_state(messages=[
            HumanMessage(content="Why is Alice ranked first?")
        ])
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", "alice")):
            result = run(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_explain_ai_message_has_content(self):
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = _make_state(messages=[
            HumanMessage(content="Can you explain why Alice is first?")
        ])
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", "alice")):
            result = run(state)
        content = result["messages"][0].content
        assert len(content) > 20

    def test_explain_stays_in_feedback_loop(self):
        """After explain, needs_human_feedback must remain True."""
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = _make_state(messages=[
            HumanMessage(content="Explain the rankings please")
        ])
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", None)):
            result = run(state)
        assert result.get("needs_human_feedback") is True

    def test_explain_does_not_set_refinement(self):
        """Explain intent must not accidentally trigger refinement."""
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = _make_state(messages=[
            HumanMessage(content="Why did alice score higher than bob?")
        ])
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", "alice")):
            result = run(state)
        assert not result.get("refinement_requested", False)

    def test_explain_with_no_shortlist_returns_gracefully(self):
        """Explain query with empty shortlist should not crash."""
        from unittest.mock import patch
        from nodes.human_feedback import run
        state = _make_state(
            candidate_shortlist=[],
            messages=[HumanMessage(content="Why is alice ranked first?")]
        )
        with patch("nodes.human_feedback._detect_intent_llm", return_value=("explain", None)):
            result = run(state)
        # Should still produce a message (even if empty of candidate data)
        assert "messages" in result or result.get("needs_human_feedback") is True


# ── Borderline Improvement Suggestions ─────────────────────────────────────────

class TestBorderlineImprovementSuggestions:
    """8.5 — Borderline candidates should receive improvement suggestions."""

    def test_borderline_candidate_gets_suggestions(self):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("border_c", "Border Case", score=60.0)  # 55 <= 60 < 70
        rec = _make_recommendation(candidate)
        assert rec["hire_recommendation"] == "borderline"
        assert len(rec["improvement_suggestions"]) > 0

    def test_improvement_suggestions_reference_gaps(self):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("border_c", "Border Case", score=62.0)
        candidate["gaps"] = ["No TypeScript", "Limited Kubernetes", "No Docker"]
        rec = _make_recommendation(candidate)
        # Each suggestion should mention one of the gaps
        for suggestion in rec["improvement_suggestions"]:
            assert any(gap in suggestion for gap in candidate["gaps"])

    def test_strong_hire_no_suggestions(self):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("top_c", "Top Candidate", score=90.0)
        rec = _make_recommendation(candidate)
        assert rec["hire_recommendation"] == "strong_hire"
        assert rec["improvement_suggestions"] == []

    def test_no_hire_no_suggestions(self):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("bottom_c", "Bottom Candidate", score=40.0)
        rec = _make_recommendation(candidate)
        assert rec["hire_recommendation"] == "no_hire"
        assert rec["improvement_suggestions"] == []

    def test_hire_no_suggestions(self):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("good_c", "Good Candidate", score=75.0)
        rec = _make_recommendation(candidate)
        assert rec["hire_recommendation"] == "hire"
        assert rec["improvement_suggestions"] == []

    @pytest.mark.parametrize("score,expected", [
        (95.0, "strong_hire"),
        (85.0, "strong_hire"),
        (80.0, "hire"),
        (70.0, "hire"),
        (65.0, "borderline"),
        (55.0, "borderline"),
        (50.0, "no_hire"),
        (0.0,  "no_hire"),
    ])
    def test_hire_thresholds(self, score: float, expected: str):
        from nodes.generate_report import _make_recommendation
        candidate = _make_candidate("c", "Candidate", score=score)
        rec = _make_recommendation(candidate)
        assert rec["hire_recommendation"] == expected, (
            f"Expected {expected!r} for score {score}, got {rec['hire_recommendation']!r}"
        )


# ── Interview Questions Intent (Phase 10+) ─────────────────────────────────────

class TestInterviewQuestionsIntent:
    """Interview questions intent is detected and invokes the tool."""

    def test_interview_questions_returns_formatted_response(self):
        from unittest.mock import MagicMock, patch
        from nodes import human_feedback as hf

        state = _make_state(messages=[HumanMessage(content="Generate interview questions for Alice")])

        mock_result = {
            "candidate_id": "alice_001",
            "technical": [{"question": "Explain React hooks.", "rationale": "Core React skill."}],
            "behavioural": [{"question": "Tell me about a time you led a team.", "rationale": "Leadership."}],
            "gap_probing": [{"question": "How would you learn Kubernetes?", "gap": "Kubernetes"}],
        }

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("interview_questions", "alice")), \
             patch("tools.interview.generate_interview_questions") as mock_tool:
            mock_tool.invoke.return_value = mock_result
            result = hf.run(state)

        msgs = result.get("messages", [])
        assert len(msgs) == 1
        content = msgs[0].content
        assert "Interview Questions" in content
        assert "React hooks" in content
        assert result.get("needs_human_feedback") is True

    def test_interview_questions_without_candidate_name_prompts_user(self):
        from unittest.mock import patch
        from nodes import human_feedback as hf

        state = _make_state(
            candidate_shortlist=[],
            messages=[HumanMessage(content="Generate interview questions")]
        )

        with patch("nodes.human_feedback._detect_intent_llm", return_value=("interview_questions", None)):
            result = hf.run(state)

        msgs = result.get("messages", [])
        assert len(msgs) == 1
        assert "mention" in msgs[0].content.lower() or "candidate" in msgs[0].content.lower()
        assert result.get("needs_human_feedback") is True
