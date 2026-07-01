"""
tests/test_state.py
====================
Unit tests for Phase 1 state models: Requirement, CandidateScore, AgentState.

Covers:
  - Field presence & annotations match architecture §3
  - Valid object construction and dict-access
  - Type-level constraints (Literal, Optional)
  - JSON serialisability
  - add_messages reducer behaviour (messages are appended, not overwritten)
"""

import json
import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from state import AgentState, CandidateScore, Requirement


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_requirement() -> Requirement:
    return {
        "skill": "React",
        "category": "must_have",
        "min_years": 3,
    }


@pytest.fixture
def valid_requirement_no_years() -> Requirement:
    return {
        "skill": "GraphQL",
        "category": "nice_to_have",
        "min_years": None,
    }


@pytest.fixture
def valid_candidate_score() -> CandidateScore:
    return {
        "candidate_id": "alice_chen_frontend",
        "name": "Alice Chen",
        "overall_score": 92.5,
        "skill_match": {"React": 1.0, "TypeScript": 0.9, "GraphQL": 0.7},
        "experience_fit": 0.95,
        "strengths": ["6 years React experience", "Led teams of 4–8 engineers"],
        "gaps": ["No mobile experience"],
        "reasoning": "Strong frontend match with deep React/TypeScript expertise.",
    }


@pytest.fixture
def valid_agent_state(valid_requirement, valid_candidate_score) -> AgentState:
    return {
        "messages": [HumanMessage(content="Find React engineers with 3+ years")],
        "raw_jd": "We need a senior React developer...",
        "parsed_requirements": [valid_requirement],
        "candidate_shortlist": [valid_candidate_score],
        "comparison_results": None,
        "current_round": 1,
        "round_history": [],
        "final_recommendations": None,
        "needs_human_feedback": False,
        "refinement_requested": False,
    }


# ── Requirement tests ──────────────────────────────────────────────────────────

class TestRequirement:

    def test_field_annotations_match_architecture(self):
        """All three fields from architecture §3 must be present."""
        annotations = Requirement.__annotations__
        assert "skill" in annotations
        assert "category" in annotations
        assert "min_years" in annotations

    def test_annotation_count(self):
        """Exactly 3 fields — no extras crept in."""
        assert len(Requirement.__annotations__) == 3

    def test_valid_must_have_construction(self, valid_requirement):
        assert valid_requirement["skill"] == "React"
        assert valid_requirement["category"] == "must_have"
        assert valid_requirement["min_years"] == 3

    def test_valid_nice_to_have_construction(self, valid_requirement_no_years):
        assert valid_requirement_no_years["category"] == "nice_to_have"
        assert valid_requirement_no_years["min_years"] is None

    def test_is_plain_dict(self, valid_requirement):
        """TypedDicts are regular dicts at runtime."""
        assert isinstance(valid_requirement, dict)

    def test_json_serialisable(self, valid_requirement, valid_requirement_no_years):
        """Both variants must round-trip through JSON."""
        for req in (valid_requirement, valid_requirement_no_years):
            dumped = json.dumps(req)
            loaded = json.loads(dumped)
            assert loaded["skill"] == req["skill"]
            assert loaded["category"] == req["category"]
            assert loaded["min_years"] == req["min_years"]

    def test_both_category_values_accepted(self):
        """Confirm both Literal values work at runtime."""
        for cat in ("must_have", "nice_to_have"):
            r: Requirement = {"skill": "Python", "category": cat, "min_years": None}
            assert r["category"] == cat

    def test_min_years_none_is_valid(self):
        r: Requirement = {"skill": "Docker", "category": "nice_to_have", "min_years": None}
        assert r["min_years"] is None

    def test_min_years_zero_is_valid(self):
        """Zero experience years is a valid edge case (no minimum stated)."""
        r: Requirement = {"skill": "Kubernetes", "category": "nice_to_have", "min_years": 0}
        assert r["min_years"] == 0


# ── CandidateScore tests ───────────────────────────────────────────────────────

class TestCandidateScore:

    def test_field_annotations_match_architecture(self):
        """All 8 fields from architecture §3 must be present."""
        annotations = CandidateScore.__annotations__
        required_fields = {
            "candidate_id", "name", "overall_score", "skill_match",
            "experience_fit", "strengths", "gaps", "reasoning",
        }
        assert required_fields == set(annotations.keys())

    def test_annotation_count(self):
        assert len(CandidateScore.__annotations__) == 8

    def test_valid_construction(self, valid_candidate_score):
        cs = valid_candidate_score
        assert cs["candidate_id"] == "alice_chen_frontend"
        assert cs["name"] == "Alice Chen"
        assert cs["overall_score"] == 92.5
        assert cs["experience_fit"] == 0.95

    def test_skill_match_is_dict_of_floats(self, valid_candidate_score):
        sm = valid_candidate_score["skill_match"]
        assert isinstance(sm, dict)
        for skill, score in sm.items():
            assert isinstance(skill, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_overall_score_range(self, valid_candidate_score):
        assert 0.0 <= valid_candidate_score["overall_score"] <= 100.0

    def test_experience_fit_range(self, valid_candidate_score):
        assert 0.0 <= valid_candidate_score["experience_fit"] <= 1.0

    def test_strengths_and_gaps_are_lists(self, valid_candidate_score):
        assert isinstance(valid_candidate_score["strengths"], list)
        assert isinstance(valid_candidate_score["gaps"], list)
        assert all(isinstance(s, str) for s in valid_candidate_score["strengths"])
        assert all(isinstance(g, str) for g in valid_candidate_score["gaps"])

    def test_reasoning_is_string(self, valid_candidate_score):
        assert isinstance(valid_candidate_score["reasoning"], str)
        assert len(valid_candidate_score["reasoning"]) > 0

    def test_is_plain_dict(self, valid_candidate_score):
        assert isinstance(valid_candidate_score, dict)

    def test_json_serialisable(self, valid_candidate_score):
        dumped = json.dumps(valid_candidate_score)
        loaded = json.loads(dumped)
        assert loaded["candidate_id"] == valid_candidate_score["candidate_id"]
        assert loaded["overall_score"] == valid_candidate_score["overall_score"]
        assert loaded["skill_match"] == valid_candidate_score["skill_match"]

    def test_empty_strengths_and_gaps_valid(self):
        """A candidate with no strengths or gaps listed is still valid."""
        cs: CandidateScore = {
            "candidate_id": "c_test",
            "name": "Test Candidate",
            "overall_score": 50.0,
            "skill_match": {},
            "experience_fit": 0.5,
            "strengths": [],
            "gaps": [],
            "reasoning": "Borderline match.",
        }
        assert cs["strengths"] == []
        assert cs["gaps"] == []


# ── AgentState tests ───────────────────────────────────────────────────────────

class TestAgentState:

    def test_field_annotations_match_architecture(self):
        """All 10 fields from architecture §3 must be present."""
        annotations = AgentState.__annotations__
        required_fields = {
            "messages",
            "raw_jd",
            "parsed_requirements",
            "candidate_shortlist",
            "comparison_results",
            "current_round",
            "round_history",
            "final_recommendations",
            "needs_human_feedback",
            "refinement_requested",
        }
        assert required_fields == set(annotations.keys())

    def test_annotation_count(self):
        assert len(AgentState.__annotations__) == 10

    def test_valid_construction(self, valid_agent_state):
        state = valid_agent_state
        assert state["raw_jd"] == "We need a senior React developer..."
        assert state["current_round"] == 1
        assert state["needs_human_feedback"] is False
        assert state["refinement_requested"] is False
        assert state["comparison_results"] is None
        assert state["final_recommendations"] is None

    def test_messages_is_list(self, valid_agent_state):
        assert isinstance(valid_agent_state["messages"], list)
        assert len(valid_agent_state["messages"]) == 1
        assert isinstance(valid_agent_state["messages"][0], HumanMessage)

    def test_parsed_requirements_is_list(self, valid_agent_state):
        reqs = valid_agent_state["parsed_requirements"]
        assert isinstance(reqs, list)
        assert all(isinstance(r, dict) for r in reqs)

    def test_candidate_shortlist_is_list(self, valid_agent_state):
        shortlist = valid_agent_state["candidate_shortlist"]
        assert isinstance(shortlist, list)
        assert all(isinstance(c, dict) for c in shortlist)

    def test_round_history_starts_empty(self, valid_agent_state):
        assert valid_agent_state["round_history"] == []

    def test_current_round_valid_values(self):
        """current_round must be 1, 2, or 3."""
        for round_num in (1, 2, 3):
            state: AgentState = {
                "messages": [],
                "raw_jd": "JD text",
                "parsed_requirements": [],
                "candidate_shortlist": [],
                "comparison_results": None,
                "current_round": round_num,
                "round_history": [],
                "final_recommendations": None,
                "needs_human_feedback": False,
                "refinement_requested": False,
            }
            assert state["current_round"] == round_num

    def test_optional_fields_accept_none(self, valid_agent_state):
        assert valid_agent_state["comparison_results"] is None
        assert valid_agent_state["final_recommendations"] is None

    def test_optional_fields_accept_values(self, valid_agent_state):
        valid_agent_state["comparison_results"] = {"c_001 vs c_002": "c_001 wins"}
        valid_agent_state["final_recommendations"] = [{"candidate_id": "c_001", "decision": "hire"}]
        assert valid_agent_state["comparison_results"] is not None
        assert valid_agent_state["final_recommendations"] is not None


# ── add_messages reducer tests ────────────────────────────────────────────────

class TestAddMessagesReducer:
    """
    Verify the add_messages reducer that powers AgentState.messages.
    LangGraph calls this when merging partial state updates; it must
    append new messages rather than overwrite the list.
    """

    def test_appends_to_existing_list(self):
        existing = [HumanMessage(content="Hello")]
        new_msgs = [AIMessage(content="Hi there!")]
        result = add_messages(existing, new_msgs)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_appends_multiple_messages(self):
        existing = [HumanMessage(content="Find React devs")]
        new_msgs = [
            AIMessage(content="Searching..."),
            AIMessage(content="Found 8 candidates."),
        ]
        result = add_messages(existing, new_msgs)
        assert len(result) == 3

    def test_appends_to_empty_list(self):
        result = add_messages([], [HumanMessage(content="First message")])
        assert len(result) == 1
        assert result[0].content == "First message"

    def test_preserves_message_types(self):
        existing = [
            SystemMessage(content="You are a recruiter assistant."),
            HumanMessage(content="Find Python devs"),
        ]
        new_msgs = [AIMessage(content="Found 5 candidates.")]
        result = add_messages(existing, new_msgs)
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_does_not_mutate_existing_list(self):
        existing = [HumanMessage(content="Original")]
        original_len = len(existing)
        add_messages(existing, [AIMessage(content="New")])
        # The original list reference should not be mutated
        assert len(existing) == original_len
