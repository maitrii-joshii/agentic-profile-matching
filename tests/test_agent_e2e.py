"""
tests/test_agent_e2e.py
========================
End-to-end integration tests for the LangGraph agent assembly (Phase 6).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from matching_agent import agent, run_agent


def _mock_rag_hits() -> list[dict]:
    return [
        {"candidate_id": "alice", "file_path": "/alice.pdf", "relevance_score": 0.9, "snippet": "React dev"},
        {"candidate_id": "bob", "file_path": "/bob.pdf", "relevance_score": 0.8, "snippet": "Node dev"},
        {"candidate_id": "charlie", "file_path": "/charlie.pdf", "relevance_score": 0.7, "snippet": "Fullstack"},
    ]


def _mock_llm_response(score: float = 80.0):
    response = MagicMock()
    response.content = json.dumps({
        "overall_score": score,
        "skill_match": {"React": 0.9},
        "experience_fit": 0.8,
        "strengths": ["Strong engineering"],
        "gaps": ["No Kubernetes"],
        "reasoning": "Solid candidate",
    })
    return response


@patch("tools.requirements.extract_requirements")
@patch("tools.rag_search.rag_search")
@patch("nodes.rank_candidates.get_llm")
@patch("nodes.rank_candidates.call_llm_with_retry")
@patch("nodes.human_feedback._detect_intent_llm", return_value=("unknown", None))
def test_agent_e2e_forward_pass(
    mock_intent: MagicMock,
    mock_call_llm: MagicMock,
    mock_get_llm: MagicMock,
    mock_rag: MagicMock,
    mock_extract: MagicMock,
) -> None:
    """Test a full forward pass: JD -> Requirements -> Search -> Rank(1,2,3) -> Report -> Feedback."""
    
    # 1. Mock extract_requirements tool
    mock_extract.invoke.return_value = {
        "must_have": [{"skill": "React", "category": "must_have", "min_years": 3}],
        "nice_to_have": [],
        "experience_range": {},
        "education": "",
    }
    
    # 2. Mock rag_search tool
    mock_rag.invoke.return_value = _mock_rag_hits()
    
    # 3. Mock LLM for scoring
    mock_llm = MagicMock()
    # Decreasing scores so ranking changes order predictably (if needed)
    mock_llm.invoke.side_effect = [
        _mock_llm_response(90.0),  # alice R1
        _mock_llm_response(80.0),  # bob R1
        _mock_llm_response(70.0),  # charlie R1
        _mock_llm_response(92.0),  # alice R2
        _mock_llm_response(82.0),  # bob R2
        _mock_llm_response(72.0),  # charlie R2
    ]
    mock_get_llm.return_value = mock_llm
    mock_call_llm.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)
    
    # 4. Mock compare_candidates for Round 3
    with patch("tools.comparison.compare_candidates") as mock_cmp:
        mock_cmp.invoke.return_value = {
            "summary": "Alice wins",
            "ranking": ["alice", "bob", "charlie"],
            "matrix": {},
            "head_to_head": "",
        }
        
        # Run agent
        result = run_agent("We need a senior React dev.")
        
    # Assertions
    # JD parsing
    assert "React" in result["raw_jd"]
    
    # Requirements
    assert len(result["parsed_requirements"]) == 1
    
    # Multi-round finished at round 3
    assert result["current_round"] == 4
    assert len(result["round_history"]) == 3
    
    # Final recommendations
    recs = result["final_recommendations"]
    assert len(recs) == 3
    # Alice should be strong_hire (score 92 >= 85)
    alice_rec = next(r for r in recs if r["candidate_id"] == "alice")
    assert alice_rec["hire_recommendation"] == "strong_hire"
    
    # Unknown intent keeps loop open
    assert result["needs_human_feedback"] is True
    assert result.get("refinement_requested", False) is False
    
    # Summary message added
    assert any(isinstance(m, AIMessage) for m in result["messages"])


def test_agent_refinement_loop() -> None:
    """Test that a refinement request sets flags and re-routes properly."""
    initial_state = {
        "messages": [
            HumanMessage(content="We need a senior React dev."),
            AIMessage(content="Here are candidates"),
            HumanMessage(content="Also require Python"),
        ],
        "raw_jd": "We need a senior React dev.",
        "parsed_requirements": [{"skill": "React", "category": "must_have", "min_years": 3}],
        "candidate_shortlist": _mock_rag_hits(),
        "current_round": 3,
        "round_history": [],
        "needs_human_feedback": True,
        "refinement_requested": False,
        "comparison_results": None,
        "final_recommendations": None,
    }
    
    with patch("nodes.human_feedback._detect_intent_llm", return_value=("refinement", None)):
        from nodes.human_feedback import run as hf_run
        hf_state = hf_run(initial_state)
    
    # It detected refinement
    assert hf_state["refinement_requested"] is True
    
    # Test router logic
    from matching_agent import route_after_feedback
    next_node = route_after_feedback(hf_state)
    assert next_node == "extract_requirements"
