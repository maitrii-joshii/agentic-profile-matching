"""
tests/test_edge_cases.py
=========================
Test Scenario 7 — Phase 10.7:
  Edge cases: empty resume dir, all corrupted, empty JD, etc.
"""

from __future__ import annotations

from unittest.mock import patch

from state import AgentState


def test_search_returns_empty_when_no_requirements():
    """If JD yields no requirements, search should return empty shortlist safely."""
    import nodes.search_resumes as sr
    
    state: AgentState = {
        "messages": [], "raw_jd": "Just a chat, not a JD", "parsed_requirements": [],
        "candidate_shortlist": [], "comparison_results": None, "current_round": 1,
        "round_history": [], "final_recommendations": None, "needs_human_feedback": False,
        "refinement_requested": False
    }
    
    with patch("tools.rag_search.rag_search") as mock_search:
        result = sr.run(state)
        mock_search.invoke.assert_not_called()
        assert result["candidate_shortlist"] == []


def test_ranking_safely_ignores_empty_shortlist():
    """If shortlist is empty, ranking nodes should safely bypass."""
    import nodes.rank_candidates as rc
    import nodes.generate_report as gr
    
    state: AgentState = {
        "messages": [], "raw_jd": "", "parsed_requirements": [],
        "candidate_shortlist": [], "comparison_results": None, "current_round": 1,
        "round_history": [], "final_recommendations": None, "needs_human_feedback": False,
        "refinement_requested": False
    }
    
    # rank_candidates
    with patch("nodes.rank_candidates.get_llm") as mock_llm:
        res_rank = rc.run(state)
        mock_llm.assert_not_called()
        assert res_rank["current_round"] == 1
        
    # generate_report
    res_rep = gr.run(state)
    assert res_rep["final_recommendations"] == []
    assert res_rep["needs_human_feedback"] is True


def test_extract_requirements_with_empty_jd():
    """If raw_jd is empty, extract_requirements should skip extraction."""
    import nodes.extract_requirements as er
    
    state: AgentState = {
        "messages": [], "raw_jd": "", "parsed_requirements": [],
        "candidate_shortlist": [], "comparison_results": None, "current_round": 1,
        "round_history": [], "final_recommendations": None, "needs_human_feedback": False,
        "refinement_requested": False
    }
    
    with patch("tools.requirements.extract_requirements") as mock_tool:
        res = er.run(state)
        mock_tool.invoke.assert_not_called()
        assert res["parsed_requirements"] == []

