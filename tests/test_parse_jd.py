"""
tests/test_parse_jd.py
=======================
Test Scenario 1 — Phase 10.2:
  End-to-end: JD → hire recommendation.

Validates that feeding a raw JD through the entire agent pipeline
produces valid final_recommendations with hire/no-hire decisions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from matching_agent import run_agent


def test_e2e_jd_to_hire_recommendation():
    """Submit a raw JD and ensure it traverses all nodes to yield recommendations."""

    # Mock the LLM to provide deterministic requirements and scores
    mock_llm = MagicMock()
    
    # We will need the LLM to respond to:
    # 1. extract_requirements
    # 2. rank_candidates (round 1) x2 (for 2 candidates)
    # 3. rank_candidates (round 2) x2
    # 4. compare_candidates (round 3)

    req_response = MagicMock()
    req_response.content = json.dumps({
        "must_have": [{"skill": "Python", "category": "must_have", "min_years": 3}],
        "nice_to_have": [],
        "experience_range": {"min_years": 3, "max_years": None},
        "education": "None"
    })
    
    score_resp_1 = MagicMock()
    score_resp_1.content = json.dumps({
        "overall_score": 90.0, "skill_match": {"Python": 1.0}, "experience_fit": 0.9, 
        "strengths": ["Great Python"], "gaps": [], "reasoning": "Strong match."
    })
    
    score_resp_2 = MagicMock()
    score_resp_2.content = json.dumps({
        "overall_score": 40.0, "skill_match": {"Python": 0.4}, "experience_fit": 0.4, 
        "strengths": [], "gaps": ["Lacks Python"], "reasoning": "Weak match."
    })

    comp_resp = MagicMock()
    comp_resp.content = json.dumps({
        "summary": "Candidate 1 wins.",
        "ranking": ["c_1", "c_2"],
        "matrix": {},
        "head_to_head": "1 is better than 2."
    })

    # The sequence of LLM calls might be more than exactly 5 depending on the graph execution, 
    # so we use a side_effect function.
    def mock_invoke(messages, *args, **kwargs):
        content = str(messages)
        if "must_have" in content or "Extract the structured requirements" in content:
            return req_response
        elif "compare" in content.lower() or "head-to-head" in content.lower():
            return comp_resp
        elif "c_1" in content:
            return score_resp_1
        else:
            return score_resp_2

    mock_llm.invoke.side_effect = mock_invoke

    # Mock ChromaDB search to return two candidates
    with patch("tools.rag_search.rag_search") as mock_search, \
         patch("prompts.system.get_llm", return_value=mock_llm), \
         patch("nodes.rank_candidates.get_llm", return_value=mock_llm), \
         patch("tools.requirements.get_llm", return_value=mock_llm), \
         patch("tools.comparison.get_llm", return_value=mock_llm):
        
        mock_search.invoke.return_value = [
            {"candidate_id": "c_1", "file_path": "/c1.pdf", "relevance_score": 0.9, "snippet": "Python dev"},
            {"candidate_id": "c_2", "file_path": "/c2.pdf", "relevance_score": 0.5, "snippet": "Java dev"},
        ]
        
        result = run_agent("I need a Python developer with 3 years experience.")
        
        # Verify the pipeline completed
        assert result.get("needs_human_feedback") is True
        
        # Verify recommendations
        recs = result.get("final_recommendations", [])
        assert len(recs) == 2
        
        c1_rec = next(r for r in recs if r["candidate_id"] == "c_1")
        c2_rec = next(r for r in recs if r["candidate_id"] == "c_2")
        
        assert c1_rec["hire_recommendation"] in ("strong_hire", "hire")
        assert c2_rec["hire_recommendation"] == "no_hire"
