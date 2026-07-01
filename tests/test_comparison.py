"""
tests/test_comparison.py
=========================
Test Scenario 3 — Phase 10.4:
  compare_candidates returns accurate side-by-side matrix.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from state import AgentState


def test_compare_candidates_tool_returns_matrix():
    """Verify that the compare_candidates tool correctly parses the LLM JSON response."""
    from tools.comparison import compare_candidates
    
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    
    expected_output = {
        "summary": "Alice is stronger than Bob.",
        "ranking": ["alice", "bob"],
        "matrix": {
            "alice": {
                "overall_score": 95,
                "skill_match": {"React": 1.0},
                "experience_fit": 0.9,
                "strengths": ["Great UI skills"],
                "gaps": [],
                "hire_recommendation": "strong_hire",
                "reasoning": "Top tier."
            },
            "bob": {
                "overall_score": 75,
                "skill_match": {"React": 0.7},
                "experience_fit": 0.8,
                "strengths": ["Good backend"],
                "gaps": ["Weak UI"],
                "hire_recommendation": "hire",
                "reasoning": "Solid."
            }
        },
        "head_to_head": "Alice beats Bob in frontend."
    }
    
    mock_resp.content = json.dumps(expected_output)
    mock_llm.invoke.return_value = mock_resp
    
    # compare_candidates relies on state. We must mock the state context or patch it.
    # Actually, compare_candidates tool just takes candidate_ids and jd_text. 
    # But wait! Inside compare_candidates, how does it fetch candidate profiles?
    # Ah, let's look at `tools/comparison.py` to see its signature.
    
    with patch("tools.comparison.get_llm", return_value=mock_llm), \
         patch("tools.comparison._fetch_candidate_chunks") as mock_fetch:
         
        mock_fetch.return_value = {"alice": "Profile texts...", "bob": "Profile texts..."}
        
        result = compare_candidates.invoke({
            "candidate_ids": ["alice", "bob"],
            "jd_text": "React Developer"
        })
        
        assert result["ranking"] == ["alice", "bob"]
        assert "matrix" in result
        assert result["matrix"]["alice"]["overall_score"] == 95
        assert result["summary"] == "Alice is stronger than Bob."
