"""
tests/test_refinement.py
========================
Test Scenario 2 — Phase 9.5:
  Mid-conversation refinement re-ranks correctly.

Validates:
  - refine_requirements updates parsed_requirements (add/remove skills).
  - search_resumes bypasses ChromaDB search if shortlist is already populated.
  - rank_candidates re-scores affected candidates.
  - generate_report identifies ranking deltas.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from state import AgentState


def _make_candidate(cid: str, score: float = 80.0) -> dict:
    return {
        "candidate_id": cid,
        "name": f"Candidate {cid}",
        "overall_score": score,
        "skill_match": {"React": 0.9},
        "experience_fit": 0.8,
        "strengths": [],
        "gaps": [],
        "reasoning": f"Old reasoning {cid}",
        "snippet": f"Resume {cid}",
    }


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content="Add TypeScript")],
        "raw_jd": "React Developer",
        "parsed_requirements": [{"skill": "React", "category": "must_have", "min_years": 3}],
        "candidate_shortlist": [
            _make_candidate("alice", 90.0),
            _make_candidate("bob", 85.0),
        ],
        "comparison_results": None,
        "current_round": 3,
        "round_history": [
            {"round": 1, "candidates": []},
            {"round": 2, "candidates": []},
            {"round": 3, "candidates": [
                {"candidate_id": "alice", "overall_score": 90.0},
                {"candidate_id": "bob", "overall_score": 85.0}
            ]},
            {"round": 1, "candidates": []},
            {"round": 2, "candidates": []},
            {"round": 3, "candidates": [
                {"candidate_id": "bob", "overall_score": 95.0},
                {"candidate_id": "alice", "overall_score": 80.0}
            ]}
        ],
        "final_recommendations": None,
        "needs_human_feedback": False,
        "refinement_requested": True,
    }
    base.update(overrides)
    return base


class TestRequirementRefinement:
    """9.2 — Incremental requirement update."""

    def test_extract_requirements_node_calls_refinement_tool(self):
        import nodes.extract_requirements as er

        state = _make_state()
        
        with patch("tools.requirements.refine_requirements") as mock_tool:
            mock_tool.invoke.return_value = {
                "must_have": [{"skill": "React"}, {"skill": "TypeScript"}],
                "nice_to_have": []
            }
            result = er.run(state)
            
            mock_tool.invoke.assert_called_once()
            args = mock_tool.invoke.call_args[0][0]
            assert "instruction" in args
            assert args["instruction"] == "Add TypeScript"
            
            assert len(result["parsed_requirements"]) == 2
            assert result["refinement_requested"] is False


class TestSearchResumesBypass:
    """9.3 — Bypassing ChromaDB on refinement."""

    def test_search_resumes_skips_search_if_shortlist_populated(self):
        import nodes.search_resumes as sr

        state = _make_state()
        
        with patch("tools.rag_search.rag_search") as mock_search:
            result = sr.run(state)
            mock_search.invoke.assert_not_called()
            
            assert result["current_round"] == 1


class TestSmartRescoring:
    """9.3 — Smart re-ranking."""

    def test_rank_candidates_uses_rescoring_prompt_when_skill_match_exists(self):
        import nodes.rank_candidates as rc

        state = _make_state(current_round=1)
        
        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            
            mock_llm = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = '{"overall_score": 88.0}'
            mock_llm.invoke.return_value = mock_resp
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)
            
            rc.run(state)
            
            # Verify the prompt passed to the LLM contains CANDIDATE_RESCORING_USER text
            # We can inspect what was passed to mock_retry
            assert mock_retry.call_count == 2  # Once for alice, once for bob
            msgs = mock_retry.call_args[0][1]
            content = msgs[1].content
            assert "Previous Score Data:" in content


class TestRankingChangeExplanation:
    """9.4 — Compare pre and post refinement rankings."""

    def test_generate_report_detects_ranking_changes(self):
        import nodes.generate_report as gr

        # State has Alice #1, Bob #2 in history.
        # Now we reverse them in the shortlist (e.g. because Bob had TypeScript).
        state = _make_state()
        state["candidate_shortlist"] = [
            _make_candidate("bob", 95.0),
            _make_candidate("alice", 80.0),
        ]
        
        result = gr.run(state)
        
        msg_content = result["messages"][0].content
        assert "Refinement Delta" in msg_content
        assert "Bob" in msg_content and "moved UP" in msg_content
        assert "Alice" in msg_content and "moved DOWN" in msg_content

    def test_generate_report_no_change(self):
        import nodes.generate_report as gr

        state = _make_state()
        # Same order as history
        
        result = gr.run(state)
        
        msg_content = result["messages"][0].content
        assert "Ranking remained unchanged" in msg_content
