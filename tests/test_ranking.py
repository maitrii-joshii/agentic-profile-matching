"""
tests/test_ranking.py
======================
Test Scenario 5 — Phase 8.7:
  Multi-round screening pipeline: Round 1 (top 10) → Round 2 (top 5) → Final (hire/no-hire)

Validates:
  - Round 1 scores all candidates, keeps top ROUND1_KEEP (10)
  - Round 2 deeply analyses top 10, narrows to ROUND2_KEEP (5)
  - Round 3 triggers head-to-head comparison, applies ranking, sets current_round=4
  - round_history accumulates one snapshot per round
  - Score consistency: scores are floats in [0, 100]
  - Full 3-round progression wired correctly through the agent router
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from state import AgentState


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_candidate(cid: str, score: float = 70.0, snippet: str = "developer") -> dict:
    return {
        "candidate_id": cid,
        "file_path": f"/resumes/{cid}.pdf",
        "relevance_score": score / 100,
        "snippet": snippet,
        "overall_score": score,
        "skill_match": {},
        "experience_fit": 0.7,
        "strengths": [f"{cid} strength"],
        "gaps": [f"{cid} gap"],
        "reasoning": f"{cid} reasoning",
    }


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content="Find a React developer")],
        "raw_jd": "Senior React Developer, 3+ years",
        "parsed_requirements": [{"skill": "React", "category": "must_have", "min_years": 3}],
        "candidate_shortlist": [],
        "comparison_results": None,
        "current_round": 1,
        "round_history": [],
        "final_recommendations": None,
        "needs_human_feedback": False,
        "refinement_requested": False,
    }
    base.update(overrides)
    return base


def _mock_llm_response(score: float) -> MagicMock:
    resp = MagicMock()
    resp.content = json.dumps({
        "overall_score": score,
        "skill_match": {"React": 0.9},
        "experience_fit": 0.8,
        "strengths": ["Strong React skills"],
        "gaps": ["No TypeScript"],
        "reasoning": f"Good candidate with score {score}",
    })
    return resp


# ── Round 1 Tests ───────────────────────────────────────────────────────────────

class TestRound1BroadScreen:
    """8.1 — Round 1: broad screen, keep top 10."""

    def test_round1_keeps_at_most_10(self):
        """Starting with 15 candidates, Round 1 should keep only top 10."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}", score=float(i * 4)) for i in range(1, 16)]
        state = _make_state(candidate_shortlist=candidates, current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = [_mock_llm_response(float(80 - i * 2)) for i in range(15)]
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert len(result["candidate_shortlist"]) <= 10

    def test_round1_advances_to_round2(self):
        """Round 1 output should set current_round=2."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}") for i in range(5)]
        state = _make_state(candidate_shortlist=candidates, current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(75.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert result["current_round"] == 2

    def test_round1_saves_snapshot(self):
        """Round 1 should append one snapshot to round_history."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}") for i in range(3)]
        state = _make_state(candidate_shortlist=candidates, current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(80.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert len(result["round_history"]) == 1
        assert result["round_history"][0]["round"] == 1

    def test_round1_sorted_by_score(self):
        """Candidates in round 1 output should be sorted descending by score."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}") for i in range(4)]
        scores = [60.0, 90.0, 45.0, 80.0]
        state = _make_state(candidate_shortlist=candidates, current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = [_mock_llm_response(s) for s in scores]
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        out_scores = [c["overall_score"] for c in result["candidate_shortlist"]]
        assert out_scores == sorted(out_scores, reverse=True)


# ── Round 2 Tests ───────────────────────────────────────────────────────────────

class TestRound2DeepAnalysis:
    """8.2 — Round 2: deep analysis, narrow to top 5."""

    def test_round2_keeps_at_most_5(self):
        """Starting with 10 candidates in Round 2, output should be at most 5."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}") for i in range(10)]
        state = _make_state(candidate_shortlist=candidates, current_round=2)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(75.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert len(result["candidate_shortlist"]) <= 5

    def test_round2_advances_to_round3(self):
        """Round 2 output should set current_round=3."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}") for i in range(3)]
        state = _make_state(candidate_shortlist=candidates, current_round=2)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(75.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert result["current_round"] == 3

    def test_round2_appends_to_existing_history(self):
        """Round 2 snapshot should be appended to the existing Round 1 history."""
        import nodes.rank_candidates as rc

        existing_history = [{"round": 1, "candidates": [{"candidate_id": "c00", "overall_score": 80.0}]}]
        candidates = [_make_candidate(f"c{i:02d}") for i in range(3)]
        state = _make_state(
            candidate_shortlist=candidates,
            current_round=2,
            round_history=existing_history,
        )

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(82.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        assert len(result["round_history"]) == 2
        assert result["round_history"][1]["round"] == 2


# ── Round 3 / Final Tests ───────────────────────────────────────────────────────

class TestRound3FinalDecision:
    """8.3 — Final round: head-to-head comparison, hire/no-hire."""

    def test_round3_calls_compare_candidates(self):
        """Round 3 should invoke compare_candidates when >= 2 candidates exist."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate("alice", 90.0), _make_candidate("bob", 75.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3, raw_jd="React dev")

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            mock_cmp.invoke.return_value = {
                "ranking": ["alice", "bob"],
                "summary": "Alice wins",
                "matrix": {},
            }
            result = rc.run(state)

        mock_cmp.invoke.assert_called_once()
        assert result.get("comparison_results") is not None

    def test_round3_applies_ranking_order(self):
        """Round 3 should reorder shortlist according to comparison ranking."""
        import nodes.rank_candidates as rc

        # Bob has higher raw score but comparison says alice wins
        candidates = [_make_candidate("bob", 88.0), _make_candidate("alice", 82.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3)

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            mock_cmp.invoke.return_value = {
                "ranking": ["alice", "bob"],
                "summary": "Alice wins on overall fit",
                "matrix": {},
            }
            result = rc.run(state)

        ids = [c["candidate_id"] for c in result["candidate_shortlist"]]
        assert ids[0] == "alice"
        assert ids[1] == "bob"

    def test_round3_sets_current_round_sentinel(self):
        """Round 3 should set current_round=4 (sentinel) to stop further looping."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate("alice", 90.0), _make_candidate("bob", 75.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3)

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            mock_cmp.invoke.return_value = {"ranking": ["alice", "bob"]}
            result = rc.run(state)

        assert result["current_round"] == 4  # sentinel: round 3 complete

    def test_round3_appends_history_snapshot(self):
        """Round 3 should append a round=3 snapshot to round_history."""
        import nodes.rank_candidates as rc

        existing = [
            {"round": 1, "candidates": []},
            {"round": 2, "candidates": []},
        ]
        candidates = [_make_candidate("alice", 90.0), _make_candidate("bob", 75.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3, round_history=existing)

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            mock_cmp.invoke.return_value = {"ranking": ["alice", "bob"]}
            result = rc.run(state)

        assert len(result["round_history"]) == 3
        assert result["round_history"][2]["round"] == 3

    def test_round3_handles_single_candidate_gracefully(self):
        """Round 3 should skip compare_candidates if only 1 candidate remains."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate("solo", 88.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3)

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            result = rc.run(state)

        mock_cmp.invoke.assert_not_called()
        assert result["candidate_shortlist"][0]["candidate_id"] == "solo"

    def test_round3_gracefully_handles_comparison_failure(self):
        """Round 3 should continue even if compare_candidates raises an exception."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate("alice", 90.0), _make_candidate("bob", 75.0)]
        state = _make_state(candidate_shortlist=candidates, current_round=3)

        with patch("tools.comparison.compare_candidates") as mock_cmp:
            mock_cmp.invoke.side_effect = RuntimeError("LLM timeout")
            result = rc.run(state)

        # Should still return a valid result without comparison_results
        assert "candidate_shortlist" in result
        assert result.get("comparison_results") is None


# ── Score Consistency Tests ─────────────────────────────────────────────────────

class TestScoreConsistency:
    """8.1-8.3 — Scores must be floats in [0, 100]."""

    def test_scores_are_floats_in_valid_range(self):
        """All overall_score values must be numeric and within [0, 100]."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i}") for i in range(5)]
        state = _make_state(candidate_shortlist=candidates, current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(77.5)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        for c in result["candidate_shortlist"]:
            assert isinstance(c["overall_score"], float)
            assert 0.0 <= c["overall_score"] <= 100.0

    def test_malformed_llm_response_falls_back_to_relevance_score(self):
        """If LLM returns invalid JSON, fall back to relevance_score * 100."""
        import nodes.rank_candidates as rc

        candidate = _make_candidate("fallback_c", score=65.0)  # relevance_score = 0.65
        candidate["relevance_score"] = 0.65
        state = _make_state(candidate_shortlist=[candidate], current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry:
            mock_llm = MagicMock()
            bad_resp = MagicMock()
            bad_resp.content = "This is not JSON"
            mock_llm.invoke.return_value = bad_resp
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)

            result = rc.run(state)

        c = result["candidate_shortlist"][0]
        assert c["overall_score"] == pytest.approx(65.0, abs=1.0)

    def test_empty_shortlist_returns_without_scoring(self):
        """Empty shortlist should return immediately without calling the LLM."""
        import nodes.rank_candidates as rc

        state = _make_state(candidate_shortlist=[], current_round=1)

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm:
            result = rc.run(state)

        mock_get_llm.assert_not_called()
        assert "candidate_shortlist" not in result or result.get("candidate_shortlist") is None


# ── Full 3-Round Progression ────────────────────────────────────────────────────

class TestFullRoundProgression:
    """8.4 — Validate the full Round 1 → 2 → Final pipeline via router."""

    def test_route_after_ranking_loops_at_round_1(self):
        """Router should loop back for rounds 1, 2, 3 and stop at 4."""
        from matching_agent import route_after_ranking

        assert route_after_ranking({"current_round": 1}) == "rank_candidates"
        assert route_after_ranking({"current_round": 2}) == "rank_candidates"
        assert route_after_ranking({"current_round": 3}) == "rank_candidates"
        assert route_after_ranking({"current_round": 4}) == "generate_report"

    def test_round_history_grows_across_rounds(self):
        """After 3 rounds, round_history should have exactly 3 snapshots."""
        import nodes.rank_candidates as rc

        candidates = [_make_candidate(f"c{i:02d}", float(90 - i * 5)) for i in range(5)]

        with patch("nodes.rank_candidates.get_llm") as mock_get_llm, \
             patch("nodes.rank_candidates.call_llm_with_retry") as mock_retry, \
             patch("tools.comparison.compare_candidates") as mock_cmp:

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = _mock_llm_response(80.0)
            mock_get_llm.return_value = mock_llm
            mock_retry.side_effect = lambda llm, msgs, **kw: llm.invoke(msgs)
            mock_cmp.invoke.return_value = {
                "ranking": [c["candidate_id"] for c in candidates[:2]],
                "summary": "top 2 ranked",
            }

            # Simulate rounds manually
            state_r1 = _make_state(candidate_shortlist=candidates, current_round=1)
            r1 = rc.run(state_r1)

            state_r2 = _make_state(
                candidate_shortlist=r1["candidate_shortlist"],
                current_round=2,
                round_history=r1["round_history"],
            )
            r2 = rc.run(state_r2)

            state_r3 = _make_state(
                candidate_shortlist=r2["candidate_shortlist"],
                current_round=3,
                round_history=r2["round_history"],
            )
            r3 = rc.run(state_r3)

        assert len(r3["round_history"]) == 3
        assert [h["round"] for h in r3["round_history"]] == [1, 2, 3]
