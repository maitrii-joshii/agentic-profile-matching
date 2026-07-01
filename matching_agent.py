"""
matching_agent.py
=================
LangGraph Agent Assembly — Phase 6.

Wires all Phase 5 nodes into a StateGraph with conditional routing
for multi-round screening and human feedback loops.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from state import AgentState
from nodes import (
    extract_requirements,
    generate_report,
    human_feedback,
    parse_jd,
    rank_candidates,
    search_resumes,
)

logger = logging.getLogger(__name__)


def route_after_ranking(state: AgentState) -> str:
    """Loop multi-round ranking.
    
    If current_round < 3, we loop back to rank_candidates for the next round.
    Otherwise, we proceed to generate_report.
    """
    current_round = state.get("current_round", 1)
    if current_round <= 3:
        return "rank_candidates"
    return "generate_report"


def route_after_feedback(state: AgentState) -> str:
    """Route based on flags set in human_feedback.
    
    If needs_human_feedback is True, we halt (END) so the UI can collect input.
    If refinement_requested is True, we loop back to extract_requirements.
    If current_round < 3 (re-rank requested), we loop back to rank_candidates.
    Otherwise (approved), we halt (END).
    """
    if state.get("needs_human_feedback"):
        # Await input from the user (UI layer handles this)
        return END

    if state.get("refinement_requested"):
        return "extract_requirements"

    # In human_feedback, if 'rerank' intent is matched, current_round is reset to 1
    if state.get("current_round", 3) < 3:
        return "rank_candidates"

    # Approved or otherwise done
    return END


# ==============================================================================
# Graph Assembly
# ==============================================================================

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("parse_jd", parse_jd.run)
graph.add_node("extract_requirements", extract_requirements.run)
graph.add_node("search_resumes", search_resumes.run)
graph.add_node("rank_candidates", rank_candidates.run)
graph.add_node("generate_report", generate_report.run)
graph.add_node("human_feedback", human_feedback.run)

def route_entry(state: AgentState) -> str:
    """Determine where to start the graph based on the initial state.
    
    If the UI is resuming an existing session, candidate_shortlist will be populated.
    In that case, we jump straight to human_feedback to process the new message.
    Otherwise, this is a fresh run and we start at parse_jd.
    """
    if len(state.get("candidate_shortlist", [])) > 0:
        return "human_feedback"
    return "parse_jd"

# Linear edges (forward pass)
graph.set_conditional_entry_point(
    route_entry,
    {
        "human_feedback": "human_feedback",
        "parse_jd": "parse_jd",
    }
)
graph.add_edge("parse_jd", "extract_requirements")
graph.add_edge("extract_requirements", "search_resumes")
graph.add_edge("search_resumes", "rank_candidates")

# Multi-round routing
graph.add_conditional_edges("rank_candidates", route_after_ranking, {
    "rank_candidates": "rank_candidates",
    "generate_report": "generate_report",
})

# Final linear
graph.add_edge("generate_report", "human_feedback")

# Feedback loop routing
graph.add_conditional_edges("human_feedback", route_after_feedback, {
    "extract_requirements": "extract_requirements",
    "rank_candidates": "rank_candidates",
    END: END,
})

agent = graph.compile()


# ==============================================================================
# Helper
# ==============================================================================

def run_agent(user_input: str) -> dict:
    """Convenience function for programmatic use (e.g. tests/E2E).
    
    Initialises a fresh state with the user message and invokes the graph.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
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
    return agent.invoke(initial_state)
