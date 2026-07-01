"""
nodes/search_resumes.py
========================
Search Resumes node — Phase 5.3.

Builds a semantic search query from ``state["parsed_requirements"]``, calls
``rag_search`` to retrieve matching candidates from ChromaDB, and populates
``state["candidate_shortlist"]`` with initial ``CandidateScore``-compatible
dicts (scores set to 0 — full scoring happens in rank_candidates).

Also sets ``state["current_round"] = 1`` to kick off the multi-round pipeline.

State writes: candidate_shortlist, current_round
"""

from __future__ import annotations

import logging

from state import AgentState

logger = logging.getLogger(__name__)

_TOP_K_SEARCH = 15   # over-fetch; rank_candidates will narrow down


def _build_query(requirements: list[dict]) -> str:
    """Construct a free-text search query from extracted requirements."""
    must_have = [r["skill"] for r in requirements if r.get("category") == "must_have"]
    nice_to_have = [r["skill"] for r in requirements if r.get("category") == "nice_to_have"]

    parts: list[str] = []
    if must_have:
        parts.append(" ".join(must_have))
    if nice_to_have:
        parts.append(" ".join(nice_to_have))

    query = " ".join(parts) if parts else "software engineer developer"
    logger.debug("search_resumes: query = %r", query)
    return query


def _rag_result_to_candidate(hit: dict) -> dict:
    """Convert a rag_search result into a minimal CandidateScore-compatible dict."""
    return {
        "candidate_id": hit["candidate_id"],
        "name": hit["candidate_id"].replace("_", " ").title(),
        "overall_score": 0.0,
        "skill_match": {},
        "experience_fit": 0.0,
        "strengths": [],
        "gaps": [],
        "reasoning": "",
        # Extra metadata carried from the search result
        "file_path": hit.get("file_path", ""),
        "relevance_score": hit.get("relevance_score", 0.0),
        "snippet": hit.get("snippet", ""),
    }


def run(state: AgentState) -> dict:
    """Build search query from requirements and populate initial candidate_shortlist.

    Returns a partial state update with ``candidate_shortlist`` and
    ``current_round`` set to 1.
    """
    from tools.rag_search import rag_search
    
    # If shortlist is already populated, this is a refinement loop.
    # Keep the existing candidates and just reset the round counter.
    if state.get("candidate_shortlist"):
        logger.info("search_resumes: Shortlist already populated (refinement). Skipping search.")
        return {"current_round": 1}

    requirements: list[dict] = state.get("parsed_requirements", [])
    if not requirements:
        logger.warning("search_resumes: no parsed_requirements — returning empty shortlist.")
        return {"candidate_shortlist": [], "current_round": 1}

    query = _build_query(requirements)
    logger.info("search_resumes: searching ChromaDB with query=%r, top_k=%d …", query, _TOP_K_SEARCH)

    hits: list[dict] = rag_search.invoke({"query": query, "top_k": _TOP_K_SEARCH})

    if not hits:
        logger.warning("search_resumes: rag_search returned no results.")
        return {"candidate_shortlist": [], "current_round": 1}

    shortlist = [_rag_result_to_candidate(h) for h in hits]
    logger.info("search_resumes: %d candidate(s) added to shortlist.", len(shortlist))

    return {
        "candidate_shortlist": shortlist,
        "current_round": 1,
    }
