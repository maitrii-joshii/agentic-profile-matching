"""
nodes/rank_candidates.py
=========================
Rank Candidates node — Phase 5.4 / 8.1-8.3.

Implements the 3-round screening pipeline:

  Round 1 (broad screen)
    - Score all candidates in shortlist against must-have requirements using
      the DeepSeek LLM.
    - Keep top 10 by overall_score.
    - Save snapshot to round_history.

  Round 2 (deep analysis)
    - Detailed per-candidate analysis of top 10: skills audit, experience
      mapping, project relevance.
    - Narrow to top 5.
    - Save snapshot to round_history.

  Round 3 (final / head-to-head)
    - Call compare_candidates for head-to-head analysis.
    - Assign hire/no-hire recommendations.
    - Save snapshot to round_history. Sets current_round = 3.

State writes: candidate_shortlist, comparison_results, current_round, round_history
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from prompts.system import call_llm_with_retry, get_llm
from state import AgentState

logger = logging.getLogger(__name__)

# ── Round limits ───────────────────────────────────────────────────────────────
ROUND1_KEEP = 10
ROUND2_KEEP = 5


# ── LLM-based scoring ──────────────────────────────────────────────────────────

_SCORING_SYSTEM = """\
You are an expert technical recruiter. Score the given candidate's resume snippet
against the job requirements. Return ONLY valid JSON with this schema:
{
  "overall_score": <0-100>,
  "skill_match": {"<skill>": <0.0-1.0>, ...},
  "experience_fit": <0.0-1.0>,
  "strengths": ["<bullet>", ...],
  "gaps": ["<bullet>", ...],
  "reasoning": "<one paragraph>"
}
"""

_SCORING_USER = """\
Requirements:
{requirements}

Candidate (ID: {candidate_id}):
{snippet}

Return the JSON score.
"""


def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _score_candidate(candidate: dict, requirements: list[dict], llm) -> dict:
    """Ask the LLM to score a single candidate against requirements."""
    req_text = "\n".join(
        f"- {r['skill']} ({r.get('category','?')}, min {r.get('min_years','?')} yrs)"
        for r in requirements
    )
    snippet = candidate.get("snippet", candidate.get("reasoning", ""))[:1200]

    messages = [SystemMessage(content=_SCORING_SYSTEM)]
    
    # If the candidate already has a skill_match dict, it's a rescoring operation
    if "skill_match" in candidate and candidate["skill_match"]:
        from prompts.templates import CANDIDATE_RESCORING_USER
        previous_data = {
            "overall_score": candidate.get("overall_score"),
            "skill_match": candidate.get("skill_match"),
            "experience_fit": candidate.get("experience_fit"),
            "reasoning": candidate.get("reasoning")
        }
        messages.append(HumanMessage(content=CANDIDATE_RESCORING_USER.format(
            requirements=req_text,
            previous_score_json=json.dumps(previous_data, indent=2),
            candidate_id=candidate["candidate_id"],
            snippet=snippet,
        )))
        logger.debug("_score_candidate: Using RESCORING prompt for %s", candidate["candidate_id"])
    else:
        messages.append(HumanMessage(content=_SCORING_USER.format(
            requirements=req_text,
            candidate_id=candidate["candidate_id"],
            snippet=snippet,
        )))

    try:
        response = call_llm_with_retry(llm, messages)
        data = json.loads(_extract_json(response.content))
    except Exception as exc:
        logger.warning("_score_candidate: scoring failed for %r — %s", candidate["candidate_id"], exc)
        data = {}

    updated = {**candidate}
    updated["overall_score"] = float(data.get("overall_score", candidate.get("relevance_score", 0.0) * 100))
    updated["skill_match"] = data.get("skill_match", {})
    updated["experience_fit"] = float(data.get("experience_fit", 0.0))
    updated["strengths"] = data.get("strengths", [])
    updated["gaps"] = data.get("gaps", [])
    updated["reasoning"] = data.get("reasoning", "")
    return updated


def _snapshot(shortlist: list[dict], round_num: int) -> dict:
    return {
        "round": round_num,
        "candidates": [
            {"candidate_id": c["candidate_id"], "overall_score": c["overall_score"]}
            for c in shortlist
        ],
    }


# ── Node entry point ───────────────────────────────────────────────────────────

def run(state: AgentState) -> dict:
    """Score and rank candidates for the current round.

    Returns a partial state update with ``candidate_shortlist``,
    ``current_round``, ``round_history``, and optionally
    ``comparison_results``.
    """
    shortlist: list[dict] = list(state.get("candidate_shortlist", []))
    requirements: list[dict] = state.get("parsed_requirements", [])
    current_round: int = state.get("current_round", 1)
    round_history: list[dict] = list(state.get("round_history", []))

    if not shortlist:
        logger.warning("rank_candidates: empty shortlist — nothing to rank.")
        # Fast-forward to bypass further routing loops
        return {"current_round": 4}

    llm = get_llm()

    # ── Round 1: broad score, keep top 10 ─────────────────────────────────────
    if current_round == 1:
        logger.info("rank_candidates: Round 1 — scoring %d candidates …", len(shortlist))
        scored = [_score_candidate(c, requirements, llm) for c in shortlist]
        scored.sort(key=lambda c: c["overall_score"], reverse=True)
        shortlist = scored[:ROUND1_KEEP]
        round_history.append(_snapshot(shortlist, 1))
        logger.info("rank_candidates: Round 1 complete — top %d kept.", len(shortlist))
        return {
            "candidate_shortlist": shortlist,
            "current_round": 2,
            "round_history": round_history,
        }

    # ── Round 2: deep analysis, keep top 5 ────────────────────────────────────
    elif current_round == 2:
        logger.info("rank_candidates: Round 2 — deep analysis of %d candidates …", len(shortlist))
        scored = [_score_candidate(c, requirements, llm) for c in shortlist]
        scored.sort(key=lambda c: c["overall_score"], reverse=True)
        shortlist = scored[:ROUND2_KEEP]
        round_history.append(_snapshot(shortlist, 2))
        logger.info("rank_candidates: Round 2 complete — top %d kept.", len(shortlist))
        return {
            "candidate_shortlist": shortlist,
            "current_round": 3,
            "round_history": round_history,
        }

    # ── Round 3: head-to-head comparison ──────────────────────────────────────
    else:
        logger.info("rank_candidates: Round 3 — head-to-head for %d candidates …", len(shortlist))
        candidate_ids = [c["candidate_id"] for c in shortlist]
        comparison_results = None

        if len(candidate_ids) >= 2:
            try:
                from tools.comparison import compare_candidates
                raw_jd: str = state.get("raw_jd", "")
                comparison_results = compare_candidates.invoke({
                    "candidate_ids": candidate_ids,
                    "jd_text": raw_jd,
                })
                # Apply ranking from comparison
                ranking: list[str] = comparison_results.get("ranking", candidate_ids)
                id_order = {cid: i for i, cid in enumerate(ranking)}
                shortlist.sort(key=lambda c: id_order.get(c["candidate_id"], 99))
            except Exception as exc:
                logger.warning("rank_candidates: Round 3 comparison failed — %s", exc)

        round_history.append(_snapshot(shortlist, 3))
        update = {
            "candidate_shortlist": shortlist,
            "current_round": 4,
            "round_history": round_history,
        }
        if comparison_results is not None:
            update["comparison_results"] = comparison_results
        return update
