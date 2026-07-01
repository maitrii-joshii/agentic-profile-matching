"""
nodes/generate_report.py
=========================
Generate Report node — Phase 5.5 / 8.5.

Compiles per-candidate match reports from the final ``candidate_shortlist``
and produces ``final_recommendations`` — a list of hire/no-hire decisions
with full reasoning, strengths/gaps, and improvement suggestions for borderline
candidates.

State writes: final_recommendations
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from state import AgentState

logger = logging.getLogger(__name__)

# Score thresholds for hire recommendations
_STRONG_HIRE = 85.0
_HIRE = 70.0
_BORDERLINE = 55.0
# Below BORDERLINE → no_hire


def _make_recommendation(candidate: dict) -> dict:
    """Build a recommendation record for one candidate."""
    score: float = candidate.get("overall_score", 0.0)

    if score >= _STRONG_HIRE:
        decision = "strong_hire"
        summary = f"Exceptional match (score {score:.0f}/100). Strongly recommend proceeding."
    elif score >= _HIRE:
        decision = "hire"
        summary = f"Good match (score {score:.0f}/100). Recommend proceeding to next stage."
    elif score >= _BORDERLINE:
        decision = "borderline"
        summary = (
            f"Partial match (score {score:.0f}/100). Consider if pipeline is thin. "
            "See improvement suggestions below."
        )
    else:
        decision = "no_hire"
        summary = f"Insufficient match (score {score:.0f}/100). Does not meet minimum threshold."

    # Improvement suggestions for borderline candidates
    suggestions: list[str] = []
    if decision == "borderline":
        gaps = candidate.get("gaps", [])
        for gap in gaps[:3]:
            suggestions.append(f"Candidate could strengthen: {gap}")

    return {
        "candidate_id": candidate["candidate_id"],
        "name": candidate.get("name", candidate["candidate_id"]),
        "overall_score": score,
        "hire_recommendation": decision,
        "summary": summary,
        "strengths": candidate.get("strengths", []),
        "gaps": candidate.get("gaps", []),
        "reasoning": candidate.get("reasoning", ""),
        "improvement_suggestions": suggestions,
        "skill_match": candidate.get("skill_match", {}),
        "experience_fit": candidate.get("experience_fit", 0.0),
    }


def _detect_refinement_changes(round_history: list[dict], new_shortlist: list[dict]) -> str:
    """Compare the new ranking with the pre-refinement ranking."""
    # Find all round 3 (final) snapshots
    final_snapshots = [h for h in round_history if h.get("round") == 3]
    if len(final_snapshots) < 2:
        return ""  # Not a refinement, or no previous final round to compare

    prev_snapshot = final_snapshots[-2]["candidates"]
    prev_ranking = {c["candidate_id"]: i + 1 for i, c in enumerate(prev_snapshot)}
    new_ranking = {c["candidate_id"]: i + 1 for i, c in enumerate(new_shortlist)}

    changes = []
    for cid, new_rank in new_ranking.items():
        if cid in prev_ranking:
            old_rank = prev_ranking[cid]
            if new_rank < old_rank:
                changes.append(f"**{cid.replace('_', ' ').title()}** moved UP from #{old_rank} to #{new_rank}")
            elif new_rank > old_rank:
                changes.append(f"**{cid.replace('_', ' ').title()}** moved DOWN from #{old_rank} to #{new_rank}")

    if not changes:
        return "\n*Ranking remained unchanged after refinement.*\n"
    
    return "\n### 🔄 Refinement Delta\n" + "\n".join(f"- {c}" for c in changes) + "\n"

def run(state: AgentState) -> dict:
    """Compile per-candidate match reports and set ``final_recommendations``.

    Also appends a summary ``AIMessage`` to the conversation so the recruiter
    sees results in the chat history.

    Returns a partial state update with ``final_recommendations`` and the
    summary message appended to ``messages``.
    """
    shortlist: list[dict] = state.get("candidate_shortlist", [])

    if not shortlist:
        logger.warning("generate_report: empty shortlist — no recommendations to generate.")
        msg = AIMessage(content="I couldn't find any candidates matching those requirements. Please try refining your criteria or uploading more resumes.")
        return {
            "final_recommendations": [],
            "needs_human_feedback": True,
            "messages": [msg],
        }

    recommendations = [_make_recommendation(c) for c in shortlist]

    # Build a human-readable summary for the chat
    lines = ["## 📋 Candidate Match Report\n"]
    
    delta_text = _detect_refinement_changes(state.get("round_history", []), shortlist)
    if delta_text:
        lines.append(delta_text + "\n")

    for i, rec in enumerate(recommendations, 1):
        emoji = {"strong_hire": "🟢", "hire": "🔵", "borderline": "🟡", "no_hire": "🔴"}.get(
            rec["hire_recommendation"], "⚪"
        )
        lines.append(
            f"{i}. {emoji} **{rec['name']}** — Score: {rec['overall_score']:.0f}/100 "
            f"({rec['hire_recommendation'].replace('_', ' ').title()})"
        )
        if rec["strengths"]:
            lines.append(f"   ✅ {rec['strengths'][0]}")
        if rec["gaps"]:
            lines.append(f"   ⚠️ Gap: {rec['gaps'][0]}")

    summary_text = "\n".join(lines)
    summary_msg = AIMessage(content=summary_text)

    logger.info(
        "generate_report: %d recommendations generated (%d hire/strong_hire).",
        len(recommendations),
        sum(1 for r in recommendations if r["hire_recommendation"] in ("hire", "strong_hire")),
    )

    return {
        "final_recommendations": recommendations,
        "needs_human_feedback": True,
        "messages": [summary_msg],
    }
