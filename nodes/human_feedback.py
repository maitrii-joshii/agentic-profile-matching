"""
nodes/human_feedback.py
========================
Human Feedback Loop node — Phase 5.6 / 9.1 / 10+.

Uses a small LLM call (INTENT_DETECTION_SYSTEM) to classify the recruiter's
message into one of:

  refinement          → recruiter wants to change requirements
  approved            → recruiter accepts the recommendations
  explain             → recruiter asks "why" / "explain"
  rerank              → recruiter wants to re-rank existing candidates
  new_search          → recruiter wants a fresh search
  interview_questions → recruiter wants interview questions for a candidate
  unknown             → anything else

Also handles explainability queries (Phase 8.6) and interview question
generation (Phase 10+).

State writes: needs_human_feedback, refinement_requested, messages
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import AgentState

logger = logging.getLogger(__name__)


# ── LLM-based intent detection ─────────────────────────────────────────────────

def _detect_intent_llm(text: str, shortlist: list[dict]) -> tuple[str, str | None]:
    """Call a small LLM to classify the recruiter's message.

    Returns a tuple of (intent_label, candidate_id_or_None).
    Falls back to 'unknown' on any error so the pipeline never crashes.
    """
    from prompts.system import call_llm_with_retry, get_llm
    from prompts.templates import INTENT_DETECTION_SYSTEM, INTENT_DETECTION_USER

    candidate_list = ", ".join(
        c.get("name", c["candidate_id"]) for c in shortlist
    ) or "none"

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=INTENT_DETECTION_SYSTEM),
            HumanMessage(content=INTENT_DETECTION_USER.format(
                candidate_list=candidate_list,
                message=text,
            )),
        ]
        response = call_llm_with_retry(llm, messages)
        raw = response.content.strip()

        # Strip markdown fences if any
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)
        intent = parsed.get("intent", "unknown")
        candidate_id = parsed.get("candidate_id") or None
        logger.info("_detect_intent_llm: intent=%r candidate_id=%r", intent, candidate_id)
        return intent, candidate_id

    except Exception as exc:
        logger.warning("_detect_intent_llm: failed (%s). Defaulting to 'unknown'.", exc)
        return "unknown", None


def _resolve_candidate_id(candidate_hint: str | None, shortlist: list[dict]) -> str | None:
    """Match a fuzzy candidate name/hint to a real candidate_id in the shortlist."""
    if not candidate_hint or not shortlist:
        return None

    hint_lower = candidate_hint.lower().replace("_", " ").strip()

    for c in shortlist:
        cid_lower = c["candidate_id"].lower().replace("_", " ")
        name_lower = c.get("name", "").lower()
        if hint_lower in cid_lower or hint_lower in name_lower:
            return c["candidate_id"]
        # Partial first-name match
        if hint_lower.split()[0] in cid_lower or hint_lower.split()[0] in name_lower:
            return c["candidate_id"]

    # Default to top candidate if we got an intent but can't resolve the name
    return shortlist[0]["candidate_id"] if shortlist else None


# ── Explainability response ────────────────────────────────────────────────────

def _build_explain_response(text: str, state: AgentState) -> str:
    """Generate an explainability response from round_history and candidate reasoning."""
    shortlist: list[dict] = state.get("candidate_shortlist", [])
    round_history: list[dict] = state.get("round_history", [])

    # Try to identify which candidate(s) the recruiter is asking about
    mentioned = []
    for c in shortlist:
        cid = c["candidate_id"].replace("_", " ").lower()
        name = c.get("name", "").lower()
        if cid in text.lower() or name in text.lower():
            mentioned.append(c)

    if not mentioned:
        mentioned = shortlist[:2]   # default to top 2

    lines = ["### 🔍 Explanation\n"]

    for c in mentioned:
        lines.append(f"**{c.get('name', c['candidate_id'])}** (score {c.get('overall_score', 0):.0f}/100)")
        if c.get("reasoning"):
            lines.append(c["reasoning"])
        if c.get("strengths"):
            lines.append("Strengths: " + "; ".join(c["strengths"][:3]))
        if c.get("gaps"):
            lines.append("Gaps: " + "; ".join(c["gaps"][:3]))
        lines.append("")

    if round_history:
        lines.append(f"*Screening ran {len(round_history)} round(s). "
                     f"Started with {len(round_history[0].get('candidates', []))} candidates.*")

    return "\n".join(lines)


# ── Interview questions response ───────────────────────────────────────────────

def _build_interview_response(candidate_id: str, state: AgentState) -> str:
    """Invoke generate_interview_questions and format results as markdown."""
    from tools.interview import generate_interview_questions

    raw_jd = state.get("raw_jd", "")

    try:
        result = generate_interview_questions.invoke({
            "candidate_id": candidate_id,
            "jd_text": raw_jd,
        })
    except Exception as exc:
        logger.error("_build_interview_response: tool failed — %s", exc)
        return f"❌ Could not generate interview questions for `{candidate_id}`: {exc}"

    cname = candidate_id.replace("_", " ").title()
    lines = [f"### 🎤 Interview Questions — {cname}\n"]

    technical = result.get("technical", [])
    if technical:
        lines.append("#### 🔧 Technical Questions")
        for i, q in enumerate(technical, 1):
            lines.append(f"**{i}. {q['question']}**")
            if q.get("rationale"):
                lines.append(f"   *{q['rationale']}*")
            lines.append("")

    behavioural = result.get("behavioural", [])
    if behavioural:
        lines.append("#### 🤝 Behavioural Questions")
        for i, q in enumerate(behavioural, 1):
            lines.append(f"**{i}. {q['question']}**")
            if q.get("rationale"):
                lines.append(f"   *{q['rationale']}*")
            lines.append("")

    gap_probing = result.get("gap_probing", [])
    if gap_probing:
        lines.append("#### ⚠️ Gap-Probing Questions")
        for i, q in enumerate(gap_probing, 1):
            lines.append(f"**{i}. {q['question']}**")
            if q.get("gap"):
                lines.append(f"   *Gap being probed: {q['gap']}*")
            lines.append("")

    return "\n".join(lines)


# ── Main node ──────────────────────────────────────────────────────────────────

def run(state: AgentState) -> dict:
    """Parse latest recruiter message and set routing flags.

    Returns a partial state update setting the appropriate flags and
    appending any explainability/interview response to messages.
    """
    messages = state.get("messages", [])
    shortlist: list[dict] = state.get("candidate_shortlist", [])

    # Find the most recent human message
    latest_text = ""
    human_count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_count += 1
            if not latest_text:
                latest_text = msg.content if isinstance(msg.content, str) else str(msg.content)

    # If there is only 1 human message, it's the initial job description.
    # We should NOT run intent detection on it, just pause and wait for real feedback.
    if human_count <= 1:
        logger.info("human_feedback: Initial run complete. Pausing for human input.")
        return {"needs_human_feedback": True}

    if not latest_text:
        logger.info("human_feedback: no new human message — awaiting input.")
        return {"needs_human_feedback": True}

    # ── LLM-based intent classification ──────────────────────────────────────
    intent, raw_candidate_hint = _detect_intent_llm(latest_text, shortlist)
    logger.info("human_feedback: intent=%r hint=%r", intent, raw_candidate_hint)

    update: dict = {"needs_human_feedback": False}

    if intent == "interview_questions":
        cid = _resolve_candidate_id(raw_candidate_hint, shortlist)
        if not cid:
            update["needs_human_feedback"] = True
            update["messages"] = [AIMessage(
                content=(
                    "I'd love to generate interview questions! "
                    "Please mention the candidate's name, e.g. *'Generate interview questions for Alice'*."
                )
            )]
        else:
            response_text = _build_interview_response(cid, state)
            update["messages"] = [AIMessage(content=response_text)]
            update["needs_human_feedback"] = True  # stay in feedback loop

    elif intent == "explain":
        response_text = _build_explain_response(latest_text, state)
        update["messages"] = [AIMessage(content=response_text)]
        update["needs_human_feedback"] = True   # stay in feedback loop

    elif intent == "approved":
        update["refinement_requested"] = False

    elif intent == "refinement":
        update["refinement_requested"] = True

    elif intent == "new_search":
        update["refinement_requested"] = True
        update["candidate_shortlist"] = []
        update["current_round"] = 1
        update["round_history"] = []
        update["final_recommendations"] = None

    elif intent == "rerank":
        update["current_round"] = 1     # re-enters rank_candidates from Round 1

    elif intent == "compare":
        logger.info("human_feedback: processing compare intent.")
        if ":" in latest_text:
            cids_str = latest_text.split(":", 1)[1]
            cids = [c.strip() for c in cids_str.split(",") if c.strip()]
        else:
            cids = [c["candidate_id"] for c in shortlist[:3]]
            
        if len(cids) >= 2:
            try:
                from tools.comparison import compare_candidates
                result = compare_candidates.invoke({
                    "candidate_ids": cids,
                    "jd_text": state.get("raw_jd", ""),
                })
                head_to_head = result.get("head_to_head", "")
                summary = result.get("summary", "")
                response_text = f"### 📊 Candidate Comparison\n**Summary:** {summary}\n\n**Head-to-Head:** {head_to_head}"
            except Exception as exc:
                logger.error("human_feedback: compare_candidates failed: %s", exc)
                response_text = f"❌ Comparison failed: {exc}"
        else:
            response_text = "I need at least 2 candidates to compare."
            
        update["messages"] = [AIMessage(content=response_text)]
        update["needs_human_feedback"] = True

    else:
        # Unknown — stay in feedback loop and acknowledge
        update["needs_human_feedback"] = True
        update["messages"] = [AIMessage(
            content=(
                "I'm not sure what you'd like to do. You can:\n"
                "- **Approve** the current recommendations\n"
                "- **Refine** requirements (e.g. 'add TypeScript as must-have')\n"
                "- **Explain** rankings (e.g. 'why is Alice ranked first?')\n"
                "- **Interview** a candidate (e.g. 'generate interview questions for Alice')\n"
                "- **Start over** with a new search\n"
                "- **Re-rank** top candidates"
            )
        )]

    return update
