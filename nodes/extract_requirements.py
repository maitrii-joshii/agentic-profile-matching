"""
nodes/extract_requirements.py
==============================
Extract Requirements node — Phase 5.2.

Calls the ``extract_requirements`` tool and stores the structured
requirements in ``state["parsed_requirements"]``.

On refinement loops (``state["refinement_requested"] == True``), existing
requirements are MERGED with the new ones rather than replaced — skills
already present are updated in-place, new skills are appended.

State writes: parsed_requirements
"""

from __future__ import annotations

import logging

from state import AgentState

logger = logging.getLogger(__name__)


from langchain_core.messages import HumanMessage

def run(state: AgentState) -> dict:
    """Call extract_requirements or refine_requirements tools.

    Returns a partial state update: ``{"parsed_requirements": [...]}``.
    """
    from tools.requirements import extract_requirements, refine_requirements

    refinement: bool = state.get("refinement_requested", False)

    if refinement:
        # Find the latest instruction from the human
        messages = state.get("messages", [])
        latest_instruction = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_instruction = str(msg.content)
                break

        existing_reqs = state.get("parsed_requirements", [])
        
        logger.info("extract_requirements node: refining requirements based on instruction: %r", latest_instruction[:50])
        result = refine_requirements.invoke({
            "current_reqs": {"must_have": [r for r in existing_reqs if r.get("category") == "must_have"], 
                             "nice_to_have": [r for r in existing_reqs if r.get("category") == "nice_to_have"]},
            "instruction": latest_instruction
        })
        
        all_refined = result.get("must_have", []) + result.get("nice_to_have", [])
        logger.info("extract_requirements node: refinement produced %d total requirements.", len(all_refined))
        
        return {"parsed_requirements": all_refined, "refinement_requested": False}

    # Base case: full extraction from scratch
    raw_jd: str = state.get("raw_jd", "")
    if not raw_jd or not raw_jd.strip():
        logger.warning("extract_requirements node: raw_jd is empty — skipping extraction.")
        return {"parsed_requirements": state.get("parsed_requirements", [])}

    logger.info("extract_requirements node: extracting from JD (%d chars) …", len(raw_jd))
    result = extract_requirements.invoke(raw_jd)

    all_new = result.get("must_have", []) + result.get("nice_to_have", [])
    logger.info("extract_requirements node: %d requirements extracted.", len(all_new))
    return {"parsed_requirements": all_new}
