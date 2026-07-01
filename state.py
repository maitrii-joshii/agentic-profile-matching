"""
state.py — Agent State & Data Models
=====================================
Shared TypedDicts used by all LangGraph nodes and tools.

Phase implementation status:
  1.1  Requirement        ✅
  1.2  CandidateScore     ✅
  1.3  AgentState         ✅
"""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Phase 1.1 — Requirement model ─────────────────────────────────────────────

class Requirement(TypedDict):
    """
    A single skill or qualification extracted from a job description.

    Attributes
    ----------
    skill : str
        Human-readable skill or qualification name.
        Examples: "React", "Python", "AWS", "Bachelor's degree"
    category : Literal["must_have", "nice_to_have"]
        Whether the requirement is mandatory (must_have) or preferred
        (nice_to_have). Used to weight candidate scoring.
    min_years : Optional[int]
        Minimum years of experience required for this skill.
        ``None`` means no explicit experience threshold was stated.
    """

    skill: str
    category: Literal["must_have", "nice_to_have"]
    min_years: Optional[int]


# ── Phase 1.2 — CandidateScore model ──────────────────────────────────────────

class CandidateScore(TypedDict):
    """
    Scoring result for a single candidate against the active job description.

    Attributes
    ----------
    candidate_id : str
        Unique identifier derived from the resume filename stem.
        Example: ``"alice_chen_frontend"``
    name : str
        Candidate's full name as parsed from the resume.
    overall_score : float
        Composite match score in the range 0–100.
        Weighted combination of skill_match and experience_fit.
    skill_match : dict[str, float]
        Per-skill scores keyed by skill name, each in the range 0–1.
        Example: ``{"React": 1.0, "TypeScript": 0.8, "GraphQL": 0.5}``
    experience_fit : float
        Score (0–1) representing how well the candidate's total years of
        experience aligns with the JD's required experience range.
    strengths : list[str]
        Free-text bullets describing areas where the candidate clearly excels.
    gaps : list[str]
        Free-text bullets describing missing or weak skills relative to the JD.
    reasoning : str
        LLM-generated paragraph explaining the overall hire/no-hire assessment,
        used to answer recruiter "why" queries (explainability).
    """

    candidate_id: str
    name: str
    overall_score: float          # 0 – 100
    skill_match: dict[str, float] # skill → score (0–1)
    experience_fit: float         # 0 – 1
    strengths: list[str]
    gaps: list[str]
    reasoning: str


# ── Phase 1.3 — AgentState ────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state object passed between all LangGraph nodes.

    Every node receives the full state and returns a *partial* update dict
    containing only the keys it writes to — LangGraph merges updates
    automatically.

    Conversation
    ------------
    messages : Annotated[list[BaseMessage], add_messages]
        Full chat history between the recruiter and the agent.
        Uses the ``add_messages`` reducer so new messages are *appended*
        rather than overwriting the whole list on each state update.

    Job Description
    ---------------
    raw_jd : str
        Original job description text as supplied by the recruiter (pasted
        inline or extracted from an uploaded file).
    parsed_requirements : list[Requirement]
        Structured list of must-have and nice-to-have requirements extracted
        from ``raw_jd`` by the ``extract_requirements`` tool.

    Candidates
    ----------
    candidate_shortlist : list[CandidateScore]
        Ranked list of candidates currently under consideration.
        Updated after each screening round.
    comparison_results : Optional[dict]
        Output of ``compare_candidates`` when a head-to-head comparison has
        been requested. ``None`` until Phase 8 Final round.

    Screening round tracking
    ------------------------
    current_round : int
        Active screening round: 1 (broad), 2 (deep), or 3 (final/head-to-head).
    round_history : list[dict]
        Snapshot of ``candidate_shortlist`` saved at the end of each round,
        enabling explainability queries and refinement diffing.

    Decision
    --------
    final_recommendations : Optional[list[dict]]
        Per-candidate hire/no-hire decisions with full reasoning, populated
        by the ``generate_report`` node. ``None`` until the report is generated.

    Control flow
    ------------
    needs_human_feedback : bool
        ``True`` when the agent is paused and waiting for recruiter input
        (e.g., after presenting ranked results).
    refinement_requested : bool
        ``True`` when the recruiter has asked to adjust requirements
        mid-session, triggering a re-route back to ``extract_requirements``.
    """

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Job Description
    raw_jd: str
    parsed_requirements: list[Requirement]

    # Candidates
    candidate_shortlist: list[CandidateScore]
    comparison_results: Optional[dict]

    # Screening round tracking
    current_round: int       # 1 | 2 | 3
    round_history: list[dict]

    # Decision
    final_recommendations: Optional[list[dict]]

    # Control flow
    needs_human_feedback: bool
    refinement_requested: bool
