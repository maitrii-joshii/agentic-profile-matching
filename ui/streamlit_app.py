"""
ui/streamlit_app.py
====================
Streamlit Chat Interface — Phase 7.

Implements the full recruiter-facing chat UI for the Agentic Profile Matcher.
Features:
  - Rich dark-themed chat interface
  - JD text paste + file upload (PDF/TXT)
  - Candidate cards with score bars and hire/no-hire badges
  - Comparison table for head-to-head analysis
  - Quick-action feedback buttons (Refine / Compare / Approve / Start Over)
  - Sidebar: settings, resume status, round history
  - Loading spinners with step-by-step progress
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import time
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Agentic Profile Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #a5b4fc;
}

/* ── Header ── */
.apm-header {
    background: linear-gradient(90deg, rgba(139,92,246,0.15) 0%, rgba(236,72,153,0.1) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.apm-header h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.apm-header .subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Chat messages ── */
.stChatMessage {
    background: rgba(30, 27, 75, 0.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(10px);
}
.stChatMessage[data-testid="chat-message-user"] {
    background: rgba(139, 92, 246, 0.12) !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
}

/* ── Candidate card ── */
.candidate-card {
    background: rgba(30, 27, 75, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}
.candidate-card:hover {
    border-color: rgba(139, 92, 246, 0.6);
}
.candidate-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
}
.candidate-id {
    font-size: 0.75rem;
    color: #64748b;
    font-family: monospace;
}

/* ── Score bar ── */
.score-bar-container {
    background: rgba(15, 12, 41, 0.6);
    border-radius: 99px;
    height: 8px;
    width: 100%;
    margin: 0.5rem 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}

/* ── Hire badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-strong-hire { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-hire        { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-borderline  { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-no-hire     { background: rgba(239, 68, 68, 0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Action buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3) !important;
}

/* ── Chat input ── */
.stChatInputContainer {
    background: rgba(15, 12, 41, 0.9) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
}

/* ── Round pill ── */
.round-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 99px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #a78bfa;
    font-weight: 600;
}

/* ── Divider ── */
hr {
    border-color: rgba(139, 92, 246, 0.15) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(30, 27, 75, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ───────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "agent_state": None,           # last full AgentState
        "chat_history": [],            # list of {"role": "user"|"assistant", "content": str}
        "processing": False,
        "show_comparison": False,
        "round_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ── Agent import (lazy to avoid slow startup) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_agent():
    from matching_agent import agent
    return agent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_colour(score: float) -> str:
    if score >= 85:
        return "#10b981"
    elif score >= 70:
        return "#3b82f6"
    elif score >= 55:
        return "#f59e0b"
    return "#ef4444"


def _badge_html(decision: str) -> str:
    labels = {
        "strong_hire": ("🟢", "Strong Hire", "strong-hire"),
        "hire":        ("🔵", "Hire",        "hire"),
        "borderline":  ("🟡", "Borderline",  "borderline"),
        "no_hire":     ("🔴", "No Hire",     "no-hire"),
    }
    emoji, text, css = labels.get(decision, ("⚪", decision, "no-hire"))
    return f'<span class="badge badge-{css}">{emoji} {text}</span>'


def _score_bar_html(score: float) -> str:
    colour = _score_colour(score)
    return f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width:{score}%; background:{colour};"></div>
    </div>
    """


def _render_candidate_card(rec: dict, rank: int) -> None:
    score = rec.get("overall_score", 0)
    colour = _score_colour(score)
    decision = rec.get("hire_recommendation", "no_hire")

    st.markdown(f"""
    <div class="candidate-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="candidate-name">#{rank} &nbsp; {rec.get("name", rec.get("candidate_id"))}</div>
                <div class="candidate-id">{rec.get("candidate_id")}</div>
            </div>
            <div style="text-align:right;">
                {_badge_html(decision)}
                <div style="font-size:1.4rem; font-weight:700; color:{colour}; margin-top:4px;">{score:.0f}<span style="font-size:0.9rem; color:#64748b;">/100</span></div>
            </div>
        </div>
        {_score_bar_html(score)}
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"Details — {rec.get('name', rec.get('candidate_id'))}", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Strengths**")
            for s in rec.get("strengths", [])[:5]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("**⚠️ Gaps**")
            for g in rec.get("gaps", [])[:5]:
                st.markdown(f"- {g}")

        if rec.get("reasoning"):
            st.markdown("**🧠 Reasoning**")
            st.markdown(f"> {rec['reasoning']}")

        if rec.get("improvement_suggestions"):
            st.markdown("**💡 Improvement Suggestions**")
            for suggestion in rec["improvement_suggestions"]:
                st.markdown(f"- {suggestion}")

        skill_match = rec.get("skill_match", {})
        if skill_match:
            st.markdown("**🔧 Skill Match**")
            skill_data = {
                "Skill": list(skill_match.keys()),
                "Score": [f"{v * 100:.0f}%" for v in skill_match.values()],
            }
            st.dataframe(skill_data, use_container_width=True, hide_index=True)


def _render_comparison_table(comparison: dict, shortlist: list[dict]) -> None:
    """Render a side-by-side comparison matrix."""
    if not comparison:
        return

    st.markdown("### 📊 Head-to-Head Comparison")
    matrix = comparison.get("matrix", {})
    ranking = comparison.get("ranking", [])

    if ranking:
        st.markdown("**🏆 Ranking:**")
        for i, cid in enumerate(ranking, 1):
            medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i - 1, f"#{i}")
            st.markdown(f"{medal} `{cid}`")

    if comparison.get("summary"):
        st.info(comparison["summary"])

    if matrix:
        # Build comparison DataFrame
        import pandas as pd
        df = pd.DataFrame(matrix).T
        if not df.empty:
            st.dataframe(df, use_container_width=True)


def _render_round_history(round_history: list[dict]) -> None:
    if not round_history:
        return
    for snap in round_history:
        rnum = snap.get("round", "?")
        candidates = snap.get("candidates", [])
        label = {1: "🔍 Round 1 — Broad Screen", 2: "🔬 Round 2 — Deep Analysis", 3: "🏆 Final Round"}.get(rnum, f"Round {rnum}")
        with st.expander(label, expanded=False):
            for c in candidates:
                score = c.get("overall_score", 0)
                st.markdown(
                    f"`{c['candidate_id']}` &nbsp; — &nbsp; "
                    f"<span style='color:{_score_colour(score)}'>{score:.0f}/100</span>",
                    unsafe_allow_html=True
                )


def _extract_text_from_upload(uploaded_file) -> str:
    """Extract text from a PDF or TXT uploaded file."""
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif uploaded_file.name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            st.warning(f"PDF parsing error: {e}. Treating as raw bytes.")
            return uploaded_file.read().decode("utf-8", errors="replace")
    return uploaded_file.read().decode("utf-8", errors="replace")


def _run_agent_pipeline(user_input: str, existing_state: Optional[dict] = None) -> dict:
    """Invoke the LangGraph agent with the user input."""
    agent = _load_agent()

    if existing_state and existing_state.get("needs_human_feedback"):
        # Continuing conversation — append message to existing state
        existing_messages = list(existing_state.get("messages", []))
        existing_messages.append(HumanMessage(content=user_input))
        new_state = dict(existing_state)
        new_state["messages"] = existing_messages
        new_state["needs_human_feedback"] = False
        return agent.invoke(new_state)
    else:
        # Fresh run
        return agent.invoke({
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
        })


def _get_last_ai_message(agent_state: dict) -> str:
    """Extract the latest AIMessage content from state."""
    messages = agent_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🎯</div>
        <div style="font-weight:700; color:#a78bfa; font-size:1.1rem;">Profile Matcher</div>
        <div style="color:#64748b; font-size:0.75rem; margin-top:0.25rem;">Powered by LangGraph + DeepSeek</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status metrics
    agent_state = st.session_state.agent_state
    shortlist = agent_state.get("candidate_shortlist", []) if agent_state else []
    recs = agent_state.get("final_recommendations", []) if agent_state else []
    current_round = agent_state.get("current_round", 0) if agent_state else 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Candidates", len(shortlist))
    with c2:
        round_label = {1: "R1", 2: "R2", 3: "R3", 4: "Done"}.get(current_round, "—")
        st.metric("Round", round_label)

    if recs:
        hire_count = sum(1 for r in recs if r.get("hire_recommendation") in ("hire", "strong_hire"))
        st.metric("Hire Recommendations", f"{hire_count}/{len(recs)}")

    st.divider()

    # Round history
    if agent_state and agent_state.get("round_history"):
        st.markdown("**📋 Screening History**")
        _render_round_history(agent_state["round_history"])
        st.divider()

    # Quick actions
    if agent_state and agent_state.get("needs_human_feedback"):
        st.markdown("**⚡ Quick Actions**")

        if st.button("✅ Approve Results", use_container_width=True, key="btn_approve"):
            st.session_state.chat_history.append({"role": "user", "content": "Looks good, approve these results."})
            with st.spinner("Finalising..."):
                result = _run_agent_pipeline("Looks good, approve these results.", agent_state)
                st.session_state.agent_state = result
                ai_msg = _get_last_ai_message(result)
                if ai_msg:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_msg})
            st.rerun()

        if shortlist and len(shortlist) >= 2:
            if st.button("📊 Compare Top 3", use_container_width=True, key="btn_compare"):
                top_ids = [c["candidate_id"] for c in shortlist[:3]]
                msg = f"Compare these candidates side by side: {', '.join(top_ids)}"
                st.session_state.chat_history.append({"role": "user", "content": msg})
                with st.spinner("Comparing..."):
                    result = _run_agent_pipeline(msg, agent_state)
                    st.session_state.agent_state = result
                    ai_msg = _get_last_ai_message(result)
                    if ai_msg:
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_msg})
                st.rerun()

    if agent_state:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True, key="btn_restart"):
            st.session_state.agent_state = None
            st.session_state.chat_history = []
            st.session_state.round_history = []
            st.rerun()

        if shortlist:
            # Pick the top candidate for interview questions suggestion
            top_candidate = shortlist[0]
            top_name = top_candidate.get("name", top_candidate["candidate_id"])
            if st.button(f"🎤 Interview {top_name.split()[0]}", use_container_width=True, key="btn_interview"):
                msg = f"Generate interview questions for {top_candidate['candidate_id']}"
                st.session_state.chat_history.append({"role": "user", "content": msg})
                with st.spinner(f"Generating interview questions for {top_name}..."):
                    result = _run_agent_pipeline(msg, agent_state)
                    st.session_state.agent_state = result
                    ai_msg = _get_last_ai_message(result)
                    if ai_msg:
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_msg})
                st.rerun()

    st.divider()
    st.markdown("""
    <div style="color:#475569; font-size:0.7rem; text-align:center;">
        LangGraph · ChromaDB · DeepSeek
    </div>
    """, unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="apm-header">
    <div style="font-size:2.5rem;">🎯</div>
    <div>
        <h1>Agentic Profile Matcher</h1>
        <p class="subtitle">AI-powered candidate screening · Multi-round analysis · Explainable recommendations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── JD Upload / Input area ──────────────────────────────────────────────────────
if not st.session_state.agent_state:
    st.markdown("### 📝 Start a New Search")

    input_mode = st.radio(
        "How would you like to provide the Job Description?",
        ["Paste JD text", "Upload JD file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_jd_text = ""
    if input_mode == "Upload JD file":
        uploaded_file = st.file_uploader(
            "Upload Job Description (PDF or TXT)",
            type=["pdf", "txt"],
            key="jd_uploader",
        )
        if uploaded_file:
            uploaded_jd_text = _extract_text_from_upload(uploaded_file)
            with st.expander("📄 Extracted JD Text", expanded=False):
                st.text_area("JD Content", uploaded_jd_text, height=200, disabled=True)
    else:
        uploaded_jd_text = st.text_area(
            "Paste your Job Description here…",
            height=160,
            placeholder="We are looking for a Senior React Engineer with 3+ years of experience...",
            key="jd_text_input",
        )

    col1, col2 = st.columns([1, 4])
    with col1:
        start_btn = st.button("🚀 Find Candidates", type="primary", use_container_width=True, key="btn_start")

    if start_btn and uploaded_jd_text.strip():
        st.session_state.chat_history.append({"role": "user", "content": uploaded_jd_text})
        with st.spinner("🔍 Analysing JD and searching candidates..."):
            progress = st.progress(0, "Parsing job description…")
            time.sleep(0.3)
            progress.progress(20, "Extracting requirements…")
            time.sleep(0.3)
            progress.progress(40, "Searching resume database…")
            try:
                result = _run_agent_pipeline(uploaded_jd_text)
                progress.progress(70, "Ranking candidates (Round 1)…")
                time.sleep(0.2)
                progress.progress(90, "Generating report…")
                time.sleep(0.2)
                progress.progress(100, "Done!")
                st.session_state.agent_state = result
                ai_content = _get_last_ai_message(result)
                if ai_content:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_content})
                else:
                    recs = result.get("final_recommendations", [])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Analysis complete. Found **{len(recs)}** candidate(s). See the cards below."
                    })
            except Exception as e:
                st.error(f"❌ Agent error: {e}")
                st.stop()
        st.rerun()
    elif start_btn:
        st.warning("Please provide a Job Description first.")

# ── Chat History ───────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 Conversation")

    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                content = entry["content"]
                if len(content) > 400:
                    with st.expander("Job Description (click to expand)"):
                        st.markdown(content)
                else:
                    st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🎯"):
                st.markdown(entry["content"])

# ── Candidate Results ──────────────────────────────────────────────────────────
agent_state = st.session_state.agent_state
if agent_state:
    recs = agent_state.get("final_recommendations") or []
    shortlist = agent_state.get("candidate_shortlist", [])
    comparison = agent_state.get("comparison_results")
    req = agent_state.get("parsed_requirements", [])

    if recs:
        st.markdown("---")
        st.markdown("### 🏆 Candidate Recommendations")

        # Summary metrics row
        m1, m2, m3, m4 = st.columns(4)
        strong = sum(1 for r in recs if r.get("hire_recommendation") == "strong_hire")
        hire = sum(1 for r in recs if r.get("hire_recommendation") == "hire")
        border = sum(1 for r in recs if r.get("hire_recommendation") == "borderline")
        nohire = sum(1 for r in recs if r.get("hire_recommendation") == "no_hire")
        m1.metric("🟢 Strong Hire", strong)
        m2.metric("🔵 Hire", hire)
        m3.metric("🟡 Borderline", border)
        m4.metric("🔴 No Hire", nohire)

        st.markdown("")

        # Candidate cards
        for i, rec in enumerate(recs, 1):
            _render_candidate_card(rec, i)

        # Comparison table
        if comparison:
            st.markdown("---")
            _render_comparison_table(comparison, shortlist)

    elif shortlist:
        st.markdown("---")
        st.info(f"🔍 Found {len(shortlist)} candidate(s) in shortlist. Screening in progress…")

    # Requirements summary
    if req:
        with st.expander("📋 Extracted Requirements", expanded=False):
            must = [r for r in req if r.get("category") == "must_have"]
            nice = [r for r in req if r.get("category") == "nice_to_have"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Must Have**")
                for r in must:
                    yrs = f" · {r['min_years']}yr+" if r.get("min_years") else ""
                    st.markdown(f"- `{r['skill']}`{yrs}")
            with c2:
                st.markdown("**Nice to Have**")
                for r in nice:
                    st.markdown(f"- `{r['skill']}`")

# ── Chat Input for ongoing conversation ────────────────────────────────────────
if agent_state and agent_state.get("needs_human_feedback"):
    st.markdown("---")

    # Feedback hint row
    with st.container():
        st.markdown(
            "💡 **You can:** refine requirements · explain rankings · "
            "📊 compare candidates · 🎤 interview a candidate · approve · or start over"
        )

    user_chat = st.chat_input(
        "Type your feedback (e.g. 'Also require Python', 'Why is Alice ranked first?', 'Generate interview questions for Alice', 'Approve')…",
        key="chat_input",
    )

    if user_chat:
        st.session_state.chat_history.append({"role": "user", "content": user_chat})
        with st.spinner("🤔 Processing your feedback…"):
            try:
                result = _run_agent_pipeline(user_chat, agent_state)
                st.session_state.agent_state = result
                ai_msg = _get_last_ai_message(result)
                if ai_msg:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_msg})
            except Exception as e:
                st.error(f"❌ Error: {e}")
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.agent_state and not st.session_state.chat_history:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:rgba(30,27,75,0.6); border:1px solid rgba(139,92,246,0.2); border-radius:12px; padding:1.25rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
            <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.25rem;">Semantic Search</div>
            <div style="color:#64748b; font-size:0.85rem;">BGE embeddings + ChromaDB for high-accuracy resume matching</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:rgba(30,27,75,0.6); border:1px solid rgba(139,92,246,0.2); border-radius:12px; padding:1.25rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">🏆</div>
            <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.25rem;">3-Round Screening</div>
            <div style="color:#64748b; font-size:0.85rem;">Broad screen → Deep analysis → Head-to-head comparison</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:rgba(30,27,75,0.6); border:1px solid rgba(139,92,246,0.2); border-radius:12px; padding:1.25rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">💬</div>
            <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.25rem;">Conversational Refinement</div>
            <div style="color:#64748b; font-size:0.85rem;">Refine requirements and re-rank mid-conversation</div>
        </div>
        """, unsafe_allow_html=True)
