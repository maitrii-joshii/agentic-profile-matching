# Agentic Profile Matching — Project Context

## Overview

Build an **agentic profile-matching system** using **LangGraph** that automates candidate screening and ranking against job descriptions. The system must support conversational interaction, iterative refinement, multi-round evaluation, and explainable recommendations.

---

## Core Components

### 1. Agent Architecture (`matching_agent.py`) — 40 %

| Aspect | Details |
|---|---|
| **Framework** | LangGraph |
| **Entry file** | `matching_agent.py` |

#### Agent State

The state must track:

- Conversation history
- Parsed job requirements
- Candidate shortlist
- Ranking reasoning & decision history

#### Workflow (Graph Nodes)

```
START → Parse Job Description → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END
```

#### Tools

| Tool | Signature | Purpose |
|---|---|---|
| File System Tools | `read_resume(file_path: str)`, `list_resumes(directory: str)` | Read and list resume files from the local file system |
| RAG Search | `rag_search(query: str, top_k: int)` | Semantic search over indexed resumes using vector embeddings |
| Extract Requirements | `extract_requirements(jd: str)` | Parse JD into **must-have** vs **nice-to-have** requirements |
| Compare Candidates | `compare_candidates(candidate_ids: list)` | Head-to-head candidate comparison |
| Generate Interview Qs | `generate_interview_questions(candidate_id: str)` | Custom technical & behavioral screening questions |

---

### 2. Interactive / Conversational Features — 30 %

#### Natural Language Interface

Must handle queries like:

- *"Find me candidates with React and 3+ years experience."*
- *"Compare the top 3 matches side by side."*
- *"Why did John rank higher than Jane?"*

#### Iterative Refinement

- Dynamically update job requirements mid-conversation
- Re-rank candidates on revised criteria
- Explain ranking changes

---

### 3. Advanced Capabilities — 30 %

#### Multi-Round Screening

| Round | Action |
|---|---|
| **Round 1** | Screen full resume DB → select top 10 |
| **Round 2** | Deep analysis of top 10 (skills, experience, projects, fit) |
| **Final** | Hire / No-Hire recommendation with reasoning |

#### Explainability

- Detailed match reports per candidate
- Strength & gap highlights
- Improvement suggestions for borderline candidates

---

## Submission Deliverables

1. LangGraph-based agent implementation
2. State machine diagram (visual)
3. Chat interface (CLI **or** Streamlit / Gradio)
4. Test scenarios covering ≥ 5 conversation flows

---

## Key Technical Decisions to Make

- **LLM provider & model** — OpenAI / Google / local model
- **Vector store** for RAG search (FAISS, Chroma, Pinecone, etc.)
- **Resume data format** — PDF, JSON, plain text
- **Chat interface choice** — CLI vs Streamlit vs Gradio
- **State persistence** — in-memory vs checkpointed
