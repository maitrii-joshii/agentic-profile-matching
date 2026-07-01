# Agentic Profile Matching — Edge Cases & Corner Scenarios

> Comprehensive catalogue of corner scenarios organised by system component, mapped to the phases in [implementationPlan.md](file:///c:/Users/MJ/Desktop/Projects/Airtribe/agentic-profile-matching/implementationPlan.md). Each entry includes the scenario, expected behaviour, and the handling strategy.

---

## 1. File System & Resume Loading (Phase 2)

### 1.1 File Access Errors

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| FS-01 | **Resume directory does not exist** | Agent reports missing directory, does not crash | Raise descriptive `FileNotFoundError`; suggest checking `RESUME_DIR` in `.env` |
| FS-02 | **Resume directory is empty** | Agent reports zero resumes found | Return empty list from `list_resumes`; downstream nodes surface "No resumes available" message |
| FS-03 | **Resume directory has no valid files** (e.g., only `.docx`, `.png`) | Agent reports zero supported resumes | Filter for `.pdf/.json/.txt` only; log skipped file extensions as warnings |
| FS-04 | **File permission denied** (OS-level read lock) | Skip file, continue with others | Wrap file reads in `try/except PermissionError`; log warning with file path |
| FS-05 | **Symlinks or circular directory references** | No infinite loop | Set `os.walk(followlinks=False)` or track visited inodes |
| FS-06 | **Extremely deep directory nesting** (100+ levels) | No stack overflow | Use iterative walk; set max depth limit (~20 levels) |

### 1.2 File Content Errors

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| FS-07 | **Corrupted PDF** (unreadable bytes, password-protected) | Skip file, log error | `try/except` around `pdfplumber.open()`; log "Skipping corrupted: {path}" |
| FS-08 | **PDF with scanned images only** (no extractable text) | Extract returns empty/minimal text | Detect empty extraction result; log "No text extracted from: {path}"; skip from indexing |
| FS-09 | **Invalid JSON file** (malformed syntax) | Skip file, log error | `try/except json.JSONDecodeError`; log and continue |
| FS-10 | **JSON file with unexpected schema** (valid JSON but no resume fields) | Treat as raw text | Fallback to `json.dumps(data)` as plain text; log schema mismatch warning |
| FS-11 | **Empty file** (0 bytes) | Skip file | Check `os.path.getsize(path) == 0` before processing; skip with warning |
| FS-12 | **Very large file** (100MB+ PDF) | No OOM crash | Set file size limit (~10MB); skip with "File too large: {path}" warning |
| FS-13 | **Non-UTF-8 encoded text file** (e.g., Latin-1, Shift-JIS) | Decode gracefully | Try `utf-8` first, fallback to `chardet` detection, then `latin-1` as last resort |
| FS-14 | **File extension doesn't match content** (`.txt` file is actually a PDF) | Handle gracefully | Detect file magic bytes; if extension mismatches content type, attempt correct parser |
| FS-15 | **Resume with only headers / no meaningful content** | Flag as low-quality | After extraction, check token count; if < 20 tokens, skip with warning |

---

## 2. Resume Ingestion & ChromaDB (Phase 2)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| ING-01 | **ChromaDB persist directory does not exist** | Auto-create | `chromadb.PersistentClient` auto-creates; verify with fallback `os.makedirs()` |
| ING-02 | **ChromaDB persist directory is read-only** | Descriptive error at startup | Check write permissions on startup; fail fast with clear message |
| ING-03 | **ChromaDB data is corrupted** (partial write from crash) | Rebuild index | Delete corrupt `chroma_data/` dir; re-run ingestion pipeline from scratch |
| ING-04 | **Duplicate resume indexed** (same file ingested twice) | No duplicate chunks | Use `collection.upsert()` with deterministic IDs (`{candidate_id}_chunk_{i}`); idempotent |
| ING-05 | **Candidate ID collision** (two different people, same generated ID) | Data mix-up | Generate IDs from file hash + name; add collision detection |
| ING-06 | **Resume produces zero chunks** (text too short to chunk) | Index as single chunk | If text < `chunk_size`, treat entire text as one chunk |
| ING-07 | **Resume produces 1000+ chunks** (very long document) | Index all, but flag | Log warning for documents with > 100 chunks; consider truncation |
| ING-08 | **BGE model download fails** (no internet, disk full) | Clear error on startup | Catch `OSError` / `HTTPError` on `SentenceTransformer()` init; suggest manual download |
| ING-09 | **ChromaDB collection already exists with different embedding dimensions** | Schema conflict | Delete and recreate collection; or version collection names (e.g., `resumes_v2`) |
| ING-10 | **Concurrent ingestion** (two processes writing simultaneously) | No data corruption | ChromaDB handles concurrency; but log warnings and consider file-based locking |

---

## 3. RAG Search (Phase 3)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| RAG-01 | **Empty vector store** (no resumes indexed) | Zero results, helpful message | Return empty list; agent surfaces "No resumes have been indexed yet. Run ingestion first." |
| RAG-02 | **Query returns zero matches** (no semantic similarity above threshold) | Empty shortlist with guidance | Return empty list; suggest broader search terms or loosened requirements |
| RAG-03 | **All top-k results are from the same candidate** | Over-representation | Deduplication already handles this; cap at 1 entry per `candidate_id` |
| RAG-04 | **top_k exceeds total candidates in store** | Return all available | ChromaDB returns `min(top_k, total_docs)`; no error |
| RAG-05 | **top_k = 0 or negative** | Sensible default | Clamp to `max(1, top_k)`; log warning if original value was invalid |
| RAG-06 | **Very long query string** (1000+ tokens) | May exceed embedding model's max length | Truncate query to model's max sequence length (512 tokens for BGE); log warning |
| RAG-07 | **Query in a different language** than resumes | Low relevance scores | BGE models are English-focused; return best available matches; log language mismatch hint |
| RAG-08 | **Special characters / SQL injection in query** | No injection | ChromaDB is not SQL-based; but sanitize input to prevent prompt injection via tool |
| RAG-09 | **ChromaDB connection lost mid-query** | Retry or fail gracefully | Wrap queries in `try/except`; retry once; if persistent, surface "Vector store unavailable" |
| RAG-10 | **Relevance scores are all very low** (< 0.3) | Warn user about weak matches | Add threshold check; if all scores < threshold, append "⚠ Low confidence matches" to response |

---

## 4. LLM Tools — Groq API (Phase 4)

### 4.1 API & Network Errors

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| LLM-01 | **Invalid or expired `GROQ_API_KEY`** | Fail fast on first call | Catch `AuthenticationError`; surface "Invalid Groq API key. Check .env file." |
| LLM-02 | **Groq rate limit exceeded** (429 status) | Retry with back-off | Exponential back-off: wait 1s → 2s → 4s; max 3 retries; then surface error |
| LLM-03 | **Groq server error** (500/503 status) | Retry then fail | Same retry strategy as LLM-02; surface "Groq service temporarily unavailable" |
| LLM-04 | **Network timeout** (no internet) | Fail with clear message | Catch `ConnectionError` / `Timeout`; surface "Cannot reach Groq API. Check internet connection." |
| LLM-05 | **Groq response is empty** (200 but no content) | Retry once | If response body is empty after retry, surface "Received empty response from LLM" |
| LLM-06 | **Model name is invalid** (typo in config) | Fail on first call | Catch `NotFoundError`; surface "Model '{name}' not found on Groq. Check configuration." |
| LLM-07 | **`GROQ_API_KEY` not set in environment** | Fail at startup | Check for key presence in app init; raise `ValueError` with setup instructions |

### 4.2 LLM Output Parsing Errors

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| LLM-08 | **LLM returns invalid JSON** (malformed structure) | Re-prompt or extract | Retry with stricter prompt ("Respond ONLY with valid JSON"); fallback to regex extraction |
| LLM-09 | **LLM returns extra commentary** around JSON | Parse JSON from noise | Use regex to extract JSON block from response (` ```json ... ``` ` or `{...}`) |
| LLM-10 | **LLM returns correct JSON but wrong schema** (missing fields) | Fill defaults | Validate against expected schema; fill missing fields with sensible defaults; log warning |
| LLM-11 | **LLM hallucinates candidate data** (invents names/skills not in resume) | Cross-validate | After LLM scoring, cross-reference mentioned skills against original resume text |
| LLM-12 | **LLM returns scores outside expected range** (e.g., score: 150) | Clamp values | Clamp `overall_score` to `[0, 100]`; log anomaly |
| LLM-13 | **LLM response exceeds `max_tokens`** (truncated mid-JSON) | Incomplete output | Detect incomplete JSON; retry with higher `max_tokens` or chunked approach |

### 4.3 extract_requirements Specific

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| LLM-14 | **JD has no clear skills** ("We need a great team player") | Empty must_have list | Return `must_have: []`; agent asks user to provide specific skills |
| LLM-15 | **JD is extremely short** (< 10 words) | Insufficient context | Agent responds "The job description is too brief. Please provide more details." |
| LLM-16 | **JD is extremely long** (5000+ words) | May exceed context window | Truncate JD to first 3000 tokens; summarise remainder; log truncation |
| LLM-17 | **JD in non-English language** | May misparse requirements | Detect language; if non-English, attempt translation or warn user |
| LLM-18 | **JD contains contradictory requirements** ("3+ years React, max 1 year experience") | Surface conflict | LLM should flag contradiction in output; agent highlights it for user review |

### 4.4 compare_candidates Specific

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| LLM-19 | **Single candidate ID passed** (nothing to compare) | Skip comparison | Return descriptive message: "Need at least 2 candidates to compare." |
| LLM-20 | **Invalid candidate ID** (not in shortlist) | Error with guidance | Validate IDs against `candidate_shortlist`; surface "Candidate '{id}' not found in current shortlist." |
| LLM-21 | **Comparing 20+ candidates** at once | May exceed context window | Cap at 10 candidates per comparison; split into groups if more |
| LLM-22 | **All candidates have identical scores** | Tie-breaking needed | Use secondary criteria (experience_fit, nice-to-have scores); explain tie in reasoning |

### 4.5 generate_interview_questions Specific

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| LLM-23 | **Candidate has no gaps** (perfect match) | No gap-probing questions | Return empty `gap_probing_questions` list; focus on depth-probing questions instead |
| LLM-24 | **Candidate profile is very thin** (minimal resume data) | Generic questions | Generate broader questions; note "Limited candidate data available" in response |

---

## 5. LangGraph Workflow Nodes (Phase 5)

### 5.1 Parse Job Description Node

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| WF-01 | **User provides file path instead of JD text** | Auto-detect and load | Check if input looks like a path; if so, call `read_resume(path)` to load content |
| WF-02 | **User sends empty message** | Prompt for JD | Agent responds "Please provide a job description to get started." |
| WF-03 | **User sends a question instead of a JD** ("How does this work?") | Handle as conversation | Detect non-JD intent; respond with help text instead of parsing |
| WF-04 | **User pastes JD with heavy HTML/markdown formatting** | Strip formatting | Clean HTML tags and markdown syntax; extract plain text |

### 5.2 Search Resumes Node

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| WF-05 | **No requirements extracted** (empty parsed_requirements) | Cannot search | Agent asks user to provide specific requirements before searching |
| WF-06 | **Requirements are extremely niche** ("15 years Rust + quantum computing") | Zero or very few matches | Return whatever matches exist; suggest loosening criteria if < 3 results |
| WF-07 | **Resume directory changed between indexing and search** (files deleted) | Stale index | `file_path` references may be broken; validate paths exist before returning results |

### 5.3 Rank Candidates Node

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| WF-08 | **Empty candidate shortlist** (nothing to rank) | Skip ranking | Surface "No candidates to rank. Try broadening your requirements." |
| WF-09 | **Only 1 candidate in shortlist** | Skip comparison, rank trivially | Return single candidate as #1; skip head-to-head comparison |
| WF-10 | **All candidates score identically** | Tie-breaking | Use secondary criteria; if still tied, present all as equal matches |
| WF-11 | **Candidate data is incomplete** (missing skills section) | Partial scoring | Score available dimensions; mark missing dimensions as "N/A"; lower confidence flag |

### 5.4 Generate Report Node

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| WF-12 | **No final recommendations exist** (pipeline interrupted) | Incomplete report | Generate partial report with available data; mark as "Incomplete — re-run recommended" |
| WF-13 | **Report generation exceeds LLM context** (too many candidates) | Truncation | Generate reports in batches (5 candidates at a time); merge results |

### 5.5 Human Feedback Loop Node

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| WF-14 | **User sends ambiguous feedback** ("hmm, maybe") | Ask for clarification | Respond "Would you like to refine requirements, compare candidates, or approve the results?" |
| WF-15 | **User sends unrelated message** ("What's the weather?") | Stay in context | Respond "I can help with candidate matching. Would you like to refine the results or try a new search?" |
| WF-16 | **User sends feedback before pipeline completes** | Race condition | Block feedback processing until current pipeline run finishes; queue the message |
| WF-17 | **User sends multiple conflicting refinements** in one message ("add React, remove React") | Detect conflict | Parse both instructions; detect contradiction; ask user to clarify |

---

## 6. LangGraph Agent Assembly (Phase 6)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| AG-01 | **Infinite refinement loop** (user keeps refining endlessly) | No infinite loop | Set max refinement iterations (~10); after limit, suggest starting a new session |
| AG-02 | **Conditional edge returns invalid node name** | Graph crash | Validate router return values; default to `END` if unrecognised |
| AG-03 | **State grows unbounded** (1000+ messages in history) | Memory exhaustion | Trim message history to last N messages (e.g., 50); archive older messages |
| AG-04 | **Node throws unhandled exception** | Pipeline crash | Wrap each node in `try/except`; on failure, set error state; route to human feedback with error message |
| AG-05 | **Graph compilation fails** (misconfigured edges) | Startup crash | Validate graph at startup; catch `ValueError` from LangGraph; log clear diagnostic |
| AG-06 | **Round counter exceeds expected max** (`current_round > 3`) | Unexpected state | Clamp `current_round` to `[1, 3]`; auto-terminate at round 3 |
| AG-07 | **State keys are missing** (partial state from interrupted run) | KeyError | Provide defaults for all state keys; use `state.get(key, default)` throughout |

---

## 7. Streamlit Chat Interface (Phase 7)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| UI-01 | **User uploads non-JD file** (image, video, ZIP) | Reject with message | Validate file extension; accept only `.pdf`, `.txt`, `.json`; show error for others |
| UI-02 | **User uploads empty file** | Reject with message | Check file size > 0; show "The uploaded file is empty" |
| UI-03 | **User uploads very large JD** (50MB PDF) | Reject with message | Set `st.file_uploader(max_upload_size_mb=5)`; show size limit error |
| UI-04 | **Session timeout / page refresh** | State loss | Store `AgentState` in `st.session_state`; persist across reruns. Warn user on refresh. |
| UI-05 | **Concurrent users** (multiple Streamlit sessions) | Shared state bleed | Each session gets its own `st.session_state`; no shared global state |
| UI-06 | **Browser back/forward navigation** | Unexpected state | Streamlit is single-page; disable or handle gracefully |
| UI-07 | **Rapid-fire message sending** (spam click Send) | Duplicate processing | Disable Send button during processing; use `st.spinner()` |
| UI-08 | **Very long agent response** (10,000+ chars) | UI clutter | Truncate display; use `st.expander("Show full response")` for overflow |
| UI-09 | **Agent response contains markdown / code** | Rendering issues | Use `st.markdown()` with `unsafe_allow_html=False`; sanitise output |
| UI-10 | **Streamlit port already in use** | App won't start | Catch `OSError`; suggest `streamlit run ... --server.port 8502` |

---

## 8. Multi-Round Screening (Phase 8)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| MR-01 | **Fewer than 10 candidates in corpus** (e.g., only 3 resumes) | Round 1 can't select "top 10" | Select all available; skip to Round 2 immediately; adjust messaging |
| MR-02 | **All candidates eliminated in Round 1** (no one meets must-haves) | Empty Round 2 | Skip remaining rounds; report "No candidates meet the minimum requirements." |
| MR-03 | **Tie for 10th place in Round 1** (candidates 10 & 11 have same score) | Arbitrary cut-off | Include all tied candidates; expand to top 11; note tie in report |
| MR-04 | **Round 2 deep analysis changes scores drastically** | Rankings flip | Expected behaviour; log pre- and post-analysis scores in `round_history` |
| MR-05 | **User skips rounds** ("Just give me a hire/no-hire for everyone") | Non-standard flow | Allow jump to final round; compress screening into single pass |
| MR-06 | **Round history grows very large** (many refinement cycles) | Memory pressure | Cap `round_history` to last 10 snapshots; archive older ones |

---

## 9. Conversational Refinement (Phase 9)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| REF-01 | **User removes all requirements** ("Remove everything") | Empty requirements | Agent asks "You've cleared all requirements. Please specify what you're looking for." |
| REF-02 | **User adds contradictory requirement** ("Must have 10 years React, max 2 years experience") | Logical conflict | Detect contradiction; ask user to clarify before proceeding |
| REF-03 | **User changes requirement that didn't exist** ("Remove Python requirement" when Python wasn't listed) | No-op with message | Check if requirement exists; respond "Python isn't in the current requirements. Current must-haves: ..." |
| REF-04 | **User refines category** ("Move React from nice-to-have to must-have") | Category switch | Find matching skill; update `category` field; re-rank accordingly |
| REF-05 | **User changes min_years** ("Increase React to 5 years") | Update existing requirement | Find matching skill; update `min_years`; re-rank |
| REF-06 | **Refinement produces same rankings** (change had no impact) | Inform user | Compare pre/post rankings; report "Rankings unchanged — the new requirement didn't affect the top candidates." |
| REF-07 | **Rapid sequential refinements** (3 changes in 3 messages) | Each triggers re-rank | Process each refinement sequentially; batch if possible |
| REF-08 | **User requests refinement before initial search** | No existing state to refine | Detect missing `parsed_requirements`; redirect to initial search flow |
| REF-09 | **Requirement text is ambiguous** ("add cloud skills") | Multiple interpretations | Agent asks "Which cloud skills? e.g., AWS, GCP, Azure, or all of them?" |
| REF-10 | **User wants to undo** ("Go back to previous requirements") | Restore prior state | Pull previous `parsed_requirements` from `round_history`; re-rank |

---

## 10. Explainability (Phase 8)

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| EXP-01 | **"Why did X rank higher than Y?"** where X or Y is not in shortlist | Invalid reference | Respond "Candidate '{name}' is not in the current shortlist. Available candidates: ..." |
| EXP-02 | **User asks "why" before any ranking exists** | No data to explain | Respond "No candidates have been ranked yet. Let me search first." |
| EXP-03 | **Candidates have very close scores** (92 vs 91) | Nuanced explanation needed | Emphasise the specific dimension(s) that created the difference |
| EXP-04 | **User asks about a candidate eliminated in Round 1** | Not in final list | Pull from `round_history`; explain why they were eliminated and what was lacking |
| EXP-05 | **User references candidate by name, not ID** ("Why did John rank higher?") | Name resolution | Fuzzy match name against `candidate_shortlist[].name`; ask for clarification if ambiguous |
| EXP-06 | **Multiple candidates share the same name** | Ambiguous reference | List all matches with IDs; ask user to specify by ID or provide distinguishing detail |

---

## 11. State & Data Integrity

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| ST-01 | **State key has `None` where a list is expected** | `TypeError` on iteration | Default all list fields to `[]` in state initialisation |
| ST-02 | **`current_round` is not set** | Routing confusion | Default to `1` if missing |
| ST-03 | **`messages` list has non-`BaseMessage` objects** | Serialisation failure | Validate message types before appending; convert raw strings to `HumanMessage` |
| ST-04 | **State is too large to serialise** (> 50MB in memory) | Memory / perf issues | Limit `round_history` depth; compress candidate data; paginate results |
| ST-05 | **Two nodes write to the same state key simultaneously** | Race condition | LangGraph processes nodes sequentially per invocation; not an issue in standard flow |
| ST-06 | **State from a previous conversation is accidentally reused** | Cross-contamination | Reset state on "Start Over"; use unique session IDs |

---

## 12. Environment & Configuration

| # | Scenario | Expected Behaviour | Handling Strategy |
|---|---|---|---|
| ENV-01 | **`.env` file is missing** | Crash on startup | Provide `.env.example`; check at startup; fail with "Copy .env.example to .env and fill in values" |
| ENV-02 | **`CHROMA_PERSIST_DIR` points to invalid path** | ChromaDB can't persist | Validate path exists and is writable on startup; create if missing |
| ENV-03 | **`RESUME_DIR` contains thousands of files** (10,000+) | Slow ingestion | Add progress bar; process in batches; log ETA |
| ENV-04 | **Python version < 3.11** | Syntax errors (TypedDict features) | Check Python version on startup; fail with minimum version requirement message |
| ENV-05 | **Dependency version conflict** | Import errors | Pin exact versions in `requirements.txt`; test with `pip check` |
| ENV-06 | **Disk full during ChromaDB write or model download** | Crash | Catch `OSError`; surface "Disk space insufficient. Free up space and retry." |

---

## Edge Case Summary by Severity

| Severity | Count | Examples |
|---|---|---|
| 🔴 **Critical** (app crash / data corruption) | 14 | LLM-01, LLM-07, AG-04, ING-03, ING-09, ENV-01, ENV-04, FS-01, FS-12, AG-05, ING-08, ENV-06, UI-04, ST-06 |
| 🟠 **High** (incorrect results / bad UX) | 22 | LLM-08–13, RAG-01–02, WF-05, WF-08, MR-02, REF-01–02, EXP-01–02, AG-01, AG-03, AG-07, LLM-14, FS-08, REF-08, WF-07 |
| 🟡 **Medium** (degraded performance / warnings) | 20 | FS-03–06, FS-13–15, ING-04–07, RAG-03–07, LLM-15–18, ENV-03, MR-01, UI-07, REF-06 |
| 🟢 **Low** (cosmetic / edge-of-edge) | 12 | FS-07, FS-09–11, ING-10, RAG-08–10, UI-05–06, UI-08–10, WF-03 |
| | **Total: 68** | |

---

## Test Coverage Matrix

| Component | Edge Cases | Test File | Phase |
|---|---|---|---|
| File System Tools | FS-01 – FS-15 | `tests/test_file_system.py` | 2 |
| Ingestion & ChromaDB | ING-01 – ING-10 | `tests/test_ingestion.py` | 2 |
| RAG Search | RAG-01 – RAG-10 | `tests/test_rag_search.py` | 3 |
| LLM Tools (Groq) | LLM-01 – LLM-24 | `tests/test_tools.py` | 4 |
| Workflow Nodes | WF-01 – WF-17 | `tests/test_parse_jd.py`, `tests/test_ranking.py` | 5 |
| Agent Assembly | AG-01 – AG-07 | `tests/test_agent_e2e.py` | 6 |
| Streamlit UI | UI-01 – UI-10 | Manual testing / `tests/test_ui.py` | 7 |
| Multi-Round Screening | MR-01 – MR-06 | `tests/test_ranking.py` | 8 |
| Refinement | REF-01 – REF-10 | `tests/test_refinement.py` | 9 |
| Explainability | EXP-01 – EXP-06 | `tests/test_explainability.py` | 8 |
| State Integrity | ST-01 – ST-06 | `tests/test_state.py` | 1 |
| Environment | ENV-01 – ENV-06 | `tests/test_config.py` | 0 |

---

## Priority Implementation Order

> Handle critical edge cases first. Build from infrastructure up.

```mermaid
flowchart LR
    A["Phase 0: ENV-01 to ENV-06"] --> B["Phase 2: FS-01 to FS-15, ING-01 to ING-10"]
    B --> C["Phase 3: RAG-01 to RAG-10"]
    C --> D["Phase 4: LLM-01 to LLM-24"]
    D --> E["Phase 5–6: WF-01 to WF-17, AG-01 to AG-07"]
    E --> F["Phase 7–9: UI, MR, REF, EXP"]

    style A fill:#dc2626,stroke:#fca5a5,color:#fff
    style B fill:#ea580c,stroke:#fdba74,color:#fff
    style C fill:#d97706,stroke:#fcd34d,color:#fff
    style D fill:#ca8a04,stroke:#fde047,color:#fff
    style E fill:#65a30d,stroke:#bef264,color:#fff
    style F fill:#0891b2,stroke:#67e8f9,color:#fff
```
