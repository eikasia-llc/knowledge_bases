# Agents Log
- status: active
- type: log
<!-- content -->
Most recent event comes first

<!-- MERGED FROM NEWER VERSION -->

- context_dependencies: {"guideline": "AGENTS.md"}
<!-- content -->

## Intervention History
- status: active
<!-- content -->

### Housekeeping Report (Initial)
- status: active
<!-- content -->
**Date:** 
**Summary:** Executed initial housekeeping protocol.
**AI Assitant:**
- **Dependency Network:** 
- **Tests:**

### Bug Fix: Advanced Analysis (Shape Mismatch)
- status: active
<!-- content -->
**Date:** 2024-05-22
**Summary:** Fixed RuntimeError in `advanced_experiment_interface.ipynb`.
- **Issue:** `compute_policy_metrics` in `src/analysis.py` passed 1D inputs `(100, 1)` to agents expecting 2D inputs `(100, 2)`.
- **Fix:** Created `src/advanced_analysis.py` with `compute_advanced_policy_metrics`.
- **Details:** The new function constructs inputs as `[p, t]` with `t` fixed at 0 (default).
- **Files Modified:** `src/advanced_simulation.py` updated to use the new analysis function.

### Bug Fix: Notebook NameError
- status: active
<!-- content -->
**Date:** 2024-05-22
**Summary:** Fixed NameError in `advanced_experiment_interface.ipynb`.
- **Issue:** The variable `ep_id` was used in a print statement but was undefined in the new JSON saving block.
- **Fix:** Removed the erroneous print statement and cleanup old comments. Validated that the correct logging uses `current_step_info['episode_count']`.

### 2026-01-31: Housekeeping Execution (Antigravity)
- id: agents_log.intervention_history.2026_01_31_housekeeping_execution_antigravity
- status: active
- type: context
- last_checked: 2026-02-01
<!-- content -->
- **Task**: Executed Housekeeping Protocol & Fixed Scraper.
- **Problem**: Scraper was falling back to static mode (3 events) due to missing dependencies.
- **Fix**: Installed `selenium`, `webdriver-manager` and updated `requirements.txt`.
- **Changes**: 
    - Verified `chromadb` installation.
    - Updated dataset via `scripts/update_dataset.py` (Now finds **53 events**).
    - Ran all unit tests and connectivity checks.
    - Updated `docs/HOUSEKEEPING.md` with corrected report.
- **Outcome**: System healthy. 53 events scraped. Minor regression in MCP tests noted.

### 2026-01-31: Fix Future Event Access (Antigravity)
- id: agents_log.intervention_history.2026_01_31_fix_future_event_access_antigravity
- status: active
- type: context
- last_checked: 2026-02-01
<!-- content -->
- **Task**: Debugging LLM refusal to access future events.
- **Problem**: LLM refused to answer questions about future events (e.g., Feb 2026) despite data being in `raw_events.json`. Error message ("I can only access information from the provided context") indicated strictly following RAG-only context rules in personality.
- **Fix**: Modified `prompts/personality.md` to explicitly authorize and mandate Tool usage (`get_events`) when text context is insufficient.
- **Changes**:
    - Updated `prompts/personality.md` "Behavioral Guidelines" and "What to Avoid".
    - Verified `get_events` tool logic with `test_events.py` (Tool works correctly).
- **Outcome**: LLM personality now prioritize Tool usage over politeness when data is missing from text chunks.

### 2026-01-31: Model Default Switch (Antigravity)
- id: agents_log.intervention_history.2026_01_31_model_default_switch_antigravity
- status: active
- type: context
- last_checked: 2026-02-01
<!-- content -->
- **Task**: Fix "Future Event Access" failure on default settings.
- **Problem**: `gemini-2.0-flash-lite` (previous default) consistently failed to use the `get_events` tool for speaker queries, hallucinating a capability limitation ("I cannot search for events by speaker").
- **Fix**: 
    - Switched default model in `app.py` from Lite (Index 1) to **Gemini 2.0 Flash** (Index 0).
    - Updated `src/mcp/server.py` tool schema to explicitly mention "speaker name" in the query description (as a best practice, though Lite still struggled).
- **Outcome**: Default system now uses the more capable Flash model, which correctly calls tools and answers future event queries.

### 2026-01-31: Enhance Calendar Prompt (Antigravity)
- id: agents_log.intervention_history.2026_01_31_enhance_calendar_prompt_antigravity
- status: active
- type: context
- last_checked: 2026-02-01
<!-- content -->
- **Task**: Improve detail in calendar-triggered event queries.
- **Change**: Updated the `auto_prompt` in `app.py` (triggered by calendar clicks) to explicitly request an "abstract or description".
- **Goal**: Ensure the LLM provides more content about the talk, not just the title/time.

### 2026-02-01: Clean and Merge Repository (Antigravity)
- id: agents_log.intervention_history.2026_02_01_clean_and_merge_repository_antigravity
- status: active
- type: context
- last_checked: 2026-02-01
<!-- content -->
- **Task**: Synchronize local knowledge base with `mcmp_chatbot` remote repository.
- **Actions**:
    - Ran `clean_repo.py` and `compare_and_merge.py` according to protocol.
    - **Updated**: `PERSONALITY_AGENT.md`, `SCRAPER_AGENT.md`, `HOUSEKEEPING.md`, `GCLOUD_AGENT.md`, `RAGS_AGENT.md`, `MCP_AGENT.md`, `AGENTS_LOG.md`, `PYUI_AGENT.md`.
    - **New**: Imported `personality.md` as reference to `content/guidelines/LEOPOLD_PERSONA.md`.
    - **Ignored**: `TODOS.md` (generic), `rag_improvements.md` (missing from remote).
    - **Protocol Update**: Added "Cleanup" step to `CLEANER_AGENT.md` to enforce `temprepo_cleaning` emptying after runs.
- **Outcome**: Knowledge base updated and cleaner protocol refined.
