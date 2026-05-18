# PR: Fix Memory Graph suggest draft coalescing (prose → Draft JSON)

**Branch:** `fix/memory-graph-suggest-draft-coalesce`  
**Worktree:** `.worktrees/fix-memory-graph-suggest-draft-coalesce`  
**Base:** `fix/mind-hub-modal-live-verification` @ `f494b2ce`  
**Head:** (see `git log -1` on branch)

## Summary

Memory Graph **Suggest** (Memory tab and **Memory graph from chat** bridge) no longer posts to `/api/chat` or copies assistant prose into the **Draft JSON** textarea. Both flows call **`POST /api/memory/graph/suggest`** and run responses through **`coalesceMemoryGraphSuggestEnvelope`**, which only accepts `draft` / `appendix_c_json` / parse-validated JSON shaped like **SuggestDraftV1**. On model or route failure, the textarea gets a **valid empty template** (with `utterance_ids` when known); diagnostics appear in the status/output panel.

## Problem

- UI showed prose such as *"There's a kind of sacred tension…"* in Draft JSON.
- Parser correctly failed with *"Could not parse a single JSON object."*
- Root cause: frontend used `/api/chat` with `verbs: ['memory_graph_suggest']` and `coalesceChatSuggestDraft` trusted `data.text` / `raw.final_text`.

## Solution

| Area | Change |
|------|--------|
| `memory-graph-draft-ui.js` | `coalesceMemoryGraphSuggestEnvelope`, `emptyValidSuggestDraft`, diagnostics collector; legacy `coalesceChatSuggestDraft` delegates to envelope |
| `memory.js` | Suggest → `/api/memory/graph/suggest`; parse-gate `sessionStorage` import |
| `app.js` | Bridge suggest → same route; `utteranceIds` from selected turns; HTTP error resets to empty template |
| Tests | Static routing regressions, Node coalescer cases, backend prose → `no_json_object` |

## API / UX contract

**Success:** `draftText` = pretty-printed validated draft; status shows `route_used`, `attempts`, `validation_errors` when diagnostic.

**Prose / no JSON:** `draftText` = empty SuggestDraftV1 template; `error` = `invalid_model_output` or `memory_graph_suggest_failed`; prose never in textarea.

**RDF / partial failure (`ok: false` with `draft`):** Textarea still gets draft JSON; UI reports error and merges `violations` into diagnostics.

## Files changed

- `services/orion-hub/static/js/memory-graph-draft-ui.js`
- `services/orion-hub/static/js/memory.js`
- `services/orion-hub/static/js/app.js`
- `services/orion-hub/tests/test_memory_graph_bridge_ui.py`
- `services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py` (new)
- `services/orion-hub/tests/test_memory_graph_suggest_fallback.py`

## Test plan

```bash
cd .worktrees/fix-memory-graph-suggest-draft-coalesce
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_fallback.py::test_cortex_prose_returns_no_json_object_without_draft \
  -q --tb=short
```

**Manual (hub running):**

1. Chat → **Memory graph from chat** → select turns with stable ids → **Suggest draft**.
2. Confirm Draft JSON is parseable JSON (empty template or extracted graph), never assistant reply prose.
3. **Continue to Memory** → Draft JSON still valid; **Validate** succeeds on empty template.
4. Memory tab → **Suggest** with same checks.

## Verification (automated)

| Command | Exit | Result |
|---------|------|--------|
| Targeted pytest (above) | 0 | 10 passed |

## Remaining risks

- Client-side draft detection is heuristic (`looksLikeMemoryGraphDraftObject`), not full Pydantic parity; backend remains authoritative.
- Live-stack repro not run in this session (mocked/static + unit tests only).

## Notes

- No `.env` / `docker-compose` / `settings.py` changes (frontend + tests only).
- Aligns with future `suggest_with_escalation` response shape (`attempts`, `route_used`, `validation_errors`) via diagnostics normalization.
