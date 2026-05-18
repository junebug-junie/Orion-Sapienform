# PR: Memory graph from chat — Playwright live verification + turn-id fix

**Branch:** `fix/memory-graph-strict-suggest-draft-v1`  
**Worktree:** `.worktrees/fix-memory-graph-strict-suggest-draft-v1`  
**Base:** `main`

## Summary

Adds a **live Playwright harness** that drives the real Hub Memory graph bridge modal, captures network/console artifacts, and classifies failures. Uses that trace to fix a **frontend turn-id bug** (user turns incorrectly received `:assistant` suffix), improve **diagnostics visibility**, remove **stale preview copy**, and add a **chillin few-shot** to the suggest prompt.

Live e2e **does not pass end-to-end** on this stack because **cortex times out** on both Quick and Brain routes (~28–31s). The harness proves the remaining blocker is **B (backend/cortex)**, not coalescer or request shape.

## Playwright command

```bash
cd .worktrees/fix-memory-graph-strict-suggest-draft-v1

# Hub must be up (default http://127.0.0.1:8080)
export HUB_E2E_BASE_URL=http://127.0.0.1:8080

PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py -v --tb=short
```

Optional: `HUB_E2E_HEADED=1` for headed browser; `HUB_E2E_SUGGEST_TIMEOUT_MS=180000` (default).

**E2e seeding:** append `?hub_e2e=1` to Hub URL — exposes `window.__ORION_HUB_E2E__.seedMemoryGraphTurns` / `openMemoryGraphBridgeForAssistantTurn` (test-only; real bridge + `/api/memory/graph/suggest` unchanged).

## Hub live status (this session)

| Item | Result |
|------|--------|
| Hub reachable | Yes — `http://127.0.0.1:8080` (200) |
| Playwright run | Executed against live Hub |
| Shower case | **Failed** — empty fallback draft (cortex timeout) |
| Chillin case | Not re-run after shower (same stack expected) |

## Artifact directory

```
services/orion-hub/tests/e2e/artifacts/memory-graph-from-chat/
  shower/
    screenshot_before_suggest.png
    screenshot_after_suggest.png
    draft_json.txt
    draft_json.parsed.json
    suggest_request.json
    suggest_response.json
    browser_console.log
    page_errors.log
    network_summary.json
    modal_text.txt
    diagnostics_text.txt
    run_summary.json
    full_page_html.html   # on failure only
```

## Network result (shower case)

| Check | Result |
|-------|--------|
| `/api/memory/graph/suggest` called | Yes (POST, 200) |
| `/api/chat` used for suggest | No |
| Request role/id lines | **Fixed** — user id `urn:uuid:e2e-shower-user-1` (no erroneous `:assistant` suffix) |

## Parsed Draft JSON summary (shower, live)

| Field | Value |
|-------|--------|
| entity_count | 0 |
| situation_count | 0 |
| edge_count | 0 |
| contains user entity | No (fallback empty graph) |
| contains Orion entity | No |
| Valid SuggestDraftV1 shape | Yes |
| Evidence-only envelope | No |
| Prose in textarea | No |

## Diagnostics summary (fallback)

```
attempts:
  [0] route=quick phase=cortex error=TimeoutError:
  [1] route=brain phase=cortex error=TimeoutError:
validation_errors: TimeoutError:, cortex
api_error: memory_graph_suggest_failed
```

Direct API repro (same stack):

```bash
curl -s -X POST http://127.0.0.1:8080/api/memory/graph/suggest \
  -H 'Content-Type: application/json' \
  -d '{"mode":"brain","verbs":["memory_graph_suggest"],"messages":[{"role":"user","content":"... role=user / role=assistant ..."}],"diagnostic":true,"options":{"diagnostic":true},"use_recall":false,"no_write":true}'
```

→ `ok: false`, both attempts `phase: cortex`, `TimeoutError`.

## Root cause classification

| Layer | Verdict |
|-------|---------|
| **A** Frontend request construction | **Was broken, fixed** — `canonicalTurnIdForMemoryGraph` preferred linkage-derived `turnId:assistant` over explicit user `turnId` |
| **B** Backend / model / cortex | **Current blocker** — Quick (8s) and Brain (20s) both timeout at cortex phase |
| **C** Validator | Not reached (no model output) |
| **D** Frontend coalescer | Not implicated — API returns no `data.draft`; empty fallback is correct |
| **E** Stale UI copy | **Fixed** — preview banner no longer says “may mean no durable memory candidate” |

## Changes

| Area | Files |
|------|--------|
| Playwright e2e | `services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py`, `memory_graph_from_chat_cases.py`, `conftest.py` |
| E2e hook | `services/orion-hub/static/js/app.js` (`?hub_e2e=1`) |
| Turn id fix | `services/orion-hub/static/js/app.js` — `canonicalTurnIdForMemoryGraph` |
| Diagnostics UI | `templates/index.html`, `app.js` — `#memoryGraphBridgeDiagnostics`, Copy button |
| Stale copy | `memory-graph-draft-ui.js` — empty-graph preview text |
| Prompt | `orion/cognition/prompts/memory_graph_suggest_prompt.j2` — chillin few-shot |
| Unit tests | `test_memory_graph_bridge_ui.py` (+2 cases) |

## Verification

```bash
cd .worktrees/fix-memory-graph-strict-suggest-draft-v1
PYTHONPATH=. ../../venv/bin/python -m pytest \
  tests/test_memory_graph_suggest_validate.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  -q --tb=short
```

**Result:** 40 passed, exit 0

```bash
PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py -v --tb=short
```

**Result:** shower (and likely chillin) **fail** until cortex/LLM gateway responds within route timeouts — artifacts document **B**.

## Remaining manual step

1. Restore cortex-gateway / cortex-exec health so `memory_graph_suggest` completes (not `TimeoutError` at `phase=cortex`).
2. Re-run Playwright e2e; expect shower case to yield nonempty User + Orion entities when model obeys prompt.
3. For live Hub during dev: sync worktree `static/`, `templates/`, and `/repo` prompt into the paths mounted by `orion-athena-hub` (or run Hub from worktree).

## Env / compose

No new env keys.

## Acceptance status

| Criterion | Status |
|-----------|--------|
| Playwright harness + artifacts | Done |
| Trace classifies failure | Done — **B** (cortex timeout) after **A** fix |
| Code fix for proven frontend bug | Done — turn id |
| Live pass with nonempty graph | **Blocked** on stack cortex timeouts |
