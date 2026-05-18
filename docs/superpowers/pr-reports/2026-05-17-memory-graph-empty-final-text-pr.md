# PR Report: memory_graph_suggest empty `final_text` (exec assembly + hub extraction)

**Branch:** `fix/memory-graph-strict-suggest-draft-v1`  
**Worktree:** `.worktrees/fix-memory-graph-strict-suggest-draft-v1`  
**Date:** 2026-05-17

## Classification

**Primary: B — Orch/Exec response assembly**  
**Contributing: C — Exec/gateway token budget too low (512 → truncated non-JSON prose)**

**Secondary (live e2e blocker after B/C fix):** Hub container `ORION_REPO_ROOT` not honored by `orion.memory_graph.project` → RDF preview 500 (deployment path, not coalescer/draft gating).

Not A (Hub was reading `final_text` only, but root cause was exec zeroing it). Not D/E as primary (LLM step runs; model returns content when budget allows).

## Where text vanished

| Layer | Before | After |
|-------|--------|-------|
| **LLM gateway** | ~41 chars prose (`completion_tokens` tiny under 512 cap) | ~2477 chars JSON (`completion_tokens=1166`, `emitted_chars=2477`) |
| **cortex-exec `final_text_assembly`** | `structured_output_rejected=True` → `final_len=0`, `status=success` | `structured_output_rejected=False`, `final_len=2477`, JSON in `final_text` |
| **Hub `memory_graph_suggest.py`** | `hub_effective_chat_text` → empty → `empty_final_text` | `hub_memory_graph_suggest_text` reads `final_text` + step `LLMGatewayService.content` fallback |
| **Hub RDF preview** | N/A (never reached) | `FileNotFoundError` `/app/config/...` → fixed via `ORION_REPO_ROOT=/repo` |

## Evidence (live)

**Cortex-exec (before):**
```
effective_max_tokens=512 … emitted_chars=41
structured_output_rejected=True … final_len=0 result_len=0
```

**Cortex-exec (after):**
```
effective_max_tokens=1536 max_tokens_source=settings.llm_memory_graph_suggest_max_tokens
emitted_chars=2477 … final_len=2477 result_len=2477
```

**Playwright shower response (after):** `ok: true`, entities User/Orion/shower, 2 situations, `route_used: quick`.

Artifacts: `services/orion-hub/tests/e2e/artifacts/memory-graph-from-chat/{shower,chillin}/`

## Changes

### cortex-exec (`services/orion-cortex-exec/`)
- `app/settings.py` — `LLM_MEMORY_GRAPH_SUGGEST_MAX_TOKENS=1536`
- `app/executor.py` — route `memory_graph_suggest` / `llm_memory_graph_suggest` to that budget
- `app/router.py` — structured-verb empty output → fail (not success); extra JSON fields; `===MEMGRAPH_SUGGEST_TRACE===`; `structured_rejection_preview` in metadata
- `.env_example` — document env var
- `tests/test_memory_graph_suggest_final_text.py` — assembly + budget tests

### Hub (`services/orion-hub/`)
- `scripts/cortex_memory_graph_text.py` — extract from `final_text` or step payloads
- `scripts/memory_graph_suggest.py` — use extractor; `text_extraction` diagnostics; trace logging
- `tests/test_cortex_memory_graph_text.py` — nested step content when `final_text` empty

### Orion (`orion/memory_graph/`)
- `project.py`, `validate.py` — `_repo_root()` honors `ORION_REPO_ROOT` (hub Docker uses `/repo`)

## Tests

```bash
# cortex-exec
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_memory_graph_suggest_final_text.py -q

# hub (worktree)
cd .worktrees/fix-memory-graph-strict-suggest-draft-v1
PYTHONPATH=. pytest \
  tests/test_memory_graph_suggest_validate.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  services/orion-hub/tests/test_memory_graph_suggest_timeout.py \
  services/orion-hub/tests/test_cortex_memory_graph_text.py -q

# live Playwright
export HUB_E2E_BASE_URL=http://127.0.0.1:8080
PYTHONPATH=. pytest services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py -v --tb=short
```

**Results:** cortex-exec 4 passed; hub unit 45 passed (worktree); Playwright **2 passed** (~85s).

## Playwright

- No 8/20s Hub `TimeoutError`
- No `empty_final_text` success
- Shower: role-grounded SuggestDraftV1 (User, Orion, shower + situations)
- Chillin: role-grounded graph with User/Juniper + Orion entities

## Deploy notes

Recreate/restart with:
- Hub: `MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC=0`, `BRAIN=0`, `VERB_TIMEOUT_MS=180000`
- cortex-exec: image rebuild or copy `app/{router,executor,settings}.py`; `LLM_MEMORY_GRAPH_SUGGEST_MAX_TOKENS=1536`
- Hub image: include `orion/memory_graph/{project,validate}.py` with `ORION_REPO_ROOT` support (or mount `/repo` and set env)

## Remaining

- Rebuild hub/cortex-exec images so container copies are not manual `docker cp`.
- Optional: persist `test_memory_graph_suggest_timeout.py` on main if only in worktree.
