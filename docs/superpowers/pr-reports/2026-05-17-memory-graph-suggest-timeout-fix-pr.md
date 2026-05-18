# PR: Memory graph suggest — fix Hub per-route timeout budget (8s+20s → verb-aligned)

**Branch:** `fix/memory-graph-strict-suggest-draft-v1`  
**Worktree:** `.worktrees/fix-memory-graph-strict-suggest-draft-v1`

## Artifact summary (before fix)

| Field | Value |
|-------|--------|
| Endpoint | `POST /api/memory/graph/suggest` |
| Request | `verbs: ["memory_graph_suggest"]`, role-grounded evidence, `diagnostic: true` |
| Total wall time | ~28–31s |
| `timeout_sec` per attempt | Quick **8.0**, Brain **20.0** |
| Error | `TimeoutError` at `phase=cortex` |
| `route_used` | null |

**Not** a mystery 30s proxy/Playwright clamp — it was **Hub `asyncio.wait_for` using `MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC=8` + `MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC=20`**, while `memory_graph_suggest.yaml` declares `timeout_ms: 180000`.

## Root cause

| Layer | Finding |
|-------|---------|
| **Source of ~28s** | `services/orion-hub/scripts/memory_graph_suggest.py` → `_call_cortex(..., timeout_sec=8|20)` from settings defaults / `.env` |
| Verb yaml `180000` | Never read by suggest path |
| Bus RPC `TIMEOUT_SEC=400` | Not the bottleneck |
| Playwright / frontend | 180s+ fetch budget; not the failure |

## Fix (verb-specific)

1. `memory_graph_suggest_timeout.py` — resolve budgets from `MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS` (default 180000) or verb yaml; per-route `0` means derive (Quick ~72s, Brain 180s).
2. `memory_graph_suggest.py` — use resolved timeouts; record `elapsed_sec`, `configured_timeout_sec`, `timeout_layer`, `bus_rpc_timeout_sec`, `reached_cortex` on each attempt; `suggest_timeout_budget` on response.
3. `cortex_client.py` — optional `rpc_timeout_sec` per call (≥ attempt wait).
4. `app/settings.py` — defaults `QUICK/BRAIN=0` (derive); `VERB_TIMEOUT_MS=180000`.
5. `.env_example`, `docker-compose.yml`, `services/orion-hub/.env` — `QUICK/BRAIN=0`, `VERB_TIMEOUT_MS=180000`.

### Before / after

| Setting | Before | After |
|---------|--------|-------|
| Quick hub wait | 8s | 72s (40% of 180s verb) |
| Brain hub wait | 20s | 180s (full verb) |
| Max escalation wall | ~28s | ~252s (72+180) |
| Verb yaml honored | No | Yes |

## Playwright re-run (Hub live)

```bash
export HUB_E2E_BASE_URL=http://127.0.0.1:8080
PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py -v --tb=short
```

| Case | Wall time | TimeoutError @ 30s? | Result |
|------|-----------|---------------------|--------|
| shower | ~55s | **No** | Fail — empty cortex text |
| chillin | ~59s | **No** | Fail — empty cortex text |

Artifacts: `services/orion-hub/tests/e2e/artifacts/memory-graph-from-chat/{shower,chillin}/`

Post-fix `suggest_response.json` (shower):

- Quick: `timeout_sec: 72`, `elapsed_sec: ~25`, `reached_cortex: true`
- Brain: `timeout_sec: 180`, `elapsed_sec: ~30`, `reached_cortex: true`
- Error: `empty_final_text` / cortex empty (not `TimeoutError`)

## Remaining blocker

**Model/gateway empty output** — cortex returns `status: success` with `content_len: 0` on both routes. Next: cortex-exec / llm-gateway / `memory_graph_suggest` verb execution (not Hub wait budget).

## Tests

```bash
PYTHONPATH=. ../../venv/bin/python -m pytest \
  tests/test_memory_graph_suggest_validate.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  services/orion-hub/tests/test_memory_graph_suggest_timeout.py -q
```

**44 passed**

## Files changed

- `services/orion-hub/scripts/memory_graph_suggest_timeout.py` (new)
- `services/orion-hub/scripts/memory_graph_suggest.py`
- `services/orion-hub/scripts/bus_clients/cortex_client.py`
- `services/orion-hub/app/settings.py`
- `services/orion-hub/.env_example`
- `services/orion-hub/docker-compose.yml`
- `services/orion-hub/.env` (local)
- `services/orion-hub/tests/test_memory_graph_suggest_timeout.py` (new)
- `services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py` (e2e wait 210s)
