# PR: Mind recall prefetch preflight (Orch → Recall → projection)

**Branch:** `fix/mind-recall-prefetch-preflight`  
**Base:** `main` (or prior convergence seam branch as appropriate)

## Summary

Orch Mind preflight now performs **Exec-parity recall** over the Redis bus before building the shared cognitive projection. Recall replies are merged into plan ctx via `recall_ctx_merge_from_reply()`, with structured diagnostics (`metadata.recall_prefetch`, `mind_projection_prebuild_ctx_summary`).

Live verification found and fixed a **catalog reply-channel bug**: Orch initially used `orion:orch:result:RecallService:*`, which is **not** registered in `orion/bus/channels.yaml`. Recall processed queries but could not publish replies; Orch saw a 30s client-side RPC timeout. The fix aligns Orch with Exec/Hub: **`orion:exec:result:RecallService:{uuid}`**.

After the fix, live prefetch succeeds in **~585 ms** with **4 fragments** (`reflect.v1`). Projection item count remains **7 (orion-only)** when recall sources are `sql_timeline` / `vector`, because the recall substrate adapter only maps `journal`, `metacog`, `tension`, and `dream` — documented follow-up, not changed in this PR.

## Problem

- Mind preflight had identity YAML but **`recall_bundle_present=false`** (recall prefetch timed out or never wrote ctx).
- Exec recall on the same turn could succeed later with a longer timeout and a valid reply channel.
- No structured observability for prefetch start/result/timeout vs projection inputs.

## Solution

| Area | Change |
|------|--------|
| **Shared query/bundle** | `orion/cognition/recall_query.py` — `build_recall_query_v1()`, `recall_ctx_merge_from_reply()`, `DEFAULT_RECALL_REPLY_PREFIX` |
| **Prefetch** | `orion/cognition/recall_prefetch.py` — bus RPC, diagnostics tuple, structured logs, configurable reply prefix |
| **Orch wiring** | `prepare_plan_context_for_mind_projection()` — prefetch before Mind; `metadata.recall_prefetch` |
| **Exec parity** | `run_recall_step` uses shared `recall_ctx_merge_from_reply()` |
| **Bus catalog** | `orion-cortex-orch` added as producer/consumer on recall request/result channels |
| **Config** | `CHANNEL_RECALL_*`, `MIND_RECALL_PREFETCH_*` in settings, `.env`, `.env_example`, `docker-compose.yml` |

## Configuration

| Variable | Default | Service |
|----------|---------|---------|
| `MIND_RECALL_PREFETCH_ENABLED` | `true` | orion-cortex-orch |
| `MIND_RECALL_PREFETCH_TIMEOUT_SEC` | `30` | orion-cortex-orch |
| `CHANNEL_RECALL_INTAKE` | `orion:exec:request:RecallService` | orch (and exec) |
| `CHANNEL_RECALL_REPLY_PREFIX` | `orion:exec:result:RecallService` | orch Mind preflight |
| `ORION_BUS_URL` | (required) | orch, exec, gateway |

**Not HTTP:** There is no `ORION_RECALL_BASE_URL`; recall is bus-only (`recall.query.v1`).

## Live verification (Athena stack)

| Attempt | `correlation_id` | Recall prefetch | Notes |
|---------|------------------|-----------------|-------|
| 1 (pre-fix) | `e02caf26-…` | Timeout 30s | Recall replied in ~0.7s but publish failed: `Channel not found in catalog: orion:orch:result:RecallService:…` |
| 2 (post-fix) | `2ebf0f3b-…` | **ok**, 585 ms, 4 fragments | `recall_bundle_present=true`; Mind run `b88569ca-…`, `cognitive_projection_seen=true` |
| Probe | — | 11 fragments, sources `sql_timeline`/`vector` | Projection still `item_count=7`, all `orion` (adapter filter) |

Gateway full response blocked by **exec missing `ORION_BUS_URL`** on redeploy (verb intake never consumed `orion:verb:request`). Added to `services/orion-cortex-exec/.env_example` and live `.env`.

## Tests

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  orion/cognition/tests/test_recall_prefetch.py \
  services/orion-cortex-orch/tests/test_mind_projection_context_enrichment.py \
  services/orion-cortex-exec/tests/test_recall_active_turn_exclusion_payload.py -q
```

**Result:** 7 passed (prefetch success, timeout degrade, Exec bundle parity, projection producer, orch integration).

## Files changed (high signal)

- `orion/cognition/recall_query.py` (new)
- `orion/cognition/recall_prefetch.py`
- `orion/cognition/projection_context.py`
- `services/orion-cortex-orch/app/mind_runtime.py`
- `services/orion-cortex-orch/app/settings.py`
- `services/orion-cortex-orch/.env` / `.env_example` / `docker-compose.yml`
- `services/orion-cortex-exec/app/executor.py`
- `services/orion-cortex-exec/.env` / `.env_example`
- `services/orion-cortex-gateway/.env` / `.env_example`
- `orion/bus/channels.yaml`
- `orion/cognition/tests/test_recall_prefetch.py` (new)
- `services/orion-cortex-orch/tests/test_mind_projection_context_enrichment.py`

## Follow-ups (out of scope)

1. **Recall adapter:** Map `sql_timeline` / `vector` (and peers) in `map_recall_bundle_to_substrate` so prefetch increases projection item count.
2. **Exec query builder:** Route `run_recall_step` through `build_recall_query_v1()` to remove inline drift.
3. **E2E gateway:** Ensure all exec lanes have `ORION_BUS_URL` after compose up.

## Deploy checklist

- [ ] Rebuild/restart: `orion-cortex-orch`, `orion-cortex-exec` (all lanes), `orion-recall`, `orion-mind`, `orion-hub`, `orion-cortex-gateway`
- [ ] Confirm orch env: `MIND_RECALL_PREFETCH_ENABLED=true`, `MIND_RECALL_PREFETCH_TIMEOUT_SEC=30`, `CHANNEL_RECALL_REPLY_PREFIX=orion:exec:result:RecallService`
- [ ] Logs: `mind_recall_prefetch_result` with `ok=true` on a `chat_general` + `mind_enabled` turn
- [ ] Optional: `recall_prefetch.result_count > 0` and `orch_preflight_input_summary.recall_bundle_present=true`
