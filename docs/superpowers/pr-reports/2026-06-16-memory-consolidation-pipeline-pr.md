# PR: Memory Consolidation Pipeline

**Branch:** `feat/memory-consolidation-pipeline`  
**Base:** `main`  
**Worktree:** `.worktrees/feat/memory-consolidation-pipeline`

## Summary

Implements durable per-turn memory consolidation: every `chat_history_log` write triggers a post-commit outbox event, a new `orion-memory-consolidation` service classifies the turn (LLM gateway quick lane + logprobs), patches `spark_meta` back through sql-writer, tracks conversation windows, closes on situational phase + boundary scores, runs `memory_graph_suggest`, and persists drafts for operator review.

## Architecture

```
Hub (trace_id = correlation_id)
  → orion:chat:history:turn → sql-writer → chat_history_log
  → orion:memory:turn:persisted (post-commit outbox, same correlation_id)
  → orion-memory-consolidation
       ├─ classify (logprobs)
       ├─ orion:chat:history:spark_meta:patch → sql-writer merge
       ├─ window state (Postgres)
       └─ on boundary → cortex memory_graph_suggest → memory_graph_suggest_drafts
```

**Sacred invariant:** `correlation_id` is the hub `trace_id` end-to-end; LLM/cortex RPCs use separate UUIDs only for reply routing.

## Commits

| SHA | Message |
|-----|---------|
| `6a0262a0` | Add memory consolidation schemas, bus channels, and classify helper |
| `c88e9c6e` | Add sql-writer outbox, patch handler, and memory consolidation service |
| `1bd87e7e` | Extend env sync for memory consolidation pipeline keys |
| (latest) | Add consolidation plan/spec docs and fix smoke script correlation IDs |

## Files changed (high level)

| Area | Files |
|------|-------|
| Schemas / bus | `orion/schemas/memory_consolidation.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml` |
| Classify helper | `orion/memory/consolidation_classify.py` |
| sql-writer | `app/settings.py`, `app/worker.py`, tests |
| Postgres | `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql` |
| Draft store | `orion/memory_graph/draft_repository.py`, `orion/memory_graph/suggest_runner.py` |
| New service | `services/orion-memory-consolidation/**` |
| Env | `.env_example`, `services/orion-sql-writer/.env_example`, `services/orion-memory-consolidation/.env_example`, `scripts/sync_local_env_from_example.py` |
| Smoke | `scripts/smoke_memory_consolidation_pipeline.py` |
| Docs | `docs/superpowers/plans/`, `docs/superpowers/specs/` |

## New bus channels

- `orion:memory:turn:persisted` → `MemoryTurnPersistedV1` (producer: sql-writer, consumer: consolidation)
- `orion:chat:history:spark_meta:patch` → `ChatHistorySparkMetaPatchV1` (producer: consolidation, consumer: sql-writer)

## Operator setup

1. Apply migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`
2. Local env synced (`.env` + service `.env` updated on host for consolidation keys)
3. Build/start consolidation service:
   ```bash
   docker compose --env-file .env --env-file services/orion-bus/.env \
     -f services/orion-memory-consolidation/docker-compose.yml up -d --build
   ```

## Verification

```bash
cd .worktrees/feat/memory-consolidation-pipeline
PYTHONPATH=. ../../orion_dev/bin/python -m pytest \
  tests/test_memory_consolidation_bus_catalog.py \
  tests/test_consolidation_classify.py \
  tests/test_memory_graph_draft_repository.py \
  services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py \
  services/orion-sql-writer/tests/test_spark_meta_patch.py \
  services/orion-memory-consolidation/tests/ -q
```

**Result:** 22 passed, exit 0

```bash
python -m compileall services/orion-memory-consolidation -q
```

**Result:** exit 0

## Test plan

- [ ] Apply Postgres migration on staging DB
- [ ] Restart sql-writer with new subscribe channel + emit flag
- [ ] Start `orion-memory-consolidation` container
- [ ] Send hub turn; confirm `spark_meta.memory_significance_score` or `memory_classify_status=degraded`
- [ ] Force boundary (`phase_change=long_gap`, high boundary score); confirm draft row in `memory_graph_suggest_drafts`
- [ ] Run `scripts/smoke_memory_consolidation_pipeline.py` against live stack

## Known gaps / follow-ups

- **90-min window fallback:** `window_fetch.should_close_by_time_gap` exists but is not yet wired into the worker close path (spec lists as fallback when phase missing).
- **Live smoke:** script requires running bus/sql-writer/postgres/gateway/cortex stack (not run in CI here).
- **Code review subagent:** API limit prevented automated review subagent; manual self-review + 22 unit tests green.

## Non-goals (unchanged)

- No auto-promotion of drafts to RDF/cards
- No spark introspector dependency
- No coverage backfill sweeps
