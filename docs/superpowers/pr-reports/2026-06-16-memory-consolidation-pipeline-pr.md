# PR #709: Memory Consolidation Pipeline

**Branch:** `feat/memory-consolidation-pipeline` → `main`  
**PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/709  
**Spec:** `docs/superpowers/specs/2026-06-16-memory-consolidation-design.md`  
**Plan:** `docs/superpowers/plans/2026-06-16-memory-consolidation-pipeline.md`

---

## Summary

Durable per-turn memory consolidation on **every** `chat_history_log` write:

1. **sql-writer** commits the turn, then emits `orion:memory:turn:persisted` (post-commit outbox).
2. **orion-memory-consolidation** classifies via LLM gateway quick lane (logprobs), patches `spark_meta` back through sql-writer, tracks open windows in Postgres, closes on situational phase + boundary scores, runs `memory_graph_suggest`, and persists drafts for operator review.

Spark introspector is **not** on this path. No coverage backfill sweeps.

---

## Architecture

```
Hub (trace_id = correlation_id)
  └─► orion:chat:history:turn
        └─► sql-writer → chat_history_log (correlation_id = trace_id)
              └─► orion:memory:turn:persisted (SAME correlation_id)
                    └─► orion-memory-consolidation
                          ├─ classify (gateway quick, logprobs; RPC uses new UUID for reply only)
                          ├─ orion:chat:history:spark_meta:patch (SAME correlation_id)
                          ├─ window state (Postgres)
                          └─ on boundary → cortex suggest → memory_graph_suggest_drafts
                                └─► sql-writer MERGE spark_meta (by correlation_id)
```

---

## correlation_id contract (non-negotiable)

| Step | ID | Rule |
|------|-----|------|
| Hub turn envelope | `trace_id` | Set once at hub (`build_chat_turn_envelope`) |
| `chat_history_log` row | same | sql-writer copies `env.correlation_id` |
| Outbox `memory.turn.persisted` | same | **`env.correlation_id` preferred** over payload fields; `derive_child()` inherits parent |
| Classify patch | same | `ChatHistorySparkMetaPatchV1.correlation_id = turn.correlation_id` |
| spark_meta merge | same | `WHERE correlation_id = patch.correlation_id` — never INSERT |
| LLM classify RPC | new UUID | Reply routing only (`orion:exec:result:LLMGatewayService:{uuid}`) |
| Cortex suggest RPC | new UUID | Orch reply routing only |
| `spark_meta.cortex_correlation_id` | different | Carried in meta; **not** used as turn key |

**Guards:** consolidation handler `assert env.correlation_id == turn.correlation_id`; outbox logs `memory_turn_persisted_corr_mismatch` if envelope/payload diverge; unit test proves outbox ignores nested `cortex_correlation_id` in spark_meta.

---

## What's new

### Bus + schemas
- `MemoryTurnPersistedV1`, `ChatHistorySparkMetaPatchV1`, window/draft DTOs
- Channels: `orion:memory:turn:persisted`, `orion:chat:history:spark_meta:patch`
- Catalog test: `tests/test_memory_consolidation_bus_catalog.py`

### sql-writer
- Post-commit emit after successful `chat.history` write
- Subscribe + handler for `chat.history.spark_meta.patch.v1` (merge into existing row)
- Settings: `SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED`, channel vars

### orion-memory-consolidation (new service)
- Classify worker (logprobs → `memory_significance_score`, `conversation_boundary_score`)
- Window store + boundary rules (0.70 / 0.85 / 0.92 thresholds)
- 90-min gap fallback when `phase_change` absent on turn
- Suggest runner → `memory_graph_suggest_drafts` (`pending_review`)
- Failed-window retry loop (30 min default)
- Dockerfile, docker-compose, `.env_example`, README

### Postgres
- Migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`
  - `memory_consolidation_windows`
  - `memory_graph_suggest_drafts`

---

## Commits (7)

| SHA | Message |
|-----|---------|
| `6a0262a0` | Schemas, bus channels, classify helper |
| `c88e9c6e` | sql-writer outbox + patch; consolidation service |
| `1bd87e7e` | Env sync script updates |
| `16606a8f` | Plan docs; smoke script UUID fix |
| `5634bd3c` | PR report |
| `f9b50fcf` | Review fixes: draft upsert, 90-min fallback, gateway reply channel |
| `b6b78669` | Prefer envelope correlation_id; hub trace_id test |

---

## Operator setup

### 1. Apply migration (required)

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_memory_consolidation_v1.sql
```

Creates `memory_consolidation_windows` and `memory_graph_suggest_drafts`.

**Note:** Applied on dev host `localhost:55432/conjourney` during PR work. **Not** verified on docker `orion-sql-db` in this session — run on whichever Postgres your stack uses.

### 2. Env (root + services)

Root `.env_example` keys added; sync via:

```bash
python3 scripts/sync_local_env_from_example.py orion-sql-writer orion-memory-consolidation
```

Key vars:
- `CHANNEL_MEMORY_TURN_PERSISTED`
- `CHANNEL_CHAT_HISTORY_SPARK_META_PATCH`
- `SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED=true`
- `MEMORY_CONSOLIDATION_ENABLED=true`
- `LLM_LOGPROB_SUMMARY_ENABLED=true`

### 3. Restart sql-writer + start consolidation

```bash
# sql-writer must subscribe to orion:chat:history:spark_meta:patch
docker compose --env-file .env --env-file services/orion-bus/.env \
  -f services/orion-memory-consolidation/docker-compose.yml up -d --build
```

---

## Verification (automated)

```bash
cd .worktrees/feat/memory-consolidation-pipeline  # or checkout branch
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  tests/test_memory_consolidation_bus_catalog.py \
  tests/test_consolidation_classify.py \
  tests/test_memory_graph_draft_repository.py \
  services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py \
  services/orion-sql-writer/tests/test_spark_meta_patch.py \
  services/orion-memory-consolidation/tests/ -q
```

**Result:** 25 passed, exit 0

```bash
python -m compileall services/orion-memory-consolidation -q
```

**Result:** exit 0

---

## Test plan (staging / live)

- [ ] Apply migration on stack Postgres (`orion-sql-db` or equivalent)
- [ ] Restart sql-writer (new subscribe channel + emit flag)
- [ ] Start `orion-memory-consolidation`
- [ ] Send one hub turn → confirm same `correlation_id` on:
  - `chat_history_log.correlation_id`
  - outbox envelope (bus tap or log)
  - `chat_history_log.spark_meta.memory_significance_score` or `memory_classify_status=degraded`
- [ ] Force boundary (`phase_change=long_gap`, high boundary score) → draft row in `memory_graph_suggest_drafts` with `status=pending_review`
- [ ] `PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py` (requires live stack)

---

## Non-goals

- Auto-promotion of drafts to RDF / memory cards
- Changes to spark introspector or recall
- Recovery sweeps for missing classification coverage

---

## Remaining risks

- **Live E2E not run** in PR session — unit tests + local migration only
- **Consolidation service** not deployed/restarted on live stack yet
- **Minor:** service starts without Postgres pool and skips work silently if `POSTGRES_URI` unset (health still returns 200)
