# Memory Consolidation Pipeline — Design Spec

**Date:** 2026-06-16  
**Status:** Approved v2 — implementation plan written  
**Scope:** Durable per-turn memory classification, conversation window closure, automated `memory_graph_suggest` in draft state, new `orion-memory-consolidation` service.

---

## Problem

Orion needs **durable automated memory consolidation that works on every run** — not optional spark side-paths, not recovery backfills for missing coverage, not architecture that depends on introspection queues or gateway phi thresholds.

Requirements:

1. **Every turn** that lands in `chat_history_log` gets memory significance + boundary scores.
2. Conversation windows close using **situational phase bounds** (90-min gap is fallback only).
3. Closed windows run `memory_graph_suggest` automatically; output stays in **draft** until human review.
4. **`correlation_id` is sacred** — the hub `trace_id` for the turn, propagated unchanged end-to-end. No new IDs for tracing turns.

---

## Core invariant (non-negotiable)

> If a turn is in `chat_history_log`, it **will** be classified and eligible for consolidation.  
> The memory pipeline **does not** depend on spark candidates, heavy introspection, or gateway publish thresholds.

**Canonical durability chain:**

```
Hub (trace_id = correlation_id)
  └─► orion:chat:history:turn          [ChatHistoryTurnV1]
        └─► orion-sql-writer           INSERT chat_history_log (correlation_id = trace_id)
              └─► orion:memory:turn:persisted   [MemoryTurnPersistedV1, SAME correlation_id]
                    └─► orion-memory-consolidation
                          ├─ classify (gateway quick lane, logprobs)
                          ├─ patch spark_meta (SAME correlation_id)
                          ├─ track open window
                          └─ on boundary → suggest → SuggestDraftV1 draft
```

Spark introspector (tissue, narrative `introspect_spark`) is **out of scope** for this pipeline. It may continue independently; memory must not wait on it.

---

## correlation_id contract

| Field | Value | Rule |
|-------|-------|------|
| `ChatHistoryTurnEnvelope.correlation_id` | Hub `trace_id` | Set once at hub; never regenerated |
| `ChatHistoryTurnV1.id` | Same as `trace_id` | Hub already does this |
| `chat_history_log.correlation_id` | Same | sql-writer copies from envelope |
| `chat_history_log.id` | Same | Primary row key |
| Classify patch envelope | Same `correlation_id` | Merge only; never INSERT a new row |
| `memory_window_id` | **New UUID** | Groups turns for one suggest draft; stores `turn_correlation_ids: […]` |

**Forbidden in this pipeline:**

- UUIDv5 derivation from trace_id (introspector `_resolve_cortex_correlation_uuid` pattern)
- Substituting cortex reply correlation for turn correlation
- Creating parallel rows keyed on a different ID

---

## Non-goals

- Auto-promotion of drafts to RDF or memory cards.
- Changes to `orion-recall` retrieval or fusion scoring.
- Grammar atom emission (follow-on after draft validation).
- Modifying spark introspector behavior for memory (no classify RPC in introspector).
- Recovery sweeps as a **coverage** mechanism (see Failure recovery below).

---

## Architecture overview

```
┌─────────────┐     orion:chat:history:turn      ┌─────────────────┐
│  orion-hub  │ ───────────────────────────────► │ orion-sql-writer │
│ trace_id=T  │         correlation_id=T         │ chat_history_log │
└─────────────┘                                  └────────┬─────────┘
                                                            │ AFTER COMMIT
                                                            ▼
                                              orion:memory:turn:persisted
                                                   correlation_id=T
                                                            │
                                                            ▼
                                              ┌──────────────────────────┐
                                              │ orion-memory-consolidation│
                                              │ 1. classify (logprobs)    │
                                              │ 2. patch spark_meta (T)   │
                                              │ 3. window state update    │
                                              │ 4. boundary → suggest     │
                                              └──────────────────────────┘
                                                            │
                         orion:chat:history:spark_meta:patch (corr=T)
                                                            ▼
                                              ┌─────────────────┐
                                              │ orion-sql-writer │ MERGE spark_meta
                                              └─────────────────┘
```

---

## Component 1: sql-writer outbox (`memory.turn.persisted`)

### Why

Parallel consumers on `orion:chat:history:turn` race: consolidation might patch before the row exists. **Publish only after commit** so consolidation always works against a durable row.

### Change

After successful `ChatHistoryLogSQL` write for `kind=chat.history`:

1. Publish `MemoryTurnPersistedV1` on `orion:memory:turn:persisted`.
2. Payload includes: `correlation_id`, `prompt`, `response`, `spark_meta`, `created_at`, `session_id` (informational only — not used for window bounds).

Envelope `correlation_id` **must equal** payload `correlation_id` **must equal** hub `trace_id`.

### Gate

If `PUBLISH_CHAT_HISTORY_LOG=false`, no turn is persisted and memory pipeline does not run. This is the only intentional skip. Document in ops runbook.

---

## Component 2: `orion-memory-consolidation` service

Single owner of classify → patch → window → suggest. No split across introspector + recovery crons.

### Subscribe

- **Primary intake:** `orion:memory:turn:persisted` only.
- **Not** `orion:spark:introspect:candidate*`.

### Per-turn handler (every persisted turn)

1. **Classify** — direct LLM gateway RPC (`quick` lane), bypass cortex-exec.
2. **Patch** — publish `ChatHistorySparkMetaPatchV1` on `orion:chat:history:spark_meta:patch` with same `correlation_id`.
3. **Window state** — append turn to open window in service state (Postgres `memory_consolidation_windows` table or Redis with Postgres backing).
4. **Boundary check** — if window should close (see Component 3), run suggest pipeline synchronously for that window.

### Classify: logprobs

Gateway: `return_logprobs=true`, `logprobs_top_k=2`, `max_tokens=3`, `logprob_summary_only=false`.

Prompt output (two lines only):

```
MEMORY: YES|NO
BOUNDARY: YES|NO
```

Softmax per line → `memory_significance_score`, `conversation_boundary_score`.

**Store derived floats only** in `spark_meta` via patch merge. Optional `memory_classify_probe` with four scalars (YES/NO logprobs per line). Do **not** store raw logprob arrays or full `llm_uncertainty` blob.

Shared helper: `orion/memory/consolidation_classify.py`.

### Patch write-back

sql-writer new handler for `chat.history.spark_meta.patch`:

- Lookup row by `correlation_id` only.
- `_merge_spark_meta(existing, patch)`.
- If row missing: **reject + log error** (should never happen if fed from `memory.turn.persisted` outbox).

### Suggest + draft

On window close:

1. Fetch all turns in window by **`correlation_id` list** from service state (not session_id).
2. Build annotated transcript (`[sig=0.82]` per turn).
3. Call `memory_graph_suggest` via cortex-exec verb (same validation as hub manual path).
4. Write `SuggestDraftV1` to existing hub memory-graph draft store.
5. Stamp each turn: `memory_window_id`, `memory_consolidated_at` via patch envelope.

---

## Component 3: Window bounds & boundary detection

### Primary: situational phase (already in `spark_meta`)

From `cortex-exec/app/situation.py` → `conversation_phase.phase_change`:

| Phase | Meaning |
|-------|---------|
| `same_breath` | < 2 min since last user turn |
| `short_pause` | < 20 min |
| `resumed_thread` | 20 min – 3 hr |
| `long_gap` | 3 – 12 hr |
| `next_day` | Day boundary crossed |
| `stale_thread` | > 48 hr |

**Window opens** on first turn after a boundary phase (or service boot).

**Window closes** when an incoming turn has:

- `phase_change ∈ {long_gap, next_day, stale_thread}` AND `conversation_boundary_score ≥ 0.70`, **or**
- `phase_change ∈ {unknown}` AND `conversation_boundary_score ≥ 0.85`, **or**
- Sharp in-thread shift: `phase_change ∈ {same_breath, short_pause, resumed_thread}` AND `conversation_boundary_score ≥ 0.92`

On close: the **closing turn starts the next window**; the **completed window** is everything before it.

### Fallback: time gap (only when phase missing on turns)

Use `discussion_window` 90-min contiguity (`_DEFAULT_CONTIGUITY_MAX_GAP_SEC = 5400`) **only** when `conversation_phase.phase_change` is absent on window turns.

### NOT used for bounds

- `session_id` (explicitly unreliable in hub)
- Spark candidate delivery
- Heavy introspection completion

---

## Component 4: Failure recovery (not coverage)

One retry mechanism for **failures**, not for **missing turns**:

| Failure | Behavior |
|---------|----------|
| Classify RPC timeout | Patch `memory_classify_status=degraded`; retry classify on next `memory.turn.persisted` for same corr (idempotent) OR inline retry 2x before degraded |
| Patch merge fails | Log error; window state still updated; ops alert |
| Suggest verb fails | Mark window `consolidation_status=failed` in service table; **retry job** scans failed windows every 30 min |
| Duplicate `memory.turn.persisted` | Idempotent classify + patch by correlation_id |

**No "rail C" backfill** scanning for rows that never got a spark candidate. If a row is in `chat_history_log`, the outbox **must** have fired. If outbox didn't fire, that's a sql-writer bug to fix — not a second pipeline.

---

## Bus catalog & schema registry

### New channels (`orion/bus/channels.yaml`)

| Channel | Schema | Producer | Consumer |
|---------|--------|----------|----------|
| `orion:memory:turn:persisted` | `MemoryTurnPersistedV1` | `orion-sql-writer` | `orion-memory-consolidation` |
| `orion:chat:history:spark_meta:patch` | `ChatHistorySparkMetaPatchV1` | `orion-memory-consolidation` | `orion-sql-writer` |

Add `orion-memory-consolidation` to `orion:chat:history:turn` consumer list only if we ever need direct tap — **v1 uses outbox only**.

Catalog test: `tests/test_memory_consolidation_bus_catalog.py`.

### New schemas (`orion/schemas/` + `orion/schemas/registry.py`)

- `MemoryTurnPersistedV1`
- `ChatHistorySparkMetaPatchV1`
- `MemoryConsolidationWindowV1` (service-internal / draft metadata)

---

## Artifact storage

| Artifact | Where | Key |
|----------|-------|-----|
| Turn row | `chat_history_log` | `correlation_id` (= trace_id) |
| Memory scores | `chat_history_log.spark_meta` | merged by correlation_id |
| Open/closed windows | `memory_consolidation_windows` (new table) | `memory_window_id` + `turn_correlation_ids[]` |
| Suggest output | Hub memory-graph draft store | `memory_window_id` |

**`chat_message`:** not updated in v1. Consolidation reads `chat_history_log`.

---

## Env & docker

Root `.env_example` (not per-service duplication):

```
CHANNEL_MEMORY_TURN_PERSISTED=orion:memory:turn:persisted
CHANNEL_CHAT_HISTORY_SPARK_META_PATCH=orion:chat:history:spark_meta:patch
MEMORY_CONSOLIDATION_ENABLED=true
LLM_LOGPROB_SUMMARY_ENABLED=true
```

Bring-up (existing pattern):

```bash
docker compose --env-file .env --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml build
```

Run `python scripts/sync_local_env_from_example.py` after example changes.

**Note:** `orion-consolidation-runtime` is substrate motif aggregation — unrelated.

---

## Service layout

```
services/orion-memory-consolidation/
  app/
    main.py              — bus subscriber, health
    worker.py            — memory.turn.persisted handler
    classify.py            — gateway RPC + logprob extraction
    window_state.py        — open window tracking (Postgres)
    window_fetch.py        — phase-bound fetch + 90-min fallback
    suggest_runner.py      — memory_graph_suggest wrapper
    retry_failed_windows.py — failed window retry only
    settings.py
  tests/
  requirements.txt
  .env_example
  docker-compose.yml
  Dockerfile
```

---

## Acceptance checks (all must pass)

- [ ] Every hub WS turn with `PUBLISH_CHAT_HISTORY_LOG=true` produces `memory.turn.persisted` within 5s of sql-writer commit.
- [ ] `memory_significance_score` present on **100%** of persisted turns (or explicit `memory_classify_status=degraded` after retries — never silent null).
- [ ] All patches use the **original** hub `trace_id` as `correlation_id`; zero new correlation IDs in the memory path.
- [ ] Gateway spark publish threshold (phi delta) does **not** affect memory classification.
- [ ] Spark introspector heavy path disabled / timed out / redis down does **not** affect memory classification.
- [ ] Boundary close uses situational `phase_change` before 90-min gap fallback.
- [ ] Suggest output is draft only; nothing auto-promoted.
- [ ] Failed suggest retries from service state; no orphan high-sig turns without `memory_window_id` after retry window.

---

## Implementation order

1. Schemas + registry + channels.yaml + catalog test.
2. sql-writer: outbox publish after chat.history commit.
3. sql-writer: spark_meta patch merge handler.
4. `orion/memory/consolidation_classify.py` + tests.
5. Scaffold `orion-memory-consolidation` (classify + patch loop).
6. Window state + boundary logic.
7. suggest_runner + draft persistence + turn stamps.
8. Failed-window retry job.
9. Root env, docker-compose, smoke test: hub turn → scores in DB → boundary → draft.

---

## Explicitly rejected (v1)

- Classify in spark introspector light/heavy path.
- Spark candidate as memory trigger.
- Recovery cron scanning for unclassified rows (coverage backfill).
- Depending on `spark.telemetry` backfill for scores.
- Asking operator to lower gateway spark threshold as part of memory design.

---

## Related files

- `services/orion-hub/scripts/chat_history.py` — canonical turn publish
- `services/orion-hub/scripts/websocket_handler.py` — `trace_id` = `correlation_id`
- `services/orion-sql-writer/app/worker.py` — chat.history persist + new outbox/patch
- `services/orion-cortex-exec/app/situation.py` — phase bounds
- `orion/discussion_window/sql_fetch.py` — gap fallback
- `orion/memory_graph/dto.py` — `SuggestDraftV1`
- `services/orion-llm-gateway/app/llm_uncertainty.py` — logprob extraction reference
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`
