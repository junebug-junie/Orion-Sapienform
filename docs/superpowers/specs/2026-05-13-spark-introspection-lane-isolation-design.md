# Spark Introspection Lane Isolation — Design Spec

**Date:** 2026-05-13  
**Status:** Pending stakeholder review of this document  
**First merged vertical slice (confirmed):** Phase 1 only — `orion-spark-introspector` bounded concurrency, semaphore around heavy introspection RPC, pressure/skip/degrade behavior, Redis-backed idempotency by `trace_id`. Phases 2–5 remain program-of-record but are deferred past that slice.

---

## Executive summary

Spark introspection is a **post-turn cognitive afterimage**, not part of the synchronous Hub chat response path. Today, after a turn returns, `spark.candidate` can drive `spark-introspector` back through cortex-orch → cortex-exec → LLM gateway, sharing execution and model capacity with the next user message.

The durable fix is **lane isolation** (chat vs spark vs background) with **queue semantics for work** and **PubSub for broadcasts**. The **first delivery** optimizes Hub latency by implementing **pressure relief inside spark-introspector** without yet splitting exec channels or Redis Streams.

---

## Problem statement

### Symptom

A user can send Hub message B while spark introspection for turn A is still running; message B may wait behind spark-related work on shared exec/LLM paths.

### Root cause

Architectural coupling, not missing provenance. `spark.candidate` already carries enough data (`trace_id`, prompt, response, spark metadata, correlation/session via metadata) to process asynchronously and persist by key.

Blocking arises from shared execution arteries:

```text
Hub chat turn returns
  -> spark candidate published
    -> spark-introspector consumes candidate
      -> cortex.orch.request (introspect_spark)
        -> orch -> exec -> LLM gateway / GPU
              -> next chat turn may wait
```

### Additional hazard

Scaling intake by adding Redis PubSub subscribers on the same **work** channel duplicates processing (broadcast semantics), not competing-consumer semantics.

---

## Design principles

1. **Chat is sacred** — user-facing chat must not wait on post-turn cognition.
2. **Spark is asynchronous** — may lag, batch, degrade, or skip under pressure.
3. **Trace IDs preserve causality** — completion order across turns is not required if persistence keys are correct.
4. **Concurrency is bounded** — no unbounded parallel LLM/GPU calls from spark.
5. **Lanes before replicas** — separate traffic before horizontal scale.
6. **Queues for work, PubSub for broadcasts** — keep PubSub for telemetry/events; use streams/consumer groups for assignable work (later phase).
7. **No duplicated cognition** — idempotency and/or single consumer per assignable unit.
8. **Observability is part of the fix** — queue depth, lane, wait time, skip/degrade, GPU route (phased; Hub card in Phase 5).

---

## Repository grounding (2026-05-13)

Facts checked in-tree:

- **`Hunter` in spark-introspector is sequential.** `services/orion-spark-introspector/app/main.py` constructs `Hunter(...)` without `concurrent_handlers`, so one handler runs at a time across all subscribed kinds (candidates, traces, vector upserts, signals).
- **Heavy path is `handle_candidate` → `bus.rpc_request` to cortex.** `services/orion-spark-introspector/app/worker.py` emits telemetry and updates tissue, then opens `OrionBusAsync`, `connect()`, `rpc_request` to `settings.channel_cortex_request` with `verb_name=introspect_spark`, then `close()`. This matches `docs/platform_routing_wiring_map.md` (spark-introspector → `orion:cortex:request` → orch → exec → LLM).
- **Single main exec request channel by default.** `orion-cortex-exec` uses `CHANNEL_EXEC_REQUEST` default `orion:cortex:exec:request` (`services/orion-cortex-exec/app/settings.py`). Lane split (Phase 2) is additive configuration and deployment.
- **In-process dedupe exists but is not cross-replica.** `_CANDIDATE_QUALITY` / `_CANDIDATE_TELEM_EMITTED` in `worker.py` reduce duplicate handling within one process; Redis idempotency (Phase 1) addresses duplicates, retries, and future multi-instance intake.

**Implementation note for Phase 1:** enabling `concurrent_handlers=True` may increase concurrent `OrionBusAsync` instances (one per in-flight candidate path). If Redis connection pressure appears, spec a follow-up to reuse a shared connected bus for RPC while keeping the semaphore around the heavy call.

---

## Target architecture (north star)

### Lane model

| Lane         | Purpose                         | Priority   | Blocking tolerance |
|-------------|----------------------------------|------------|----------------------|
| chat        | interactive Hub messages       | highest    | near zero            |
| spark       | post-turn introspection        | low/medium | high                 |
| background  | dreams, metacog, scheduled     | low        | high                 |

### Desired flow

```text
Turn N chat -> chat lane -> response to Hub -> spark candidate -> queued or bounded async spark work
Turn N+1 chat -> chat lane immediately; does not wait for Turn N spark completion
```

### Final bus semantics

- **Broadcast events:** Redis PubSub (health, telemetry, traces, UI notifications).
- **Work requests:** Redis Streams with consumer groups (exec/LLM/recall lanes) — Phase 4.

---

## Phase 1 — First merged slice (user choice **A**)

**Goal:** Reduce immediate gumming without exec channel surgery or stream migration.

**Scope:**

- Spark-introspector: concurrent `Hunter` handlers with audited shared state.
- Semaphore around **only** the expensive cortex RPC (`rpc_request` / orch path), not around lightweight telemetry, tissue updates, or signal handlers unless a separate lock is required for shared mutable tissue state.
- Environment-driven limits, staleness, drop-on-pressure, optional min interval between heavy runs.
- Redis idempotency: `SETNX` inflight + `done` key + cleanup (TTLs as in original task text).
- Structured logs: `spark_introspection_start`, `spark_introspection_complete`, `spark_introspection_skipped`, `spark_introspection_degraded` (exact logger event names or JSON fields to match existing service logging conventions).

**Settings (illustrative names; finalize in implementation plan):**

- `SPARK_INTROSPECTION_MAX_INFLIGHT` (default `1`)
- `SPARK_INTROSPECTION_TIMEOUT_SEC`
- `SPARK_INTROSPECTION_QUEUE_MAX_AGE_SEC` (staleness for skipping heavy path)
- `SPARK_INTROSPECTION_DROP_ON_PRESSURE`
- `SPARK_INTROSPECTION_MIN_INTERVAL_SEC`

**Pressure behavior:**

- Stale candidate: skip heavy introspection; still emit lightweight telemetry where applicable.
- Saturated semaphore with drop-on-pressure: skip or degrade low-salience candidates per policy; log `trace_id` / correlation / session.
- On skip/degrade, **do not** omit required lightweight paths already defined as “always emit” in code review.

**Acceptance tests (Phase 1):**

1. Hub message A returns; spark candidate emitted.
2. While spark A heavy path is in flight, Hub message B starts and completes without waiting for spark A’s introspection RPC.
3. Spark A remains keyed by original `trace_id`.
4. Duplicate `trace_id` heavy introspection does not create duplicate persisted introspection rows (Redis + existing in-process dedupe behavior documented).
5. Unit tests: semaphore behavior; duplicate `trace_id` skip; stale candidate skips heavy LLM path but emits telemetry.

**Risk:** `concurrent_handlers=True` can expose races in module-level `TISSUE`, caches, and counters. Mitigation: keep default `MAX_INFLIGHT=1` until tests prove safety; add locks around tissue mutation if review finds gaps.

---

## Phase 2 — Split spark from chat at exec intake

Dedicated channels (catalog + env):

- `orion:cortex:exec:request:chat`
- `orion:cortex:exec:request:spark`
- `orion:cortex:exec:request:background`

Orch resolves lane from verb/mode; exec runs as separate processes per lane. **Constraint until Phase 4:** at most one consumer per PubSub lane to avoid duplicate exec.

Routing policy examples: `chat_general` / `chat_quick` → chat; `introspect_spark` → spark; `dream_cycle`, `log_orion_metacognition` → background; defaults for other verbs as agreed in implementation.

---

## Phase 3 — LLM / GPU route isolation

Lane-specific LLM routes and profiles (e.g. `spark.introspection.v1` on background stack). No silent fallback from spark/background to chat route unless explicitly configured. Degraded telemetry when background route is unavailable.

---

## Phase 4 — Redis Streams / consumer groups

Add `enqueue`, `consume_group`, `ack` (or equivalent) to bus layer; `QueueRabbit`-style chassis; stream names and consumer groups per lane; idempotency keys `kind:trace_id:step_name:service`; pending recovery policy.

---

## Phase 5 — Observability and operator controls

Standard fields per work item (`trace_id`, `correlation_id`, `session_id`, `lane`, `priority`, `verb`, queue wait, runtime, skip/degrade reasons, `llm_lane`, `model_profile`). Hub “Cognition Lanes” card and read-only inspect actions first; mutating controls later.

---

## Implementation order (full program)

1. Spark-introspector bounded concurrency + idempotency (Phase 1).
2. Lane metadata and logs without routing change (can overlap early with Phase 1 logging).
3. Dedicated exec intake channels + routing (Phase 2).
4. Route `introspect_spark` to spark exec lane.
5. Spark/background LLM profiles and routes (Phase 3).
6. Hub lane observability (Phase 5 partial).
7. PubSub work intake → Redis Streams (Phase 4).
8. Horizontal workers per lane with consumer groups.

---

## Task blocks (for implementation planning)

| Block | Scope |
|-------|--------|
| **A** | Phase 1 in `services/orion-spark-introspector`: `concurrent_handlers`, semaphore, env, skip/degrade, Redis idempotency, logs, tests. |
| **B** | Lane metadata propagation (orch/exec/gateway logs) without changing routing. |
| **C** | Dedicated exec lanes + orch resolver + multi-exec deploy shape. |
| **D** | LLM route isolation and spark profile. |
| **E** | Streams, `QueueRabbit`, idempotency keys, recovery policy. |

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Races in spark-introspector under concurrency | Default inflight 1; semaphores; tissue lock if needed; audit globals before raising inflight. |
| GPU oversubscription | Phase 3 routes; deny chat fallback by default. |
| PubSub duplicate workers | One worker per lane until Phase 4; Redis idempotency in Phase 1. |
| Out-of-order spark completion | Acceptable; persist by `trace_id` and timestamps; UI treats as post-turn. |

---

## Definition of done (full program)

1. User-facing chat proceeds while spark introspection runs.
2. Spark work is lane-tagged `spark` in telemetry.
3. Spark has bounded inflight concurrency.
4. Spark does not use the chat GPU/model route by default.
5. Duplicate spark jobs guarded by idempotency.
6. Lane health observable in Hub/logs.
7. Horizontal scaling uses queue/consumer-group semantics, not PubSub broadcast for work.

---

## Architecture doctrine (one line)

Spark introspection is a causally linked afterimage of a chat turn, not a dependency of the next turn; it belongs on a bounded background cognition lane with trace-keyed provenance, explicit degrade/skip behavior, and queue semantics before horizontal scale.
