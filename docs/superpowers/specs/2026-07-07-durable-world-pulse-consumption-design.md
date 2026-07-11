# Durable World-Pulse → Concept-Induction Consumption — Design

- Date: 2026-07-07
- Status: Proposal (pending Juniper approval)
- Owner: Juniper / Orion
- Related services: `orion-world-pulse` (producer), `orion-spark-concept-induction` (consumer)
- Motivation trace: live 2026-07-07 investigation — world-pulse run `aa24753b` published `world.pulse.run.result.v1`; `orion-actions` received it, `orion-spark-concept-induction` did not. The grounded episode journal never fired because the triggering event was silently lost on pub/sub.

## Arsonist summary

The autonomy episode journal works (verified live: real Firecrawl articles, honest salience), but it is fed by a **fire-and-forget pub/sub channel with no replay**. `world.pulse.run.result.v1` fires roughly once per world-pulse run. The concept-induction worker drains all 13 intake channels in a **single serial `pubsub.listen()` loop** and runs the full episode path (Firecrawl fetch + a journal-compose RPC with a 120s timeout) **inline** in that loop. Any moment the loop is not draining — a slow handler, a transient socket/output-buffer event on the Redis server — a rare once-per-run publish is dropped with **zero trace**. High-frequency channels (`self_state` every few seconds) survive statistically; a once-per-run event has exactly one chance. The existence of `scripts/replay_world_pulse_run_to_bus.py` shows this is already a recurring, hand-patched failure.

Fix the transport for this one must-not-miss seam: put `world.pulse.run.result.v1` on a **durable Redis Stream with a consumer group**, using the primitive the repo already has (`RedisStreamWorkQueue`), and consume it in a **dedicated task** decoupled from the high-frequency pub/sub loop.

## Current architecture

| Hop | File / symbol | Behavior |
|---|---|---|
| Producer | `services/orion-world-pulse/app/services/pipeline.py` → `bus.publish("orion:world_pulse:run:result", env)` | pub/sub only, no durability |
| Transport | Redis pub/sub channel `orion:world_pulse:run:result` | no replay, no ack, no per-subscriber buffering guarantee |
| Consumer | `orion/spark/concept_induction/bus_worker.py:1054` `async with self.bus.subscribe(*patterns)` → serial `async for msg in iter_messages(pubsub)` → inline `handle_envelope` (fetch + 120s journal RPC) | one blocked/half-open moment = permanent loss |
| Bus chassis | `orion/core/bus/async_service.py` `OrionBusAsync` | pub/sub + RPC only; pubsub connection has `socket_timeout=None`, **no** `health_check_interval` |

### Already-present durable primitive (reuse, do not reinvent)

`orion/core/bus/work_queue.py` `RedisStreamWorkQueue`:
- `enqueue(stream, envelope, maxlen=...)` — `XADD` with `OrionCodec` envelope bytes + trim.
- `ensure_group`, `read_group(block_ms)` — `XGROUP CREATE` (BUSYGROUP-safe) + `XREADGROUP >`.
- `ack`, `pending_summary`, `pending_range`, `autoclaim(min_idle_ms)` — PEL inspection + crash recovery.
- `send_to_dlq` / `send_malformed_to_dlq` — dead-letter for poison messages; per-entry decode isolation.

This is tested (`orion/core/bus/tests/test_resilience.py`, work_queue tests) and codec-compatible with existing envelopes.

## Goals

1. Zero lost `world.pulse.run.result.v1` for concept-induction across worker restarts, brief disconnects, and slow handlers.
2. Decouple the rare, heavy-to-process episode trigger from the high-frequency pub/sub loop.
3. Observability: lag/pending is inspectable (`XINFO`), and a stuck message goes to a DLQ instead of vanishing.
4. Reuse `RedisStreamWorkQueue`. No new stream framework.

## Non-goals

- Migrating other consumers (`orion-hub`, `orion-actions`) off pub/sub. They keep pub/sub for back-compat. This patch is scoped to the one seam that silently lost data.
- A bus-wide pub/sub→stream migration. That is a separate, larger initiative.
- Changing episode-journal content, salience, or the seed builder (already shipped in `feat/grounded-episode-journal`).
- Replacing `OrionBusAsync` pub/sub. Streams live alongside it.

## Design (thin seam)

### Seam 1 — Producer dual-writes to a durable stream

`orion-world-pulse` keeps its existing `bus.publish(...)` (hub/actions unchanged) **and** additionally enqueues the same envelope to a stream:

```python
# services/orion-world-pulse: after building the run-result envelope
await bus.publish("orion:world_pulse:run:result", env)          # existing, back-compat
await work_queue.enqueue(STREAM_WORLD_PULSE_RUN_RESULT, env, maxlen=WP_RUN_RESULT_STREAM_MAXLEN)
```

- Stream key: `orion:stream:world_pulse:run:result` (distinct from the pub/sub channel; Redis stream vs pub/sub are separate keyspaces).
- `maxlen` (approximate) bounds memory; a few runs/day → small (default 1000).
- Dual-write is additive and behind a flag (`WP_RUN_RESULT_STREAM_ENABLED`); off = today's behavior.

### Seam 2 — Consumer reads the stream in a dedicated task

In `concept_induction/bus_worker.py`:
- **Remove** `orion:world_pulse:run:result` from `intake_channels` (the pub/sub list).
- Add a second async task `run_stream_consumer()` that:
  1. `await queue.ensure_group(STREAM, GROUP, start_id="$")` — `GROUP = cg:concept-induction`, `consumer = f"{node}:{pid}"`.
  2. Loop: `batch = await queue.read_group(STREAM, GROUP, consumer, count=1, block_ms=5000)`.
  3. For each `StreamMessage`: reuse the **same** `handle_envelope(msg.envelope, STREAM)` path, then `await queue.ack(STREAM, GROUP, msg.message_id)`. Ack only after successful handling.
  4. On handler exception: leave unacked → retried via periodic `autoclaim(min_idle_ms=WP_AUTOCLAIM_IDLE_MS)`; after `WP_MAX_ATTEMPTS` deliveries send to DLQ `orion:stream:world_pulse:run:result:dlq` and ack.
  5. Periodic `autoclaim` reclaims messages from a crashed prior consumer (crash recovery).
- Because this task is separate from the pub/sub loop, a 120s journal compose no longer blocks high-frequency channels, and vice versa — and a restart mid-compose redelivers the unacked message.

### Data flow (after)

```
world-pulse run
  ├─ bus.publish(orion:world_pulse:run:result)      → orion-hub, orion-actions   (pub/sub, unchanged)
  └─ queue.enqueue(orion:stream:world_pulse:run:result)  (durable)
        → concept-induction consumer group cg:concept-induction
        → read_group(block) → handle_envelope → substrate act → episode journal
        → ack   (or autoclaim redelivery on crash / DLQ on poison)
```

## Schema / bus / API changes

- No envelope/schema change. Same `world.pulse.run.result.v1` / `WorldPulseRunResultV1`, encoded by the same `OrionCodec`.
- `orion/bus/channels.yaml`: add a stream-kind entry for `orion:stream:world_pulse:run:result` (producer `orion-world-pulse`, consumer `orion-spark-concept-induction`) and a DLQ entry. Keep the existing pub/sub channel entry; update its `consumer_services` to drop the implicit `*`/concept-induction note if the catalog tracks that.
- `orion/schemas/registry.py`: no change (same schema).

## Env / config changes

Add to both services (`.env_example` + `settings.py` + `docker-compose.yml`), then `python scripts/sync_local_env_from_example.py`:

- `orion-world-pulse`: `WP_RUN_RESULT_STREAM_ENABLED` (default `false` → safe rollout), `WP_RUN_RESULT_STREAM_KEY=orion:stream:world_pulse:run:result`, `WP_RUN_RESULT_STREAM_MAXLEN=1000`.
- `orion-spark-concept-induction`: `WP_RUN_RESULT_STREAM_ENABLED`, `WP_RUN_RESULT_STREAM_KEY`, `WP_RUN_RESULT_STREAM_GROUP=cg:concept-induction`, `WP_RUN_RESULT_BLOCK_MS=5000`, `WP_RUN_RESULT_MAX_ATTEMPTS=5`, `WP_RUN_RESULT_AUTOCLAIM_IDLE_MS=120000`, `WP_RUN_RESULT_DLQ_KEY=orion:stream:world_pulse:run:result:dlq`.
- `ORION_BUS_URL` reused as-is for the stream client.

## Error handling / edge cases

- Consumer crash mid-handle: message stays in PEL, redelivered via `autoclaim` on the next consumer. No loss.
- Poison message (decode fails / handler always throws): after `WP_MAX_ATTEMPTS` → DLQ + ack, so it stops blocking the group. `read_group` already isolates per-entry decode errors.
- Group lag while consumer is down: messages accumulate in the stream (bounded by `maxlen`); consumer catches up on restart. If down longer than `maxlen` worth of runs, oldest trimmed — acceptable for a low-rate channel with generous maxlen.
- Dual-write partial failure (pub/sub ok, enqueue throws): log + continue; pub/sub path unaffected. Enqueue failure is a producer-side warning, not a run failure.
- Flag off: identical to today (pure pub/sub); trivial rollback.

## Testing

### Gate (unit)
- Producer: with flag on, `enqueue` called with the run-result envelope + configured `maxlen`; pub/sub still published (back-compat).
- Consumer: `read_group` → `handle_envelope` → `ack` happy path (fake `RedisStreamWorkQueue`); unacked-on-exception; `autoclaim` redelivery; DLQ after `MAX_ATTEMPTS`.
- Catalog: `python scripts/check_bus_channels.py` passes with the new stream + DLQ entries.
- Env parity: `python scripts/check_env_template_parity.py`.

### Live smoke (the real acceptance)
- Flag on both services. Trigger a world-pulse run **while concept-induction is stopped**; start it; confirm it reads the pending stream entry and fires the episode journal (proves durability — the exact failure from today).
- Restart concept-induction mid-compose; confirm redelivery via `autoclaim`, no duplicate journal beyond at-least-once (idempotency note below).
- `XINFO GROUPS orion:stream:world_pulse:run:result` shows the group + lag draining to 0.

## Acceptance checks

- [ ] A world-pulse run that occurs while concept-induction is down still produces an episode journal after it restarts.
- [ ] `world.pulse.run.result.v1` for concept-induction is acked; PEL drains; DLQ empty in the happy path.
- [ ] pub/sub consumers (`orion-hub`, `orion-actions`) still receive the event unchanged.
- [ ] Flag off → byte-for-byte current behavior.
- [ ] Unit + catalog + env-parity gates green; live durability smoke captured.

## Open question (needs a decision, at-least-once)

Streams are **at-least-once**. If a redelivery happens after a journal was already composed, we could double-compose. Options:
- (a) Idempotency guard keyed on `spawned_correlation_id` (skip if an episode journal for that run already exists) — recommended, small, and also protects against the existing replay script.
- (b) Accept rare duplicates (low rate, low harm).

Recommend (a) as part of this patch. Confirm.

## Rollback / disable

- `WP_RUN_RESULT_STREAM_ENABLED=false` on either service reverts to pure pub/sub. Producer stops enqueueing; consumer falls back to the pub/sub subscription (keep the pub/sub subscribe path behind the same flag so disabling restores channel-13 behavior).
- No schema migration to undo. Stream keys can be deleted if abandoning.

## Files likely to touch

- `services/orion-world-pulse/app/services/pipeline.py` — dual-write enqueue (flagged).
- `services/orion-world-pulse/{settings.py,.env_example,docker-compose.yml}` — stream flag/key/maxlen.
- `orion/spark/concept_induction/bus_worker.py` — dedicated stream consumer task; drop channel from pub/sub list when flag on.
- `orion/spark/concept_induction/settings.py` + `services/orion-spark-concept-induction/{.env_example,docker-compose.yml}` — consumer env.
- `orion/bus/channels.yaml` — stream + DLQ catalog entries.
- `orion/memory/.../episode idempotency` (if option (a)) — guard by `spawned_correlation_id`.
- Tests: `orion/spark/concept_induction/tests/`, `services/orion-world-pulse/tests/`.
- `.env` synced via `python scripts/sync_local_env_from_example.py`.

## Recommended next patch

Land Seam 1 + Seam 2 behind `WP_RUN_RESULT_STREAM_ENABLED` in one branch with the durability live smoke, plus idempotency option (a). Producer and consumer are separable if we want two smaller PRs (contract/producer, then consumer), but the seam is small enough for one.
