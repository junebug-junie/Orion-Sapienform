# 2026-05-13 — Redis Streams WorkQueue + QueueRabbit (Phase 4A/4B)

## Goal

Add **Redis Streams + consumer groups** beside existing **OrionBusAsync PubSub** so one-worker workloads are assigned to exactly one consumer per delivery, with retry, DLQ, pending recovery, and service-level idempotency hooks. PubSub remains for broadcasts and reply channels.

## Doctrine

- **PubSub** tells everyone something happened.
- **Streams** assign one worker to do the work.
- Streams are **at-least-once**; exactly-once is not claimed.

## Deliverables

| Artifact | Path |
|----------|------|
| Work queue + RPC helper | `orion/core/bus/work_queue.py` |
| Queue chassis | `orion/core/bus/queue_service_chassis.py` |
| Unit tests | `tests/test_work_queue.py`, `tests/test_queue_service_chassis.py` |

## Constraints (non-negotiable)

- Do **not** migrate chat, recall, spark, exec, LLM, RDF, or vector traffic in this phase.
- Do **not** replace `OrionBusAsync` or change its public API.
- `BaseEnvelope` is frozen — copy envelopes when updating work/retry metadata.
- `QueueRabbit` subclasses `BaseChassis` (heartbeat, errors, identity, shutdown).
- `QueueRabbit` uses `OrionBusAsync` for PubSub replies (optional `reply_bus` override for reply publishes only); stream I/O uses a **separate** `RedisStreamWorkQueue` client.
- `queue_rpc_request`: **subscribe to `reply_channel` before enqueue** (avoid fast-reply races).
- Decode failures: DLQ + ack if DLQ configured; else leave pending + loud logs.
- Optional live Redis test gated by `ORION_REDIS_STREAM_TEST_URL`.

## Acceptance

Foundation complete when items 1–12 in the originating task specification are satisfied (unchanged PubSub/Rabbit/Hunter, stream ops, chassis behavior, tests, optional two-consumer integration).

## Execution note

Implemented in-repo per `/executing-plans` request; full step-by-step spec is preserved in the task transcript / agent log.
