# PR: Phase 4A/4B — Redis Streams work queue + QueueRabbit chassis

## Branch

- **Head:** `feat/bus-redis-streams-workqueue-phase4` (push to `origin`)
- **Base:** `main`

Create PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/bus-redis-streams-workqueue-phase4

---

## Summary

Adds a **foundation-only** substrate for safe work assignment: **Redis Streams + consumer groups** beside the existing **OrionBusAsync** PubSub bus. PubSub remains the nervous system for broadcasts, telemetry, and reply channels; streams carry **“exactly one consumer should do this job”** workloads with retry, optional DLQ, pending recovery (`XAUTOCLAIM`), and `queue_rpc_request` (subscribe to `reply_to` **before** enqueue to avoid fast-reply races). **No** migration of chat, recall, spark, exec, LLM, RDF, or vector traffic; **no** change to `OrionBusAsync`, Rabbit, or Hunter APIs.

---

## What changed

| Area | Change |
|------|--------|
| **`orion/core/bus/work_queue.py`** | `RedisStreamWorkQueue`: `XADD` / `XREADGROUP` / `XACK` / `XPENDING` / `XAUTOCLAIM`, `enqueue`, `requeue`, `send_to_dlq`, `send_malformed_to_dlq`, optional XINFO helpers. **`ReadGroupBatch`**: per-entry decode so one bad envelope in a batch does not strand earlier valid messages. `queue_rpc_request`: subscribe then enqueue; `reply_to` vs `reply_channel` mismatch warning. Owned Redis client closed on `close()`; injected client left open. |
| **`orion/core/bus/queue_service_chassis.py`** | **`QueueRabbit(BaseChassis)`**: consumer-group loop, handler → optional PubSub reply → ack; retry/requeue and DLQ after `max_attempts`; `not_before` / `expires_at` + `stale_policy`; decode-error path; reply publish failures routed through same retry/DLQ path as handler errors; concurrent mode with **bounded** reads (wait for in-flight capacity); `work_queue.close()` in `_run` `finally`. |
| **Tests** | `tests/test_work_queue.py`, `tests/test_queue_service_chassis.py`: mocked Redis; batch decode + malformed DLQ coverage; live two-consumer test **skipped** unless `ORION_REDIS_STREAM_TEST_URL` is set. |
| **Plan** | `docs/superpowers/plans/2026-05-13-redis-streams-workqueue-queuerabbit-phase4-implementation.md` |

---

## API note (callers)

- `RedisStreamWorkQueue.read_group` and `autoclaim` return **`ReadGroupBatch`** (`messages`, `decode_errors`), not a bare `list[StreamMessage]`.

---

## Verification (ran locally)

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_work_queue.py tests/test_queue_service_chassis.py -q --tb=short
# 29 passed, 1 skipped (live Redis test without ORION_REDIS_STREAM_TEST_URL)

PYTHONPATH=. ./venv/bin/python -m compileall -q orion/core/bus/work_queue.py orion/core/bus/queue_service_chassis.py
```

**Optional live test:** set `ORION_REDIS_STREAM_TEST_URL=redis://localhost:6379/15` and re-run pytest to exercise two consumers, one message.

---

## Risk / follow-up

- **Adoption:** Wire services to `QueueRabbit` + `RedisStreamWorkQueue` in later PRs; this PR does not enable any production queue by default.
- **Stash:** Local `orion-hub` edits and `scripts/git-stash-table.sh` were stashed before branching from `main` (`git stash list` — pops when you are ready).

---

## Suggested PR title

**feat(bus): Redis Streams work queue + QueueRabbit chassis (Phase 4 foundation)**
