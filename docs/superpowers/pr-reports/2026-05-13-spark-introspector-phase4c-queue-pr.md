# PR: Spark introspector — queue-backed heavy introspection (Phase 4C)

## Branch

- **Head:** `feat/spark-introspector-queue-backed-heavy-introspection`
- **Suggested base:** `feat/bus-redis-streams-workqueue-phase4` (this service uses `QueueRabbit` + `RedisStreamWorkQueue` from that branch). If the bus PR is already merged into `main`, rebase onto `main` and open against `main`.

Create PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/spark-introspector-queue-backed-heavy-introspection

---

## Summary

Implements **Phase 4C** for `orion-spark-introspector`: optional **Redis Stream** work queue for **heavy** spark introspection (cortex RPC), alongside existing **inline** heavy path. Rich candidates enqueue a versioned job envelope (`spark.introspection.job.v1`); a **`QueueRabbit`** consumer in app lifespan claims work, runs `run_heavy_spark_introspection` with the **decoded job** so cortex **options** (execution/LLM lanes, `allow_degrade`, `allow_chat_fallback`, `max_tokens`, `timeout_sec`) match the enqueue-time snapshot. **Redis-backed idempotency** (inflight SETNX + done marker) is fail-closed on Redis errors when enabled. **Operator debug** for queue depth/pending is gated behind env + **header-only** shared secret with fixed-length digest comparison. **`.env_example`** and **`docker-compose.yml`** expose queue, idempotency, and debug settings (local `.env` remains gitignored; operators should copy from `.env_example`).

---

## What changed

| Area | Change |
|------|--------|
| **`app/queue_jobs.py`** | `SparkIntrospectionJobV1`, `build_spark_introspection_job_envelope`, `extract_spark_introspection_job`, stable `envelope_correlation_uuid`. |
| **`app/queue_worker.py`** | Consumer handler: expiry, `run_heavy_spark_introspection(..., from_queue=True, job=job)`, logging. |
| **`app/worker.py`** | Enqueue path when queue enabled; inline fallback when enqueue fails (if inline enabled); `run_heavy` accepts optional `job`, resolves **cortex correlation** once, returns **`correlation_id`** consistent with orch envelopes; cortex **`wait_for` timeout** follows job/settings timeout. |
| **`app/introspection_guard.py`** | Shared Redis client, `is_done` / `try_claim_inflight` / `mark_done` / `release_inflight`; `is_done` documents fail-open read vs fail-closed claim. |
| **`app/main.py`** | Lifespan: `QueueRabbit` + `RedisStreamWorkQueue` when queue enabled; `GET /debug/spark/introspection-queue` (mirrored under `/spark`). |
| **`app/settings.py`** | Queue stream/group/consumer/DLQ, reclaim, stale policy, inline vs queue toggles, debug enable + token. |
| **`.env_example`**, **`docker-compose.yml`** | Phase 4C and debug variables; compose passes non-secret `SPARK_INTROSPECTION_*` defaults. |
| **Tests** | Jobs, worker, candidate enqueue, guard, debug route, correlation resolve, redis-unavailable skip, optional Redis integration (`ORION_REDIS_STREAM_TEST_URL`); **`conftest.py`** `chdir` into service dir for `app.main` imports. |
| **Plan** | `docs/superpowers/plans/2026-05-13-phase4c-queue-backed-spark-heavy-introspection.md` |

---

## Verification (ran locally)

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-spark-introspector/tests/ -q --tb=short
# 28 passed, 1 skipped (live Redis stream test without ORION_REDIS_STREAM_TEST_URL)
```

---

## Risk / follow-up

- **Catalog:** Redis stream keys (`SPARK_INTROSPECTION_QUEUE_STREAM` / DLQ) are not entries in `orion/bus/channels.yaml` (stream transport vs Rabbit routing catalog); document or add if you want a single operator index.
- **WebSocket** `introspection.update` still uses `trace_id` as `correlation_id`; bus/cortex paths use resolved UUID — align only if product wants UI correlation unified.
- **Depends on** bus Phase 4 branch for `orion.core.bus.queue_service_chassis` / `work_queue` imports.

---

## Suggested PR title

**feat(spark-introspector): queue-backed heavy introspection (Phase 4C)**

---

## Suggested PR description (copy-paste)

### What & why

Heavy spark introspection (cortex orch RPC) can now run **off the hot candidate Hunter path** via a **Redis Stream consumer group** (`QueueRabbit` + `RedisStreamWorkQueue`), while keeping an **inline** path for smaller deployments or enqueue failure fallback.

### Key behavior

- **Job envelope `spark.introspection.job.v1`** freezes candidate text, meta, lanes, and cortex option fields at enqueue time; the worker applies those fields when executing heavy introspection.
- **Idempotency:** Redis inflight + done keys; claim path fails closed when Redis is unavailable (no duplicate heavy runs across workers).
- **Correlation:** Stable UUID resolution for cortex / publish envelopes; `run_heavy` return dict uses the same resolved id string.
- **Debug:** Queue status JSON behind `SPARK_INTROSPECTION_QUEUE_DEBUG_ENABLE` + `SPARK_INTROSPECTION_QUEUE_DEBUG_TOKEN` (header `X-Spark-Introspection-Debug-Token` only).

### How to test

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-spark-introspector/tests/ -q
```

Optional: set `ORION_REDIS_STREAM_TEST_URL` for the stream integration test.

### Checklist

- [ ] Base branch contains bus `QueueRabbit` / `RedisStreamWorkQueue` (or rebase after bus PR merges).
- [ ] Operators update `.env` from `.env_example` for queue profile if enabling queue.
