# Spark Introspection Phase 1 (Task Block A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bounded async handling and pressure relief inside `orion-spark-introspector` so heavy `introspect_spark` cortex RPC work is semaphored, idempotent across retries/replicas (Redis), skippable under staleness or saturation, and observable via structured log lines—without changing exec channels or orch routing.

**Architecture:** Keep `Hunter` as the bus consumer but enable `concurrent_handlers=True` so traces/signals/upserts are not blocked behind a slow candidate. Serialize **mutations** on shared `TISSUE` / candidate-cache state with one `asyncio.Lock` for the candidate-heavy path. Wrap **only** the `OrionBusAsync` connect + `rpc_request` + teardown (and publish of introspection result) in an `asyncio.Semaphore(settings.spark_introspection_max_inflight)` plus optional non-waiting acquire when `SPARK_INTROSPECTION_DROP_ON_PRESSURE=true`. Use Redis keys on the same URL as `ORION_BUS_URL` for inflight `SETNX` and done markers keyed by `trace_id`. Emit four log patterns: `spark_introspection_start`, `spark_introspection_complete`, `spark_introspection_skipped`, `spark_introspection_degraded`.

**Tech Stack:** Python 3.12, FastAPI/uvicorn (unchanged), `redis==5.0.4` (`redis.asyncio`), Pydantic settings, pytest, existing `OrionBusAsync` / `BaseEnvelope` types.

**Spec source:** `docs/superpowers/specs/2026-05-13-spark-introspection-lane-isolation-design.md` (Phase 1 / Task Block A only).

---

## File map

| File | Responsibility |
|------|----------------|
| Modify: `services/orion-spark-introspector/app/settings.py` | New env-backed fields for semaphore, timeouts, staleness, drop-on-pressure, min interval, Redis TTLs, idempotency toggle, optional Redis URL override. |
| Modify: `services/orion-spark-introspector/app/main.py` | Pass `concurrent_handlers=True` into `Hunter`. |
| Create: `services/orion-spark-introspector/app/introspection_guard.py` | Thin async Redis helper: `try_claim_inflight`, `mark_done`, `release_inflight`, `is_done` using key prefix from settings. |
| Modify: `services/orion-spark-introspector/app/worker.py` | `_candidate_path_lock`, integrate guard + staleness (`env.created_at`) + min-interval + semaphore + `asyncio.wait_for` RPC timeout + structured logs; keep telemetry/tissue ordering per spec. |
| Modify: `services/orion-spark-introspector/.env_example` | Document every new variable (AGENTS.md parity). |
| Create: `services/orion-spark-introspector/pytest.ini` | `pythonpath = .`, `testpaths = tests` so `from app.*` resolves when pytest cwd is the service directory. |
| Create: `services/orion-spark-introspector/tests/test_introspection_guard.py` | Unit tests for Redis guard logic with `AsyncMock`. |
| Create: `services/orion-spark-introspector/tests/test_handle_candidate_pressure.py` | Unit tests for staleness skip, duplicate done skip, semaphore/drop behavior using monkeypatch/mocks (no real Redis/bus). |

---

### Task 1: Settings and operator contract (`.env_example`)

**Files:**
- Modify: `services/orion-spark-introspector/app/settings.py`
- Modify: `services/orion-spark-introspector/.env_example`

- [ ] **Step 1: Add fields to `Settings`**

Append to `class Settings` (types and defaults exactly as below; use `Field` + `AliasChoices` where multiple env names help operators):

```python
    # Spark heavy introspection pressure / idempotency (Phase 1 lane isolation)
    spark_introspection_max_inflight: int = Field(1, alias="SPARK_INTROSPECTION_MAX_INFLIGHT")
    spark_introspection_timeout_sec: float = Field(45.0, alias="SPARK_INTROSPECTION_TIMEOUT_SEC")
    spark_introspection_queue_max_age_sec: float = Field(180.0, alias="SPARK_INTROSPECTION_QUEUE_MAX_AGE_SEC")
    spark_introspection_drop_on_pressure: bool = Field(True, alias="SPARK_INTROSPECTION_DROP_ON_PRESSURE")
    spark_introspection_acquire_timeout_sec: float = Field(0.0, alias="SPARK_INTROSPECTION_ACQUIRE_TIMEOUT_SEC")
    spark_introspection_min_interval_sec: float = Field(0.0, alias="SPARK_INTROSPECTION_MIN_INTERVAL_SEC")
    spark_introspection_idempotency_enable: bool = Field(True, alias="SPARK_INTROSPECTION_IDEMPOTENCY_ENABLE")
    spark_introspection_redis_url: str | None = Field(None, alias="SPARK_INTROSPECTION_REDIS_URL")
    spark_introspection_key_prefix: str = Field("spark:introspection", alias="SPARK_INTROSPECTION_KEY_PREFIX")
    spark_introspection_inflight_ttl_sec: int = Field(300, alias="SPARK_INTROSPECTION_INFLIGHT_TTL_SEC")
    spark_introspection_done_ttl_sec: int = Field(86400, alias="SPARK_INTROSPECTION_DONE_TTL_SEC")
```

Semantics (document in `.env_example` comments):

- `SPARK_INTROSPECTION_ACQUIRE_TIMEOUT_SEC`: when `>0`, `asyncio.wait_for(sem.acquire(), timeout=...)`; when `0` and `DROP_ON_PRESSURE=true`, treat as immediate timeout if semaphore busy (no queueing). When `DROP_ON_PRESSURE=false`, use a sensible default acquire wait of `60.0` in code if this env is `0` (document that interaction in `.env_example`).
- `SPARK_INTROSPECTION_REDIS_URL`: if unset, use `orion_bus_url` for idempotency keys.

- [ ] **Step 2: Mirror all keys into `.env_example`**

Under a new section `# --- Spark introspection pressure (Phase 1) ---`, list each variable with a one-line comment matching the semantics above. Do not put real secrets; use placeholders for URLs if duplicated.

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/app/settings.py services/orion-spark-introspector/.env_example
git commit -m "feat(spark-introspector): add Phase 1 introspection pressure settings"
```

---

### Task 2: Redis introspection guard module

**Files:**
- Create: `services/orion-spark-introspector/app/introspection_guard.py`

- [ ] **Step 1: Implement the module**

```python
from __future__ import annotations

import logging
from typing import Optional

from redis.asyncio import Redis

logger = logging.getLogger("orion-spark-introspector")


def _inflight_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:inflight:{trace_id}"


def _done_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:done:{trace_id}"


async def is_done(redis: Redis, *, prefix: str, trace_id: str) -> bool:
    v = await redis.get(_done_key(prefix, trace_id))
    return v is not None and v != b""


async def try_claim_inflight(
    redis: Redis,
    *,
    prefix: str,
    trace_id: str,
    owner: str,
    ttl_sec: int,
) -> bool:
    """
    Returns True if this instance claimed inflight work (SETNX ok).
    Returns False if another holder already claimed.
    """
    key = _inflight_key(prefix, trace_id)
    ok = await redis.set(key, owner, nx=True, ex=int(ttl_sec))
    return bool(ok)


async def mark_done(
    redis: Redis,
    *,
    prefix: str,
    trace_id: str,
    status: str,
    ttl_sec: int,
) -> None:
    await redis.set(_done_key(prefix, trace_id), status, ex=int(ttl_sec))


async def release_inflight(redis: Redis, *, prefix: str, trace_id: str) -> None:
    await redis.delete(_inflight_key(prefix, trace_id))
```

- [ ] **Step 2: No standalone test file yet** (covered in Task 4).

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/app/introspection_guard.py
git commit -m "feat(spark-introspector): add Redis introspection idempotency guard"
```

---

### Task 3: `Hunter` concurrent handlers

**Files:**
- Modify: `services/orion-spark-introspector/app/main.py`

- [ ] **Step 1: Enable concurrent delivery**

Change the `Hunter(...)` construction to:

```python
    svc = Hunter(
        _cfg(),
        patterns=patterns,
        handler=multiplexer,
        concurrent_handlers=True,
    )
```

- [ ] **Step 2: Log at INFO that concurrent handlers are enabled** (one line after Hunter construction).

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/app/main.py
git commit -m "feat(spark-introspector): enable concurrent Hunter handlers"
```

---

### Task 4: Unit tests for `introspection_guard` (TDD against Redis contract)

**Files:**
- Create: `services/orion-spark-introspector/pytest.ini`
- Create: `services/orion-spark-introspector/tests/test_introspection_guard.py`

- [ ] **Step 1: Add `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Do **not** add `pytest-asyncio` unless you prefer it repo-wide; this repo’s `requirements-dev.txt` does not include it. Use `asyncio.run()` inside plain `def` tests.

- [ ] **Step 2: Write failing tests**

```python
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app import introspection_guard as ig


def test_try_claim_inflight_setnx_true():
    async def _go() -> None:
        redis = MagicMock()
        redis.set = AsyncMock(return_value=True)
        assert await ig.try_claim_inflight(redis, prefix="p", trace_id="t1", owner="n1", ttl_sec=30) is True
        redis.set.assert_awaited_once()
        assert redis.set.await_args.kwargs.get("nx") is True

    asyncio.run(_go())


def test_try_claim_inflight_setnx_false():
    async def _go() -> None:
        redis = MagicMock()
        redis.set = AsyncMock(return_value=None)
        assert await ig.try_claim_inflight(redis, prefix="p", trace_id="t1", owner="n1", ttl_sec=30) is False

    asyncio.run(_go())


def test_is_done_true_when_key_present():
    async def _go() -> None:
        redis = MagicMock()
        redis.get = AsyncMock(return_value=b"ok")
        assert await ig.is_done(redis, prefix="p", trace_id="t1") is True

    asyncio.run(_go())


def test_release_deletes_inflight():
    async def _go() -> None:
        redis = MagicMock()
        redis.delete = AsyncMock(return_value=1)
        await ig.release_inflight(redis, prefix="p", trace_id="t1")
        redis.delete.assert_awaited_once()

    asyncio.run(_go())
```

- [ ] **Step 3: Run tests (expect PASS after Task 2)**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-spark-introspector
../../venv/bin/python -m pytest tests/test_introspection_guard.py -q --tb=short
```

If `venv` missing, use `../../orion_dev/bin/python` per AGENTS.md.

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add services/orion-spark-introspector/pytest.ini services/orion-spark-introspector/tests/test_introspection_guard.py
git commit -m "test(spark-introspector): cover introspection Redis guard"
```

---

### Task 5: Worker integration — locks, staleness, semaphore, RPC timeout, logs, Redis lifecycle

**Files:**
- Modify: `services/orion-spark-introspector/app/worker.py`

**Order constraint (spec acceptance):** Run existing validation, `_emit_candidate_telemetry`, and `_update_tissue_from_candidate` (when applicable) **before** any check that `return`s without heavy RPC for **staleness** or **redis done**. Those skips must still leave telemetry/tissue work already executed for that candidate pass.

Design rules (implement heavy-path gates only after light paths above):

1. **Module-level state**

```python
import asyncio
from redis.asyncio import Redis

_INTRO_SEM: asyncio.Semaphore | None = None
_CANDIDATE_MUTATION_LOCK = asyncio.Lock()
_LAST_HEAVY_INTRO_MONO: float = 0.0
_REDIS_CLIENT: Redis | None = None


def _intro_sem() -> asyncio.Semaphore:
    global _INTRO_SEM
    if _INTRO_SEM is None:
        n = max(1, int(settings.spark_introspection_max_inflight))
        _INTRO_SEM = asyncio.Semaphore(n)
    return _INTRO_SEM


async def _redis_for_idempotency() -> Redis | None:
    global _REDIS_CLIENT
    if not settings.spark_introspection_idempotency_enable:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    url = settings.spark_introspection_redis_url or settings.orion_bus_url
    _REDIS_CLIENT = Redis.from_url(url, decode_responses=False)
    return _REDIS_CLIENT
```

Call `Redis.from_url` once; close on process shutdown is optional for Phase 1 (document as follow-up) or wire `lifespan` shutdown in `main.py` if low-cost.

2. **Staleness** — immediately after light paths, before heavy work:

```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
age_sec = (now - env.created_at).total_seconds()
if age_sec > float(settings.spark_introspection_queue_max_age_sec):
    logger.info(
        "spark_introspection_skipped trace_id=%s correlation_id=%s reason=stale age_sec=%.3f max_age_sec=%.3f",
        trace_id,
        str(env.correlation_id),
        age_sec,
        float(settings.spark_introspection_queue_max_age_sec),
    )
    return
```

3. **Redis done short-circuit** (skip heavy + skip re-publish if already done):

```python
redis = await _redis_for_idempotency()
if redis is not None and await introspection_guard.is_done(redis, prefix=settings.spark_introspection_key_prefix, trace_id=trace_id):
    logger.info(
        "spark_introspection_skipped trace_id=%s correlation_id=%s reason=redis_done",
        trace_id,
        str(env.correlation_id),
    )
    return
```

4. **Min interval** — global throttle:

```python
import time

global _LAST_HEAVY_INTRO_MONO
mono = time.monotonic()
if settings.spark_introspection_min_interval_sec > 0:
    delta = mono - _LAST_HEAVY_INTRO_MONO
    if _LAST_HEAVY_INTRO_MONO > 0 and delta < float(settings.spark_introspection_min_interval_sec):
        logger.info(
            "spark_introspection_degraded trace_id=%s correlation_id=%s reason=min_interval delta_sec=%.3f min_sec=%.3f",
            trace_id,
            str(env.correlation_id),
            delta,
            float(settings.spark_introspection_min_interval_sec),
        )
        return
```

5. **Tissue + cache mutations** — wrap existing candidate mutation sections that touch `TISSUE` / `_CANDIDATE_*` in `async with _CANDIDATE_MUTATION_LOCK:` (narrow scope: only the blocks that mutate shared state for this handler path, not the entire function).

6. **Heavy RPC region** — new inner async function `_heavy_introspect_rpc(...)` containing lines currently from `OrionCodec()` through `await bus.close()`, and wrap the body:

```python
owner = f"{settings.node_name}:{_PRODUCER_BOOT_ID}"
claimed = False
if redis is not None:
    claimed = await introspection_guard.try_claim_inflight(
        redis,
        prefix=settings.spark_introspection_key_prefix,
        trace_id=trace_id,
        owner=owner,
        ttl_sec=int(settings.spark_introspection_inflight_ttl_sec),
    )
    if not claimed:
        logger.info(
            "spark_introspection_skipped trace_id=%s correlation_id=%s reason=inflight_not_claimed",
            trace_id,
            str(env.correlation_id),
        )
        return

logger.info(
    "spark_introspection_start trace_id=%s correlation_id=%s",
    trace_id,
    str(env.correlation_id),
)

sem = _intro_sem()
acquire_timeout = float(settings.spark_introspection_acquire_timeout_sec)
if settings.spark_introspection_drop_on_pressure and acquire_timeout <= 0:
    acquire_timeout = 0.0  # immediate
elif not settings.spark_introspection_drop_on_pressure and acquire_timeout <= 0:
    acquire_timeout = 60.0

try:
    if acquire_timeout > 0:
        await asyncio.wait_for(sem.acquire(), timeout=acquire_timeout)
    else:
        await sem.acquire()
except asyncio.TimeoutError:
    if redis is not None and claimed:
        await introspection_guard.release_inflight(redis, prefix=settings.spark_introspection_key_prefix, trace_id=trace_id)
    logger.info(
        "spark_introspection_degraded trace_id=%s correlation_id=%s reason=semaphore_busy drop_on_pressure=%s",
        trace_id,
        str(env.correlation_id),
        settings.spark_introspection_drop_on_pressure,
    )
    return
```

Note: `asyncio.wait_for(..., timeout=None)` waits indefinitely—use that branch when not dropping.

After acquire, use `try/finally: sem.release()`.

Inside semaphore, run RPC with:

```python
try:
    msg = await asyncio.wait_for(
        bus.rpc_request(
            settings.channel_cortex_request,
            req,
            reply_channel=reply_channel,
            timeout_sec=float(settings.cortex_timeout_sec),
        ),
        timeout=float(settings.spark_introspection_timeout_sec),
    )
except asyncio.TimeoutError as e:
    logger.info(
        "spark_introspection_degraded trace_id=%s correlation_id=%s reason=rpc_timeout timeout_sec=%.3f",
        trace_id,
        str(env.correlation_id),
        float(settings.spark_introspection_timeout_sec),
    )
    # treat as RPC failure path for user-visible text
```

On successful publish of introspection (existing publish block), set:

```python
_LAST_HEAVY_INTRO_MONO = time.monotonic()
if redis is not None:
    await introspection_guard.mark_done(
        redis,
        prefix=settings.spark_introspection_key_prefix,
        trace_id=trace_id,
        status="ok",
        ttl_sec=int(settings.spark_introspection_done_ttl_sec),
    )
```

In `finally` after RPC attempt (success or failure), if `claimed`:

```python
await introspection_guard.release_inflight(redis, prefix=..., trace_id=trace_id)
```

Ensure `mark_done` is **not** written for hard failures if spec wants retry—spec says duplicate guard: on failure, **delete inflight** and omit `mark_done` so a later candidate can retry.

7. **Log completion** after successful text extraction + publish:

```python
logger.info(
    "spark_introspection_complete trace_id=%s correlation_id=%s",
    trace_id,
    str(env.correlation_id),
)
```

- [ ] **Step 1: Implement as above**, adjusting imports (`from app import introspection_guard` or relative `from . import introspection_guard`—match package layout).

- [ ] **Step 2: Manual smoke** (optional but recommended): run spark-introspector locally with `SPARK_INTROSPECTION_MAX_INFLIGHT=1` and observe logs when two candidates fire quickly.

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/app/worker.py
git commit -m "feat(spark-introspector): bound heavy introspection with semaphore and Redis idempotency"
```

---

### Task 6: Worker pressure unit tests (mocked bus / redis)

**Files:**
- Create: `services/orion-spark-introspector/tests/test_handle_candidate_pressure.py`

- [ ] **Step 1: Write tests** using `asyncio.run` for async portions (no `pytest-asyncio` required).

Patterns:

1. **Stale skip:** Build `BaseEnvelope` with `created_at=datetime.now(timezone.utc) - timedelta(seconds=999)`, payload minimal valid `SparkCandidatePayload` dict, `monkeypatch` `worker._emit_candidate_telemetry` to `AsyncMock`, `monkeypatch` `worker.OrionBusAsync` to fail if instantiated.

2. **Redis done skip:** `monkeypatch` `worker._redis_for_idempotency` to return mock redis whose `get` returns `b"ok"` when key matches done pattern; assert no `OrionBusAsync`.

3. **Semaphore busy + drop:** `settings` object patch `spark_introspection_drop_on_pressure=True`, `spark_introspection_acquire_timeout_sec=0`, pre-fill semaphore by acquiring once in test setup, call handler—expect degraded log path (use `caplog`).

Use `from app.worker import handle_candidate` and construct envelopes with `ServiceRef(name="hub", node="test", version="1", instance=None)` consistent with `BaseEnvelope` requirements.

- [ ] **Step 2: Run**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-spark-introspector
../../venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/tests/test_handle_candidate_pressure.py
git commit -m "test(spark-introspector): cover staleness and pressure skips for introspection"
```

---

### Task 7: Service runner verification (AGENTS.md)

**Files:** none

- [ ] **Step 1: Full service test path**

```bash
cd /mnt/scripts/Orion-Sapienform
./scripts/test_service.sh orion-spark-introspector
```

Expected: exit code 0; pytest collects new tests.

- [ ] **Step 2: Commit** (only if you changed anything; otherwise no commit).

---

## Self-review (plan vs spec)

| Spec requirement | Task covering it |
|------------------|------------------|
| `concurrent_handlers=True` on Hunter | Task 3 |
| Semaphore only around expensive cortex RPC | Task 5 (region contains codec, bus connect, rpc_request, publish introspection, bus close) |
| Env limits: max inflight, timeout, max age, drop on pressure, min interval | Tasks 1, 5 |
| Redis `SETNX` inflight + done + delete inflight | Tasks 2, 5 |
| Preserve lightweight telemetry/tissue when skipping heavy | Task 5 ordering: staleness / redis-done returns only after `_emit_candidate_telemetry` and `_update_tissue_from_candidate` for this handler pass. |
| Logs: start/complete/skipped/degraded | Task 5 |
| Unit tests: semaphore, duplicate trace, stale | Tasks 4–6 |
| `.env_example` parity | Task 1 |

**Plan fix note:** None — Task 5 header locks ordering vs spec.

**Placeholder scan:** None intentional.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-13-spark-introspection-phase1-implementation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

2. **Inline Execution** — run tasks sequentially in this session with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Which approach do you want?**
