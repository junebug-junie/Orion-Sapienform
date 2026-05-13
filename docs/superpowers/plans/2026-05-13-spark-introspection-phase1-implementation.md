# Spark Introspection Phase 1 (Task Block A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bounded async handling and pressure relief inside `orion-spark-introspector` so heavy `introspect_spark` cortex RPC work is semaphored, idempotent across retries/replicas (Redis), skippable under staleness or saturation, and observable via structured log lines—without changing exec channels or orch routing.

**Architecture:** Keep `Hunter` as the bus consumer but enable `concurrent_handlers=True` so traces/signals/upserts are not blocked behind a slow candidate. Use **two** narrow locks: `_CANDIDATE_CACHE_LOCK` for `_CANDIDATE_*` / `_TURN_EFFECT_*` / telemetry emission that mutates those structures, and `_TISSUE_LOCK` for `TISSUE` mutation and snapshot generation—**never** one giant lock around the whole handler, and do not serialize unrelated handlers on each other more than necessary. **Redis `done`** short-circuits **before** `_emit_candidate_telemetry` and `_update_tissue_from_candidate` so duplicate deliveries do not double-imprint tissue or duplicate durable telemetry rows; only a structured skip log (see Task 5 ordering). Wrap **only** the `OrionBusAsync` connect + `rpc_request` + teardown (and publish of introspection result) in an `asyncio.Semaphore`; acquire **only** via `_try_acquire_intro_sem()` so `DROP_ON_PRESSURE=true` with `ACQUIRE_TIMEOUT_SEC=0` never falls through to a blocking `await sem.acquire()` when the slot is busy. Use Redis keys on the same URL as `ORION_BUS_URL` for inflight `SETNX` and done markers keyed by `trace_id`. Emit four log patterns: `spark_introspection_start`, `spark_introspection_complete`, `spark_introspection_skipped`, `spark_introspection_degraded`. Emergency operator switch: `SPARK_INTROSPECTION_ENABLE_HEAVY=false` runs telemetry + tissue then exits before heavy RPC.

**Tech Stack:** Python 3.12, FastAPI/uvicorn (unchanged), `redis==5.0.4` (`redis.asyncio`), Pydantic settings, pytest, existing `OrionBusAsync` / `BaseEnvelope` types.

**Spec source:** `docs/superpowers/specs/2026-05-13-spark-introspection-lane-isolation-design.md` (Phase 1 / Task Block A only).

---

## Pre-execution patches (do not improvise)

Apply these during implementation; they supersede earlier wording in this file if anything conflicts.

1. **Semaphore:** Implement `_try_acquire_intro_sem() -> bool` exactly as in Task 5. **Never** use a bare `await sem.acquire()` when `SPARK_INTROSPECTION_DROP_ON_PRESSURE=true` and `SPARK_INTROSPECTION_ACQUIRE_TIMEOUT_SEC<=0`; immediate drop must use the `sem.locked()` fast path (returns `False` without blocking).
2. **Redis `done`:** If `is_done`, **return after structured log only**—no `_emit_candidate_telemetry`, no `_update_tissue_from_candidate`, no heavy RPC (avoids duplicate tissue imprint and duplicate durable telemetry).
3. **Locks:** Two locks only (`_CANDIDATE_CACHE_LOCK`, `_TISSUE_LOCK`); scope them to mutations described in Task 5. `handle_signal` / trace paths may need the same `_TISSUE_LOCK` if they mutate `TISSUE` concurrently with candidates—audit in Task 5 and extend lock scope minimally if required.
4. **Tests:** Reset `worker._INTRO_SEM`, `worker._REDIS_CLIENT`, `worker._LAST_HEAVY_INTRO_MONO` (and any other module singletons touched) between tests that depend on clean state.
5. **Staleness:** Normalize `env.created_at` to UTC-aware before subtracting from `datetime.now(timezone.utc)`.
6. **Kill switch:** Add `SPARK_INTROSPECTION_ENABLE_HEAVY` (default `true`). When `false`, still run validation, cache gates, `_emit_candidate_telemetry`, and `_update_tissue_from_candidate` (Task 5 steps 1–5), then log `spark_introspection_degraded reason=heavy_disabled` and `return` before heavy RPC; document in `.env_example`.

---

## File map

| File | Responsibility |
|------|----------------|
| Modify: `services/orion-spark-introspector/app/settings.py` | New env-backed fields including `SPARK_INTROSPECTION_ENABLE_HEAVY`, semaphore, timeouts, staleness, drop-on-pressure, min interval, Redis TTLs, idempotency toggle, optional Redis URL override. |
| Modify: `services/orion-spark-introspector/app/main.py` | Pass `concurrent_handlers=True` into `Hunter`. |
| Create: `services/orion-spark-introspector/app/introspection_guard.py` | Thin async Redis helper: `try_claim_inflight`, `mark_done`, `release_inflight`, `is_done` using key prefix from settings. |
| Modify: `services/orion-spark-introspector/app/worker.py` | `_CANDIDATE_CACHE_LOCK`, `_TISSUE_LOCK`, `_try_acquire_intro_sem()`, redis_done early exit, normalized `created_at`, `enable_heavy` gate, guard + staleness + min-interval + RPC timeout + structured logs. |
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
    spark_introspection_enable_heavy: bool = Field(True, alias="SPARK_INTROSPECTION_ENABLE_HEAVY")
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

- `SPARK_INTROSPECTION_ENABLE_HEAVY`: when `false`, still run validation, cache gates, `_emit_candidate_telemetry`, and `_update_tissue_from_candidate` (per handle order), then **return** before cortex RPC (emergency kill switch).
- `SPARK_INTROSPECTION_ACQUIRE_TIMEOUT_SEC` + `SPARK_INTROSPECTION_DROP_ON_PRESSURE`: semaphore acquire is **only** via `_try_acquire_intro_sem()` in Task 5. When `DROP_ON_PRESSURE=true` and this value is `<=0`, a busy semaphore returns immediately (no blocking wait). When `DROP_ON_PRESSURE=false` and this value is `<=0`, the helper falls through to a blocking `await sem.acquire()` (wait for a slot); optionally document that operators may set an explicit positive timeout instead.
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

### Task 5: Worker integration — locks, ordering, semaphore helper, RPC timeout, logs, Redis lifecycle

**Files:**
- Modify: `services/orion-spark-introspector/app/worker.py`

#### Canonical `handle_candidate` order

Implement **exactly** this sequence (adjust only if a merge conflict forces mechanical relocation—preserve semantics):

1. **Validate** `SparkCandidatePayload`; derive `trace_id`; `_prune_candidate_caches()` (any pruning that mutates `_CANDIDATE_*` must run under `_CANDIDATE_CACHE_LOCK`).
2. **Quality / dedupe gates** that mutate `_CANDIDATE_QUALITY`, `_CANDIDATE_LAST_SEEN_TS`, `_CANDIDATE_TELEM_EMITTED`, or early-return branches that touch those structures: keep existing logic but wrap those mutations in `async with _CANDIDATE_CACHE_LOCK:`. If the existing quality gate emits partial telemetry inside that section, keep it there unchanged.
3. **Redis `done` (duplicate completion):** `redis = await _redis_for_idempotency()` (reuse this handle for later SETNX). If `redis` and `await introspection_guard.is_done(...)`:
   - `logger.info("spark_introspection_skipped trace_id=%s correlation_id=%s reason=redis_done", ...)`
   - **`return`** without `_emit_candidate_telemetry`, without `_update_tissue_from_candidate`, without heavy RPC (no duplicate tissue imprint, no duplicate durable telemetry rows).
4. **`async with _CANDIDATE_CACHE_LOCK:`:** `_emit_candidate_telemetry(...)` (existing body).
5. **`async with _TISSUE_LOCK:`:** if `not candidate.introspection`, `_update_tissue_from_candidate(...)` (existing).
6. If `candidate.introspection`: `return` (existing).
7. **`SPARK_INTROSPECTION_ENABLE_HEAVY`:** if `not settings.spark_introspection_enable_heavy`: log `spark_introspection_degraded ... reason=heavy_disabled`; `return`.
8. **Staleness** (telemetry + tissue already ran; stale still imprinted once): normalize time, then compare:

```python
from datetime import datetime, timezone

created_at = env.created_at
if created_at.tzinfo is None:
    created_at = created_at.replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
age_sec = (now - created_at).total_seconds()
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

9. **Min interval** (global throttle between successful heavy completions):

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

10. **SETNX inflight** — compute `owner = f"{settings.node_name}:{_PRODUCER_BOOT_ID}"`. If `redis` is `None`, set `claimed = True` (semaphore only; no inflight key). Else `claimed = await introspection_guard.try_claim_inflight(...)`; if not `claimed`, log `spark_introspection_skipped ... reason=inflight_not_claimed` and `return`.
11. **`spark_introspection_start`** log (`trace_id`, `correlation_id`).
12. **Semaphore:** `sem = _intro_sem()`; `acquired_slot = False`; try `acquired_slot = await _try_acquire_intro_sem()` except `asyncio.TimeoutError` → `acquired_slot = False`. If not `acquired_slot`, then if `redis is not None` and `claimed`, `release_inflight(...)`; log `spark_introspection_degraded ... reason=semaphore_busy`; `return`.
13. **Heavy RPC** in `try` / `finally: if acquired_slot: sem.release()` — paste the existing `worker.py` implementation from `OrionCodec()` through `await bus.close()` (including `asyncio.wait_for` around `rpc_request` with `spark_introspection_timeout_sec`). Do not leave placeholder `pass`.
14. **Success path:** after publish + WS broadcast succeed, `_LAST_HEAVY_INTRO_MONO = time.monotonic()`; if `redis is not None`, `await introspection_guard.mark_done(...)`; log `spark_introspection_complete`.
15. **Inflight cleanup:** Ensure `release_inflight` is called **exactly once** per `claimed` SETNX success: on semaphore acquire failure call it before `return`; on RPC failure call it in a `finally` (or equivalent) after the heavy `try`; on RPC success call it **after** `mark_done` (order: `mark_done` then `release_inflight`, matching the original spec’s “done + delete inflight” semantics). Do not double-release on paths that already released before an early `return`.

**Cross-handler note:** If `handle_trace` / `handle_signal` mutate `TISSUE` or `_ACTIVE_SIGNALS` concurrently with candidates, extend locking minimally (e.g. same `_TISSUE_LOCK` around `TISSUE` writes in those handlers) so `concurrent_handlers=True` does not interleave unsafe mutations—verify during implementation.

#### Module-level state and helpers

```python
import asyncio
import time
from datetime import datetime, timezone

from redis.asyncio import Redis

_INTRO_SEM: asyncio.Semaphore | None = None
_CANDIDATE_CACHE_LOCK = asyncio.Lock()
_TISSUE_LOCK = asyncio.Lock()
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


async def _try_acquire_intro_sem() -> bool:
    """
    Acquire intro semaphore without accidentally blocking when DROP_ON_PRESSURE
    and ACQUIRE_TIMEOUT_SEC<=0 (immediate drop if busy).
    """
    sem = _intro_sem()
    acquire_timeout = float(settings.spark_introspection_acquire_timeout_sec)

    if settings.spark_introspection_drop_on_pressure and acquire_timeout <= 0:
        if sem.locked():
            return False
        await sem.acquire()
        return True

    if acquire_timeout > 0:
        await asyncio.wait_for(sem.acquire(), timeout=acquire_timeout)
        return True

    await sem.acquire()
    return True
```

`asyncio.Semaphore.locked()` is available in Python 3.10+ (this service targets 3.12 per `Dockerfile`).

Call `Redis.from_url` once; optional `close()` on app shutdown in `main.py` lifespan (follow-up if skipped).

- [ ] **Step 1: Implement as above**, using `from app import introspection_guard` or `from . import introspection_guard` per package layout.

- [ ] **Step 2: Manual smoke** (optional): two rapid candidates with `SPARK_INTROSPECTION_MAX_INFLIGHT=1`, `DROP_ON_PRESSURE=true`, `ACQUIRE_TIMEOUT_SEC=0`; confirm second does **not** block waiting on semaphore.

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/app/worker.py
git commit -m "feat(spark-introspector): bound heavy introspection with semaphore and Redis idempotency"
```

---

### Task 6: Worker pressure unit tests (mocked bus / redis)

**Files:**
- Create: `services/orion-spark-introspector/tests/test_handle_candidate_pressure.py`
- Create: `services/orion-spark-introspector/tests/conftest.py`

- [ ] **Step 1: Write tests** using `asyncio.run` for async portions (no `pytest-asyncio` required).

Add `services/orion-spark-introspector/tests/conftest.py` with an **autouse** fixture that resets module singletons between tests:

```python
import pytest
from app import worker as spark_worker


@pytest.fixture(autouse=True)
def _reset_spark_introspector_worker_singletons():
    spark_worker._INTRO_SEM = None
    spark_worker._REDIS_CLIENT = None
    spark_worker._LAST_HEAVY_INTRO_MONO = 0.0
    yield
    spark_worker._INTRO_SEM = None
    spark_worker._REDIS_CLIENT = None
    spark_worker._LAST_HEAVY_INTRO_MONO = 0.0
```

Patterns:

1. **Stale skip:** Build `BaseEnvelope` with old `created_at` (include one test with **naive** `datetime` to prove UTC normalization). Monkeypatch `worker.OrionBusAsync` to raise if constructed. Assert `_emit_candidate_telemetry` ran (AsyncMock) when the path should still emit before staleness check.

2. **Redis done skip:** Monkeypatch `worker._redis_for_idempotency` to return a mock whose `get` indicates done for the done-key; assert **`_emit_candidate_telemetry` and `_update_tissue_from_candidate` are not called** (wrap with `AsyncMock` / spy). Assert no `OrionBusAsync`.

3. **Semaphore busy + drop:** Patch settings `spark_introspection_drop_on_pressure=True`, `spark_introspection_acquire_timeout_sec=0`. Pre-hold the semaphore: `sem = worker._intro_sem(); await sem.acquire()` inside `asyncio.run`, then invoke `handle_candidate` for a second message—expect `spark_introspection_degraded` / `semaphore_busy` in logs (`caplog`). Rely on autouse fixture to reset `_INTRO_SEM` between tests.

4. **`enable_heavy`:** With `spark_introspection_enable_heavy=False`, assert telemetry/tissue mocks were invoked but `OrionBusAsync` was not.

Use `from app.worker import handle_candidate` and construct envelopes with `ServiceRef(name="hub", node="test", version="1", instance=None)` consistent with `BaseEnvelope` requirements.

- [ ] **Step 2: Run**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-spark-introspector
../../venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add services/orion-spark-introspector/tests/test_handle_candidate_pressure.py services/orion-spark-introspector/tests/conftest.py
git commit -m "test(spark-introspector): cover introspection pressure, redis_done, and kill switch"
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
| Preserve lightweight telemetry/tissue when skipping heavy | **Stale:** after `_emit_candidate_telemetry` + `_update_tissue_from_candidate` (steps 4–5). **`redis_done`:** step 3 returns **before** steps 4–5 (no duplicate telemetry/tissue). **`enable_heavy=false`:** steps 4–5 run, then return before heavy RPC. |
| Logs: start / complete / skipped / degraded | Task 5 steps 11, 14–15 |
| Unit tests: semaphore, duplicate trace, stale, kill switch | Tasks 4–6 |
| Emergency kill switch | Task 1 (`SPARK_INTROSPECTION_ENABLE_HEAVY`), Task 5 step 7 |
| Semaphore immediate drop semantics | Pre-execution patch §1; Task 5 `_try_acquire_intro_sem` |
| Two locks + cross-handler audit | Pre-execution patch §3; Task 5 |
| Test isolation for memoized semaphore | Task 6 `conftest.py` autouse fixture |
| `created_at` timezone-safe staleness | Task 5 step 8 |
| `.env_example` parity | Task 1 |

**Placeholder scan:** None intentional.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-13-spark-introspection-phase1-implementation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

2. **Inline Execution** — run tasks sequentially in this session with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Which approach do you want?**
