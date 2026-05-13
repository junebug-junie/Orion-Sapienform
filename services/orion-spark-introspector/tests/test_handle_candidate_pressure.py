from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker as spark_worker
from app.worker import handle_candidate


def _svc() -> ServiceRef:
    return ServiceRef(name="hub", node="test", version="1", instance=None)


def _rich_payload(trace_id: str = "trace-stale-1") -> dict:
    return {
        "trace_id": trace_id,
        "prompt": "hello",
        "response": "world",
        "spark_meta": {
            "phi_after": {"coherence": 0.5, "novelty": 0.1},
            "mode": "brain",
            "trace_verb": "chat",
        },
    }


def test_stale_skip_after_telemetry_emits(monkeypatch, caplog):
    caplog.set_level("INFO")
    emit = AsyncMock()
    monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", emit)
    monkeypatch.setattr(
        spark_worker,
        "OrionBusAsync",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OrionBusAsync should not be used")),
    )

    spark_worker.settings.spark_introspection_queue_max_age_sec = 5.0
    old = datetime.now(timezone.utc) - timedelta(seconds=30)
    env = BaseEnvelope(
        kind="spark.candidate",
        source=_svc(),
        correlation_id=uuid4(),
        created_at=old,
        payload=_rich_payload(),
    )

    asyncio.run(handle_candidate(env))
    emit.assert_awaited_once()


def test_stale_skip_naive_created_at_normalized(monkeypatch, caplog):
    caplog.set_level("INFO")
    emit = AsyncMock()
    monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", emit)
    monkeypatch.setattr(
        spark_worker,
        "OrionBusAsync",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OrionBusAsync should not be used")),
    )

    spark_worker.settings.spark_introspection_queue_max_age_sec = 5.0
    naive_old = datetime(2000, 1, 1, 12, 0, 0)
    env = BaseEnvelope(
        kind="spark.candidate",
        source=_svc(),
        correlation_id=uuid4(),
        created_at=naive_old,
        payload=_rich_payload(trace_id="trace-naive-stale"),
    )

    asyncio.run(handle_candidate(env))
    emit.assert_awaited_once()


def test_redis_done_skips_emit_tissue_and_bus(monkeypatch):
    emit = AsyncMock()
    tissue = AsyncMock()
    monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", emit)
    monkeypatch.setattr(spark_worker, "_update_tissue_from_candidate", tissue)
    monkeypatch.setattr(
        spark_worker,
        "OrionBusAsync",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OrionBusAsync should not be used")),
    )

    dummy_redis = MagicMock()
    monkeypatch.setattr(spark_worker, "_redis_for_idempotency", AsyncMock(return_value=dummy_redis))
    with patch.object(spark_worker.introspection_guard, "is_done", new=AsyncMock(return_value=True)):
        env = BaseEnvelope(
            kind="spark.candidate",
            source=_svc(),
            correlation_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            payload=_rich_payload(trace_id="trace-redis-done"),
        )
        asyncio.run(handle_candidate(env))

    emit.assert_not_awaited()
    tissue.assert_not_awaited()


def test_semaphore_busy_drop_logs_degraded(caplog, monkeypatch):
    caplog.set_level("INFO")
    monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", AsyncMock())
    monkeypatch.setattr(spark_worker, "_update_tissue_from_candidate", AsyncMock())
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_drop_on_pressure", True)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_acquire_timeout_sec", 0.0)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_max_inflight", 1)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_enable_heavy", True)

    async def _go() -> None:
        sem = spark_worker._intro_sem()
        await sem.acquire()
        try:
            env = BaseEnvelope(
                kind="spark.candidate",
                source=_svc(),
                correlation_id=uuid4(),
                created_at=datetime.now(timezone.utc),
                payload=_rich_payload(trace_id="trace-sem-busy"),
            )
            await handle_candidate(env)
        finally:
            sem.release()

    asyncio.run(_go())
    assert any("spark_introspection_degraded" in r.message and "semaphore_busy" in r.message for r in caplog.records)


def test_try_acquire_intro_sem_concurrent_drop_only_one_slot(monkeypatch):
    """DROP_ON_PRESSURE + zero timeout: two waiters must not both acquire (non-blocking semantics)."""
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_drop_on_pressure", True)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_acquire_timeout_sec", 0.0)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_max_inflight", 1)

    async def _go() -> None:
        spark_worker._INTRO_SEM = None

        async def one() -> bool:
            return await spark_worker._try_acquire_intro_sem()

        a, b = await asyncio.gather(one(), one())
        assert sum(1 for x in (a, b) if x) == 1
        sem = spark_worker._intro_sem()
        if a or b:
            sem.release()

    asyncio.run(_go())


def test_enable_heavy_false_skips_bus(monkeypatch):
    emit = AsyncMock()
    tissue = AsyncMock()
    monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", emit)
    monkeypatch.setattr(spark_worker, "_update_tissue_from_candidate", tissue)
    monkeypatch.setattr(spark_worker.settings, "spark_introspection_enable_heavy", False)
    monkeypatch.setattr(
        spark_worker,
        "OrionBusAsync",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OrionBusAsync should not be used")),
    )

    env = BaseEnvelope(
        kind="spark.candidate",
        source=_svc(),
        correlation_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        payload=_rich_payload(trace_id="trace-no-heavy"),
    )
    asyncio.run(handle_candidate(env))

    emit.assert_awaited()
    tissue.assert_awaited()
