from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app.queue_jobs import build_spark_introspection_job_envelope
from app.queue_worker import handle_spark_introspection_job
from app.settings import Settings
from app.worker import SparkCandidatePayload


def _job_env(settings: Settings) -> BaseEnvelope:
    cand = SparkCandidatePayload(
        trace_id="t-worker",
        prompt="p",
        response="r",
        spark_meta={"phi_before": {}, "as_of_ts": datetime.now(timezone.utc).isoformat()},
    )
    src = ServiceRef(name="spark-introspector", node="n", version="1")
    base = BaseEnvelope(
        kind="spark.candidate",
        source=ServiceRef(name="hub", node="n1"),
        correlation_id=uuid4(),
        payload=cand.model_dump(mode="json"),
    )
    return build_spark_introspection_job_envelope(cand, base, settings, src)


def test_valid_job_calls_run_heavy() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.return_value = {
                "ok": True,
                "status": "complete",
                "reason": "published",
                "trace_id": "t-worker",
                "correlation_id": None,
            }
            out = await handle_spark_introspection_job(env)
            assert out is None
            rh.assert_awaited_once()
            call_kw = rh.await_args.kwargs
            assert call_kw.get("from_queue") is True
            assert call_kw.get("job") is not None
            assert str(call_kw["job"].trace_id) == "t-worker"

    asyncio.run(_run())


def test_expired_job_skips_heavy() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        tr = dict(env.trace or {})
        w = dict(tr.get("work") or {})
        w["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        tr["work"] = w
        env = env.model_copy(update={"trace": tr})
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            out = await handle_spark_introspection_job(env)
            assert out is None
            rh.assert_not_awaited()

    asyncio.run(_run())


def test_redis_done_skipped_ack_none() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.return_value = {
                "ok": True,
                "status": "skipped",
                "reason": "redis_done",
                "trace_id": "t-worker",
                "correlation_id": None,
            }
            out = await handle_spark_introspection_job(env)
            assert out is None

    asyncio.run(_run())


def test_degraded_returns_none() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.return_value = {
                "ok": True,
                "status": "degraded",
                "reason": "timeout",
                "trace_id": "t-worker",
                "correlation_id": None,
            }
            out = await handle_spark_introspection_job(env)
            assert out is None

    asyncio.run(_run())


def test_retryable_connection_error_raises() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.side_effect = ConnectionError("boom")
            with pytest.raises(ConnectionError):
                await handle_spark_introspection_job(env)

    asyncio.run(_run())


def test_failed_status_raises_for_retry() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.return_value = {
                "ok": False,
                "status": "failed",
                "reason": "x",
                "trace_id": "t-worker",
                "correlation_id": None,
            }
            with pytest.raises(RuntimeError):
                await handle_spark_introspection_job(env)

    asyncio.run(_run())


def test_redis_unavailable_status_returns_none() -> None:
    async def _run() -> None:
        settings = Settings()
        env = _job_env(settings)
        with patch("app.queue_worker.run_heavy_spark_introspection", new_callable=AsyncMock) as rh:
            rh.return_value = {
                "ok": True,
                "status": "skipped",
                "reason": "redis_unavailable",
                "trace_id": "t-worker",
                "correlation_id": None,
            }
            out = await handle_spark_introspection_job(env)
            assert out is None

    asyncio.run(_run())


def test_malformed_job_returns_none() -> None:
    async def _run() -> None:
        env = BaseEnvelope(
            kind="spark.introspection.job.v1",
            source=ServiceRef(name="spark-introspector", node="n"),
            correlation_id=uuid4(),
            payload={"schema_version": "wrong"},
        )
        out = await handle_spark_introspection_job(env)
        assert out is None

    asyncio.run(_run())
