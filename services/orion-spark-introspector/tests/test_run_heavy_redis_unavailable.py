from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import introspection_guard as ig
from app import worker
from app.worker import SparkCandidatePayload


def test_try_claim_inflight_false_when_redis_none_and_idempotency_on() -> None:
    class S:
        spark_introspection_idempotency_enable = True
        spark_introspection_key_prefix = "spark:introspection"
        spark_introspection_inflight_ttl_sec = 60

    async def _run() -> None:
        ok = await ig.try_claim_inflight(None, settings=S(), trace_id="t1", owner="n:1")
        assert ok is False

    asyncio.run(_run())


def test_run_heavy_skipped_redis_unavailable_when_idempotency_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.settings, "spark_introspection_idempotency_enable", True)

    async def _run() -> None:
        with patch.object(worker.ig, "get_redis_client", new=AsyncMock(return_value=None)):
            cand = SparkCandidatePayload(
                trace_id="redis-skip-1",
                prompt="p",
                response="r",
                spark_meta={
                    "phi_before": {},
                    "as_of_ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            env = BaseEnvelope(
                kind="spark.candidate",
                source=ServiceRef(name="hub", node="n1"),
                correlation_id=uuid4(),
                payload=cand.model_dump(mode="json"),
            )
            out = await worker.run_heavy_spark_introspection(
                candidate=cand,
                source_env=env,
                correlation_id=None,
                bus=None,
                from_queue=False,
            )
            assert out["status"] == "skipped"
            assert out["reason"] == "redis_unavailable"
            assert out["correlation_id"] == str(env.correlation_id)

    asyncio.run(_run())
