from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.work_queue import RedisStreamWorkQueue

from app.queue_jobs import build_spark_introspection_job_envelope
from app.settings import Settings
from app.worker import SparkCandidatePayload


@pytest.mark.skipif(not os.environ.get("ORION_REDIS_STREAM_TEST_URL"), reason="ORION_REDIS_STREAM_TEST_URL not set")
def test_redis_stream_enqueue_and_ack_one_message() -> None:
    async def _run() -> None:
        url = os.environ["ORION_REDIS_STREAM_TEST_URL"]
        suffix = uuid4().hex[:10]
        stream = f"orion:queue:spark:introspection:test:{suffix}"
        group = f"test-group-{suffix}"
        consumer = f"test-consumer-{suffix}"

        settings = Settings()
        cand = SparkCandidatePayload(
            trace_id=f"int-test-{suffix}",
            prompt="p",
            response="r",
            spark_meta={"phi_before": {}, "as_of_ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()},
        )
        base = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub", node="n1"),
            correlation_id=uuid4(),
            payload=cand.model_dump(mode="json"),
        )
        src = ServiceRef(name="spark-introspector", node="test", version="0")
        job_env = build_spark_introspection_job_envelope(cand, base, settings, src)

        wq = RedisStreamWorkQueue(url, codec=OrionCodec())
        await wq.connect()
        try:
            await wq.ensure_group(stream, group, start_id="0", mkstream=True)
            mid = await wq.enqueue(stream, job_env)
            batch = await wq.read_group(stream, group, consumer, count=1, block_ms=2000)
            assert len(batch.messages) == 1
            assert batch.messages[0].message_id == mid
            await wq.ack(stream, group, mid)
        finally:
            try:
                await wq.client.delete(stream)
            except Exception:
                pass
            await wq.close()

    asyncio.run(_run())
