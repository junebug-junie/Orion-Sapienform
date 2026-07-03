"""
Cross-service contract: rich hub candidate may enqueue once; gateway introspect
feedback (thin meta) must not enqueue again when publish policy + rich-meta gate apply.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker as spark_worker


def test_feedback_loop_terminates_after_rich_seed_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulate: hub rich candidate enqueues once; thin gateway republication is blocked
    by SPARK_INTROSPECTION_REQUIRE_RICH_META (gateway publish skip is unit-tested separately).
    """

    async def _run() -> None:
        monkeypatch.setattr(spark_worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(spark_worker.settings, "spark_introspection_inline_heavy_enabled", False)
        monkeypatch.setattr(spark_worker.settings, "spark_introspection_require_rich_meta", True)

        enqueue_trace_ids: list[str] = []

        wq = MagicMock()

        async def _enqueue(stream, job_env, **kwargs):
            payload = job_env.payload if isinstance(job_env.payload, dict) else {}
            trace_id = str(payload.get("trace_id") or "")
            enqueue_trace_ids.append(trace_id)
            return f"msg-{len(enqueue_trace_ids)}"

        wq.enqueue = AsyncMock(side_effect=_enqueue)

        async def _get_wq():
            return wq

        monkeypatch.setattr(spark_worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(spark_worker, "run_heavy_spark_introspection", AsyncMock())
        monkeypatch.setattr(spark_worker, "_update_tissue_from_candidate", AsyncMock())
        monkeypatch.setattr(spark_worker, "_emit_candidate_telemetry", AsyncMock())

        seed_tid = str(uuid4())
        rich_env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub_http", node="athena"),
            correlation_id=seed_tid,
            payload={
                "trace_id": seed_tid,
                "prompt": "hello",
                "response": "hi",
                "spark_meta": {
                    "turn_change_appraisal": {
                        "turn_change_status": "ok",
                        "novelty_score": 0.72,
                    }
                },
            },
        )
        await spark_worker.handle_candidate(rich_env)

        feedback_tid = str(uuid4())
        thin_env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="llm-gateway", node="athena"),
            correlation_id=feedback_tid,
            payload={
                "trace_id": feedback_tid,
                "prompt": "Analyze the state shift.",
                "response": "loop introspection text",
                "spark_meta": {
                    "latest_user_message": "Analyze the state shift.",
                    "trace_verb": "introspect_spark",
                },
            },
        )
        await spark_worker.handle_candidate(thin_env)

        assert enqueue_trace_ids == [seed_tid]

    asyncio.run(_run())
