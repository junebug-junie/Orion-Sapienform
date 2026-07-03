from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker


def _candidate_payload(tid: str, *, spark_meta: dict | None = None) -> dict:
    return {
        "trace_id": tid,
        "prompt": "hello",
        "response": "hi",
        "spark_meta": spark_meta or {},
    }


def test_minimal_candidate_does_not_enqueue_when_rich_meta_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", True)

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        rh = AsyncMock()
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", rh)
        monkeypatch.setattr(worker, "_update_tissue_from_candidate", AsyncMock())
        monkeypatch.setattr(worker, "_emit_candidate_telemetry", AsyncMock())

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="llm-gateway", node="n1"),
            correlation_id=uuid4(),
            payload=_candidate_payload("thin-1", spark_meta={"latest_user_message": "Analyze the state shift."}),
        )
        await worker.handle_candidate(env)
        wq.enqueue.assert_not_awaited()
        rh.assert_not_awaited()

    asyncio.run(_run())


def test_rich_candidate_enqueues_when_rich_meta_required(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", True)

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", AsyncMock())
        monkeypatch.setattr(worker, "_update_tissue_from_candidate", AsyncMock())
        monkeypatch.setattr(worker, "_emit_candidate_telemetry", AsyncMock())

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub_http", node="n1"),
            correlation_id=uuid4(),
            payload=_candidate_payload(
                "rich-1",
                spark_meta={"turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.7}},
            ),
        )
        await worker.handle_candidate(env)
        wq.enqueue.assert_awaited_once()

    asyncio.run(_run())


def test_introspection_field_never_enqueues(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", False)

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(worker, "_emit_candidate_telemetry", AsyncMock())

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="spark-introspector", node="n1"),
            correlation_id=uuid4(),
            payload={
                **_candidate_payload("done-1", spark_meta={"phi_before": {"coherence": 0.5}}),
                "introspection": "already done",
            },
        )
        await worker.handle_candidate(env)
        wq.enqueue.assert_not_awaited()

    asyncio.run(_run())
