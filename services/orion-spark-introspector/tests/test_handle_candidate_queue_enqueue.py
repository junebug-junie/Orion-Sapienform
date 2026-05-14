from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker


def _rich_candidate_payload(tid: str) -> dict:
    return {
        "trace_id": tid,
        "prompt": "p",
        "response": "r",
        "spark_meta": {"phi_before": {"coherence": 0.5}},
    }


def test_queue_enabled_enqueues_and_skips_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        rh = AsyncMock()
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", rh)

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub", node="n1"),
            correlation_id=uuid4(),
            payload=_rich_candidate_payload("enq-1"),
        )
        await worker.handle_candidate(env)
        wq.enqueue.assert_awaited_once()
        rh.assert_not_awaited()

    asyncio.run(_run())


def test_queue_disabled_inline_calls_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", False)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", True)

        rh = AsyncMock(
            return_value={
                "ok": True,
                "status": "complete",
                "reason": "published",
                "trace_id": "inl-1",
                "correlation_id": None,
            }
        )
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", rh)

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub", node="n1"),
            correlation_id=uuid4(),
            payload=_rich_candidate_payload("inl-1"),
        )
        await worker.handle_candidate(env)
        rh.assert_awaited_once()

    asyncio.run(_run())


def test_enqueue_failure_inline_disabled_no_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)

        async def _boom_wq():
            raise RuntimeError("redis down")

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _boom_wq)
        rh = AsyncMock()
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", rh)

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub", node="n1"),
            correlation_id=uuid4(),
            payload=_rich_candidate_payload("fail-1"),
        )
        await worker.handle_candidate(env)
        rh.assert_not_awaited()

    asyncio.run(_run())


def test_enqueue_failure_inline_enabled_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", True)

        async def _boom_wq2():
            raise RuntimeError("redis down")

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _boom_wq2)
        rh = AsyncMock(
            return_value={
                "ok": True,
                "status": "complete",
                "reason": "published",
                "trace_id": "fb-1",
                "correlation_id": None,
            }
        )
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", rh)

        env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub", node="n1"),
            correlation_id=uuid4(),
            payload=_rich_candidate_payload("fb-1"),
        )
        await worker.handle_candidate(env)
        rh.assert_awaited_once()

    asyncio.run(_run())
