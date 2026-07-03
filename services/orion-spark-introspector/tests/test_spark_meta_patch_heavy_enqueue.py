from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker


def _reset_candidate_caches() -> None:
    worker._CANDIDATE_QUALITY.clear()
    worker._CANDIDATE_LAST_SEEN_TS.clear()
    worker._CANDIDATE_TELEM_EMITTED.clear()
    worker._CANDIDATE_SPARK_META.clear()
    worker._CANDIDATE_STIMULUS.clear()
    worker._CANDIDATE_HEAVY_ENQUEUED.clear()


def _rich_patch_payload(corr: str) -> dict:
    return {
        "correlation_id": corr,
        "spark_meta": {
            "turn_change_appraisal": {
                "turn_change_status": "ok",
                "novelty_score": 0.82,
            },
            "novelty": 0.82,
        },
    }


def test_patch_enqueues_heavy_after_thin_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        _reset_candidate_caches()
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", True)

        corr = str(uuid4())
        thin_env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="llm-gateway", node="n1"),
            correlation_id=corr,
            payload={
                "trace_id": corr,
                "prompt": "user said hello",
                "response": "assistant replied",
                "spark_meta": {"latest_user_message": "user said hello"},
            },
        )

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(worker, "run_heavy_spark_introspection", AsyncMock())
        monkeypatch.setattr(worker, "_update_tissue_from_candidate", AsyncMock())
        monkeypatch.setattr(worker, "_emit_candidate_telemetry", AsyncMock())

        await worker.handle_candidate(thin_env)
        wq.enqueue.assert_not_awaited()

        patch_env = BaseEnvelope(
            kind="chat.history.spark_meta.patch.v1",
            correlation_id=corr,
            source=ServiceRef(name="memory-consolidation", node="athena"),
            payload=_rich_patch_payload(corr),
        )
        monkeypatch.setattr(worker, "_broadcast_tissue_update", AsyncMock())
        await worker.handle_spark_meta_patch(patch_env)

        wq.enqueue.assert_awaited_once()
        job_env = wq.enqueue.await_args.args[1]
        payload = job_env.payload if isinstance(job_env.payload, dict) else {}
        assert payload.get("trace_id") == corr
        assert payload.get("prompt") == "user said hello"

    asyncio.run(_run())


def test_duplicate_patch_does_not_double_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        _reset_candidate_caches()
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_inline_heavy_enabled", False)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", True)

        corr = str(uuid4())
        thin_env = BaseEnvelope(
            kind="spark.candidate",
            source=ServiceRef(name="hub_http", node="n1"),
            correlation_id=corr,
            payload={
                "trace_id": corr,
                "prompt": "q",
                "response": "a",
                "spark_meta": {},
            },
        )

        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(worker, "_update_tissue_from_candidate", AsyncMock())
        monkeypatch.setattr(worker, "_emit_candidate_telemetry", AsyncMock())
        monkeypatch.setattr(worker, "_broadcast_tissue_update", AsyncMock())

        await worker.handle_candidate(thin_env)

        patch_env = BaseEnvelope(
            kind="chat.history.spark_meta.patch.v1",
            correlation_id=corr,
            source=ServiceRef(name="memory-consolidation", node="athena"),
            payload=_rich_patch_payload(corr),
        )
        await worker.handle_spark_meta_patch(patch_env)
        await worker.handle_spark_meta_patch(patch_env)

        assert wq.enqueue.await_count == 1

    asyncio.run(_run())


def test_patch_without_stimulus_does_not_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        _reset_candidate_caches()
        monkeypatch.setattr(worker.settings, "spark_introspection_queue_enabled", True)
        monkeypatch.setattr(worker.settings, "spark_introspection_require_rich_meta", True)

        corr = str(uuid4())
        wq = MagicMock()
        wq.enqueue = AsyncMock(return_value="99-0")

        async def _get_wq():
            return wq

        monkeypatch.setattr(worker, "get_stream_enqueue_wq", _get_wq)
        monkeypatch.setattr(worker, "_broadcast_tissue_update", AsyncMock())

        patch_env = BaseEnvelope(
            kind="chat.history.spark_meta.patch.v1",
            correlation_id=corr,
            source=ServiceRef(name="memory-consolidation", node="athena"),
            payload=_rich_patch_payload(corr),
        )
        await worker.handle_spark_meta_patch(patch_env)
        wq.enqueue.assert_not_awaited()

    asyncio.run(_run())
