from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.memory_consolidation import MEMORY_TURN_PERSISTED_KIND

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_memory_outbox_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _source() -> ServiceRef:
    return ServiceRef(name="test-hub", version="0.0.1", node="local")


@pytest.mark.asyncio
async def test_chat_history_emits_memory_turn_persisted(monkeypatch):
    corr = str(uuid4())
    writes: list[tuple] = []
    published: list[tuple[str, BaseEnvelope]] = []

    def _fake_write_row(sql_model_cls, data: dict) -> bool:
        writes.append((sql_model_cls.__tablename__, data))
        return True

    bus = AsyncMock()

    async def _capture_publish(channel: str, env: BaseEnvelope) -> None:
        published.append((channel, env))

    bus.publish = _capture_publish
    monkeypatch.setattr(worker, "_write_row", _fake_write_row)
    monkeypatch.setattr(worker.settings, "sql_writer_emit_memory_turn_persisted", True)
    monkeypatch.setattr(worker.settings, "channel_memory_turn_persisted", "orion:memory:turn:persisted")

    env = BaseEnvelope(
        kind="chat.history",
        correlation_id=corr,
        source=_source(),
        payload={
            "prompt": "hello",
            "response": "hi there",
            "session_id": "sess-1",
            "spark_meta": {"conversation_phase": {"phase_change": "same_breath"}},
        },
    )

    await worker.handle_envelope(env, bus=bus)

    assert any(t == "chat_history_log" for t, _ in writes)
    assert len(published) == 1
    channel, out_env = published[0]
    assert channel == "orion:memory:turn:persisted"
    assert out_env.kind == MEMORY_TURN_PERSISTED_KIND
    assert str(out_env.correlation_id) == corr
    assert out_env.payload["correlation_id"] == corr
    assert out_env.payload["prompt"] == "hello"
    assert out_env.payload["response"] == "hi there"


@pytest.mark.asyncio
async def test_outbox_preserves_hub_trace_id_not_cortex_or_spark_ids(monkeypatch):
    """Hub sets envelope.correlation_id = trace_id; spark/cortex may carry other IDs in meta."""
    trace_id = str(uuid4())
    cortex_corr = str(uuid4())
    published: list[tuple[str, BaseEnvelope]] = []

    def _fake_write_row(sql_model_cls, data: dict) -> bool:
        assert data.get("correlation_id") == trace_id
        assert data.get("id") == trace_id
        return True

    bus = AsyncMock()

    async def _capture_publish(channel: str, env: BaseEnvelope) -> None:
        published.append((channel, env))

    bus.publish = _capture_publish
    monkeypatch.setattr(worker, "_write_row", _fake_write_row)
    monkeypatch.setattr(worker.settings, "sql_writer_emit_memory_turn_persisted", True)

    env = BaseEnvelope(
        kind="chat.history",
        correlation_id=trace_id,
        source=_source(),
        payload={
            "id": trace_id,
            "correlation_id": trace_id,
            "prompt": "user said hi",
            "response": "orion replied",
            "spark_meta": {
                "cortex_correlation_id": cortex_corr,
                "conversation_phase": {"phase_change": "same_breath"},
            },
        },
    )

    await worker.handle_envelope(env, bus=bus)

    assert len(published) == 1
    _, out_env = published[0]
    assert str(out_env.correlation_id) == trace_id
    assert out_env.payload["correlation_id"] == trace_id
    assert out_env.payload["spark_meta"].get("cortex_correlation_id") == cortex_corr
