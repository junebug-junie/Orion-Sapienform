from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.bus_listener import _handle_request
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.context_exec import ContextExecRunV1


def _request_envelope(*, causality_chain=None) -> BaseEnvelope:
    kwargs: dict = {
        "kind": "context.exec.request.v1",
        "source": ServiceRef(name="cortex-exec", version="0.2.0"),
        "correlation_id": uuid4(),
        "reply_to": "orion:exec:result:ContextExecService:test-corr",
        "payload": {"text": "trace autopsy", "mode": "trace_autopsy"},
    }
    if causality_chain is not None:
        kwargs["causality_chain"] = causality_chain
    return BaseEnvelope(**kwargs)


def _bus_for_envelope(env: BaseEnvelope) -> SimpleNamespace:
    return SimpleNamespace(
        codec=SimpleNamespace(
            decode=lambda _raw: SimpleNamespace(ok=True, envelope=env, error=None)
        ),
        publish=AsyncMock(),
    )


def _ok_run() -> ContextExecRunV1:
    return ContextExecRunV1(
        run_id="ctxrun_test",
        status="ok",
        mode="trace_autopsy",
        text="trace autopsy",
        final_text="trace summary",
        runtime_debug={"engine": "context_exec"},
    )


@pytest.mark.asyncio
async def test_bus_success_reply_uses_empty_causality_list():
    env = _request_envelope(causality_chain=[])
    bus = _bus_for_envelope(env)
    runner = SimpleNamespace(run=AsyncMock(return_value=_ok_run()))

    await _handle_request(bus, {"data": b"x"}, runner)  # type: ignore[arg-type]

    bus.publish.assert_awaited_once()
    reply_channel, reply_env = bus.publish.await_args.args
    assert reply_channel == env.reply_to
    assert reply_env.kind == "context.exec.result.v1"
    assert reply_env.causality_chain == []


@pytest.mark.asyncio
async def test_bus_reply_normalizes_missing_causality_to_empty_list():
    env = _request_envelope()
    assert env.causality_chain == []
    bus = _bus_for_envelope(env)
    runner = SimpleNamespace(run=AsyncMock(return_value=_ok_run()))

    await _handle_request(bus, {"data": b"x"}, runner)  # type: ignore[arg-type]

    _, reply_env = bus.publish.await_args.args
    assert reply_env.causality_chain == []


@pytest.mark.asyncio
async def test_bus_error_reply_uses_empty_causality_list():
    env = _request_envelope(causality_chain=[])
    bus = _bus_for_envelope(env)
    runner = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("runner blew up")))

    await _handle_request(bus, {"data": b"x"}, runner)  # type: ignore[arg-type]

    bus.publish.assert_awaited_once()
    _, reply_env = bus.publish.await_args.args
    assert reply_env.kind == "context.exec.result.v1"
    assert reply_env.causality_chain == []
    assert "runner blew up" in str(reply_env.payload.get("structured", {}).get("context_exec_error", ""))
