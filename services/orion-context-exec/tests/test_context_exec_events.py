from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from uuid import UUID

import pytest

from app.events import ContextExecEventEmitter
from app.runner import ContextExecRunner
from orion.schemas.context_exec import ContextExecRequestV1

PARENT_CORR = "550e8400-e29b-41d4-a716-446655440042"


@pytest.mark.asyncio
async def test_runner_emits_lifecycle_events():
    bus = SimpleNamespace(publish=AsyncMock())
    emitter = ContextExecEventEmitter(bus, correlation_id=PARENT_CORR)  # type: ignore[arg-type]
    runner = ContextExecRunner(events=emitter)
    run = await runner.run(
        ContextExecRequestV1(
            text="Why did corr 1 fail open?",
            mode="trace_autopsy",
            correlation_id=PARENT_CORR,
        )
    )
    assert run.status == "ok"
    kinds = [call.args[1].kind for call in bus.publish.await_args_list]
    assert "context.exec.started.v1" in kinds
    assert "context.exec.verb_step.v1" in kinds
    assert "context.exec.finished.v1" in kinds
    for call in bus.publish.await_args_list:
        env = call.args[1]
        assert isinstance(env.correlation_id, UUID)
        assert str(env.correlation_id) == PARENT_CORR
        assert env.payload.get("run_id") != str(env.correlation_id)
        assert env.payload.get("correlation_id") == PARENT_CORR
