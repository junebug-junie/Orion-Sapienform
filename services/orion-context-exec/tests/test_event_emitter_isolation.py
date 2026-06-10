from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.runner import ContextExecRunner
from orion.schemas.context_exec import ContextExecRequestV1

CORR_A = "550e8400-e29b-41d4-a716-446655440001"
CORR_B = "550e8400-e29b-41d4-a716-446655440002"


@pytest.mark.asyncio
async def test_each_run_gets_fresh_emitter_correlation():
    bus = SimpleNamespace(publish=AsyncMock())
    runner = ContextExecRunner(bus=bus)  # type: ignore[arg-type]

    await runner.run(
        ContextExecRequestV1(
            text="Why did corr 1 fail open?",
            mode="trace_autopsy",
            correlation_id=CORR_A,
        )
    )
    await runner.run(
        ContextExecRequestV1(
            text="Why did corr 2 fail open?",
            mode="trace_autopsy",
            correlation_id=CORR_B,
        )
    )

    corrs = [str(call.args[1].correlation_id) for call in bus.publish.await_args_list]
    assert CORR_A in corrs
    assert CORR_B in corrs
    for call in bus.publish.await_args_list:
        env = call.args[1]
        assert isinstance(env.correlation_id, UUID)
        assert env.payload.get("correlation_id") == str(env.correlation_id)
