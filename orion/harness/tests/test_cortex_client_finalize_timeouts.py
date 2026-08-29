from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.harness.cortex_client import HarnessCortexClient
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.cortex.types import ExecutionStep


def _make_plan_request(verb_name: str) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name=verb_name,
            steps=[
                ExecutionStep(
                    verb_name=verb_name,
                    step_name=f"llm_{verb_name}",
                    order=0,
                    services=["LLMGatewayService"],
                )
            ],
        ),
        args=PlanExecutionArgs(request_id=str(uuid4())),
    )


@pytest.mark.asyncio
async def test_voice_finalize_uses_voice_timeout() -> None:
    bus = AsyncMock()
    decode_result = MagicMock(ok=True, envelope=MagicMock(payload={"result": {}}))
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})

    client = HarnessCortexClient(
        bus,
        request_channel="orion:cortex:exec:request",
        result_prefix="orion:exec:result",
        timeout_sec=180.0,
        voice_finalize_timeout_sec=300.0,
    )

    await client(_make_plan_request("orion_voice_finalize"))

    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == 300.0


@pytest.mark.asyncio
async def test_system_error_reply_raises_instead_of_returning_as_result() -> None:
    """Regression test for 2026-08-29: before this check, a system.error
    reply (no "result" key) fell through to `return payload`, handing the
    caller a raw {"error": ..., "details": ...} dict as if it were real
    plan-execution output -- silent data corruption with no exception
    anywhere in the harness finalize chain."""
    bus = AsyncMock()
    decode_result = MagicMock(
        ok=True,
        envelope=MagicMock(
            kind="system.error",
            payload={"error": "tts_synthesis_failed", "details": "CUDA out of memory"},
        ),
    )
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})

    client = HarnessCortexClient(
        bus,
        request_channel="orion:cortex:exec:request",
        result_prefix="orion:exec:result",
        timeout_sec=180.0,
    )

    with pytest.raises(RuntimeError, match="tts_synthesis_failed"):
        await client(_make_plan_request("orion_voice_finalize"))


@pytest.mark.asyncio
async def test_finalize_reflect_uses_reflect_timeout() -> None:
    bus = AsyncMock()
    decode_result = MagicMock(ok=True, envelope=MagicMock(payload={"result": {}}))
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})

    client = HarnessCortexClient(
        bus,
        request_channel="orion:cortex:exec:request",
        result_prefix="orion:exec:result",
        timeout_sec=180.0,
        voice_finalize_timeout_sec=300.0,
    )

    await client(_make_plan_request("harness_finalize_reflect"))

    assert bus.rpc_request.await_args.kwargs["timeout_sec"] == 180.0
