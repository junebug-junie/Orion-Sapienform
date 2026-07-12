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
