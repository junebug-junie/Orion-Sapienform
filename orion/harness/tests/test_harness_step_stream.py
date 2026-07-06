from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


@pytest.mark.asyncio
async def test_harness_runner_publishes_run_step_events() -> None:
    published: list[dict[str, Any]] = []

    async def _mock_fcc_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "step",
            "step": {"type": "assistant", "raw": {"type": "assistant"}},
        }
        yield {"type": "final", "llm_response": "draft", "metadata": {"exit_code": 0}}

    bus = AsyncMock()

    async def _capture_publish(channel: str, envelope: Any) -> None:
        if channel == "orion:harness:run:step":
            published.append(envelope.payload)

    bus.publish = AsyncMock(side_effect=_capture_publish)

    request = HarnessRunRequestV1(
        correlation_id="c-step-stream",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(
        bus,
        step_channel="orion:harness:run:step",
        fcc_runner=_mock_fcc_runner,
    )
    result = await runner.run(request)

    assert result.draft_text == "draft"
    assert len(published) == 1
    assert published[0]["correlation_id"] == "c-step-stream"
    assert published[0]["step_index"] == 0
