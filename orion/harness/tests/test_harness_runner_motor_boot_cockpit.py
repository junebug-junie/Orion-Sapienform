from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


@pytest.mark.asyncio
async def test_harness_runner_publishes_motor_boot_step_with_exact_prompt() -> None:
    thought = make_thought(imperative="Check logs first.", tone="direct")
    captured_prompt: dict[str, str] = {}
    captured_steps: list[dict[str, Any]] = []

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured_prompt["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    async def capture_step(
        bus: Any,
        *,
        correlation_id: str,
        step_index: int,
        step: dict[str, Any],
        channel: str,
        source_name: str = "orion-harness-governor",
    ) -> None:
        captured_steps.append(
            {
                "correlation_id": correlation_id,
                "step_index": step_index,
                "step": step,
                "channel": channel,
                "source_name": source_name,
            }
        )

    request = HarnessRunRequestV1(
        correlation_id="c-motor-boot",
        thought_event=thought,
        user_message="what broke?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)

    with (
        patch("orion.harness.runner.publish_harness_run_step", capture_step),
        patch("orion.harness.runner.read_last_tool_fetch", AsyncMock(return_value=None)),
    ):
        await runner.run(request)

    boot_steps = [
        s
        for s in captured_steps
        if isinstance(s.get("step"), dict)
        and s["step"].get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER
    ]
    assert len(boot_steps) == 1
    boot = boot_steps[0]
    assert boot["step_index"] == -1
    assert boot["correlation_id"] == "c-motor-boot"
    assert boot["step"]["prompt"] == captured_prompt["prompt"]
    assert boot["step"]["prompt_char_len"] == len(captured_prompt["prompt"])
    assert boot["step"]["prompt_char_len"] > 0
