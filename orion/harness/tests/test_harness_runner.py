from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.runner import HarnessMotorResult, HarnessRunner, build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessRunRequestV1


async def _mock_fcc_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "step": {
            "type": "assistant",
            "raw": {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ]
                },
            },
        },
    }
    yield {"type": "final", "llm_response": "internal draft answer", "metadata": {"exit_code": 0}}


@pytest.mark.asyncio
async def test_harness_runner_surfaces_fcc_error_code() -> None:
    async def _error_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "error",
            "error": "fcc turn timed out after 120.0s",
            "error_code": "fcc_timeout",
        }

    request = HarnessRunRequestV1(
        correlation_id="c-timeout",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_error_runner)
    result = await runner.run(request)

    assert result.draft_text == ""
    assert result.compliance_verdict == "failed"
    assert result.grounding_status == "fcc_timeout"


@pytest.mark.asyncio
async def test_harness_runner_collects_grammar_receipts_and_draft() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-runner",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner)

    result = await runner.run(request)

    assert result.draft_text == "internal draft answer"
    assert result.step_count == 1
    assert len(result.grammar_receipts) == 1
    assert result.grammar_receipts[0].tool_name == "Read"
    assert result.draft_molecule is not None
    assert result.draft_molecule.draft_text == "internal draft answer"
    assert result.compliance_verdict == "completed"


@pytest.mark.asyncio
async def test_harness_runner_publishes_lifecycle_grammar_on_motor_run() -> None:
    thought = make_thought()
    request = HarnessRunRequestV1(
        correlation_id="c-lifecycle",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    runner = HarnessRunner(bus, fcc_runner=_mock_fcc_runner, node_name="test-node")
    lifecycle_publish = AsyncMock()

    with patch(
        "orion.harness.runner.publish_harness_lifecycle_grammar",
        lifecycle_publish,
    ):
        result = await runner.run(request)

    assert result.step_count > 0
    lifecycle_publish.assert_awaited_once()
    events = lifecycle_publish.await_args.kwargs["events"]
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert "exec_request_received" in roles
    assert "exec_plan_started" in roles
    assert "exec_step_started" in roles
    assert "exec_step_completed" in roles
    assert "exec_result_assembled" in roles
    assembled = next(
        e for e in events if e.atom and e.atom.semantic_role == "exec_result_assembled"
    )
    assert "reasoning_present=True" in assembled.atom.summary
    assert "thinking_source=harness_fcc" in assembled.atom.summary


@pytest.mark.asyncio
async def test_harness_runner_uses_compile_harness_prefix() -> None:
    thought = make_thought(imperative="Check logs first.", tone="direct")
    captured: dict[str, str] = {}

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    request = HarnessRunRequestV1(
        correlation_id="c-prefix",
        thought_event=thought,
        user_message="what broke?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)
    await runner.run(request)

    prompt = captured["prompt"]
    assert "Imperative: Check logs first." in prompt
    assert "Tone: direct" in prompt
    assert "what broke?" in prompt
    assert "Execute your imperative" in prompt
