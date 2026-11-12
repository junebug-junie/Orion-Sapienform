"""Living Prime Part 1: the Prime block reaches the motor prompt, after the Situation block."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.prefix import PRIME_BLOCK_BRIEF, compile_harness_prefix
from orion.harness.runner import HarnessRunner, build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRepairOverlayV1, HarnessRunRequestV1

BLOCK = "PRIME (my own slow loop, what I hold between turns), last tick 10 s ago:\n- Holding (2 h): heat [prime_concern:c1]"
SITUATION = "Situation:\n- Local context: midday."


def _prefix(**kw: Any) -> str:
    return compile_harness_prefix(make_thought(), repair_overlay=HarnessRepairOverlayV1(), user_message="hi", **kw)


def test_prefix_places_the_block_right_after_the_situation_block_with_its_brief() -> None:
    p = _prefix(situation_prompt_fragment=SITUATION, prime_block=BLOCK)
    assert BLOCK in p and PRIME_BLOCK_BRIEF in p
    assert p.index(SITUATION) < p.index(BLOCK) < p.index("TASK THIS TURN")


def test_prefix_without_a_block_is_unchanged() -> None:
    assert _prefix(situation_prompt_fragment=SITUATION) == _prefix(situation_prompt_fragment=SITUATION, prime_block=None)
    assert "PRIME" not in _prefix() and PRIME_BLOCK_BRIEF not in _prefix(prime_block="")


def test_build_harness_prompt_threads_the_block() -> None:
    prompt = build_harness_prompt(thought=make_thought(), user_message="hi", repair_overlay=HarnessRepairOverlayV1(),
                                  prime_block=BLOCK)
    assert BLOCK in prompt


@pytest.mark.asyncio
async def test_runner_puts_the_request_block_in_the_motor_prompt_and_motor_boot_step() -> None:
    captured: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []

    async def _fcc(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        captured.update(kwargs)
        yield {"type": "final", "llm_response": "answer", "metadata": {"exit_code": 0}}

    async def _step(bus: Any, *, step: dict[str, Any], **_: Any) -> None:
        steps.append(step)

    request = HarnessRunRequestV1(correlation_id="c-prime", thought_event=make_thought(), user_message="hi",
                                  permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(),
                                  prime_block=BLOCK)
    with (patch("orion.harness.runner.publish_harness_run_step", _step),
          patch("orion.harness.runner.read_last_tool_fetch", AsyncMock(return_value=None))):
        await HarnessRunner(AsyncMock(), fcc_runner=_fcc).run(request)
    assert BLOCK in captured["prompt"]
    boot = [s for s in steps if s.get("_cockpit") == "cockpit.motor_boot.v1"]
    assert boot and BLOCK in boot[0]["prompt"]


def test_request_without_the_field_still_parses() -> None:
    """An older Hub sends no prime_block: the governor renders no block."""
    full = HarnessRunRequestV1(correlation_id="c", thought_event=make_thought(), user_message="hi",
                               permissions=ContextExecPermissionV1(), answer_contract=AnswerContract())
    wire = full.model_dump(mode="json")
    wire.pop("prime_block")
    assert HarnessRunRequestV1.model_validate(wire).prime_block is None
