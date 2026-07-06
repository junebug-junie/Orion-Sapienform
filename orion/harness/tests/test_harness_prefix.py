from __future__ import annotations

from orion.harness.operator_brief import HARNESS_MOTOR_MAX_READ_LINES
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_compile_harness_prefix_includes_stance_slice() -> None:
    thought = make_thought(imperative="Inspect the module.", tone="direct")
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how does coalition work?",
    )
    assert "Task mode: direct_response" in prompt
    assert "Answer strategy: direct" in prompt
    assert "how does coalition work?" in prompt


def test_compile_harness_prefix_imperative_first_unified_brief() -> None:
    thought = make_thought(
        imperative="Search orion/thought for stance_react; cite file paths.",
        tone="direct",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="implement a function",
    )
    assert "Tools are available from the start" in prompt
    assert "Search orion/thought for stance_react" in prompt
    assert "Imperative: Search orion/thought" in prompt
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt
    assert "Orion harness motor — repo/technical turn" not in prompt  # old gated brief
    assert "prefer rg/Grep" in prompt  # unified brief includes tool guidance
    assert f"longer than {HARNESS_MOTOR_MAX_READ_LINES} lines" in prompt


def test_compile_harness_prefix_ignores_answer_contract() -> None:
    thought = make_thought()
    contract = AnswerContract(
        request_kind="repo_technical",
        requires_repo_grounding=True,
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        answer_contract=contract,
    )
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt


def test_harness_motor_instruction_imperative_forward() -> None:
    thought = make_thought(imperative="Inspect docker logs for orion-hub.")
    instruction = harness_motor_instruction(thought=thought, answer_contract=None)
    assert "Execute your imperative" in instruction
    assert f"over {HARNESS_MOTOR_MAX_READ_LINES} lines" in instruction
    assert "rg/Grep" in instruction
