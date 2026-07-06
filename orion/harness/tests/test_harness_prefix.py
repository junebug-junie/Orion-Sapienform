from __future__ import annotations

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


def test_compile_harness_prefix_adds_repo_operator_brief_for_repo_contract() -> None:
    thought = make_thought()
    contract = AnswerContract(
        request_kind="repo_technical",
        requires_repo_grounding=True,
        preferred_render_style="steps",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="where is coalition.py?",
        answer_contract=contract,
    )
    assert "prefer rg/Grep" in prompt
    assert "Requires repo grounding: yes" in prompt
    assert "Request kind: repo_technical" in prompt


def test_harness_motor_instruction_repo_turn() -> None:
    thought = make_thought()
    contract = AnswerContract(request_kind="repo_technical", requires_repo_grounding=True)
    instruction = harness_motor_instruction(thought=thought, answer_contract=contract)
    assert "repo tools" in instruction
