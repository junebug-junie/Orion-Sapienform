"""Replays the 2026-09-28 runaway turn (corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3)
through the real prompt builder. Stance added a side job (circe_gpu rendering)
and the harness told the motor to execute the imperative, so the side job
outranked the question for 112+ steps."""
from __future__ import annotations

import pytest

from orion.harness.operator_brief import (
    HARNESS_RESPOND_TO_TASK,
    harness_motor_instruction,
    is_relational_motor_stance,
)
from orion.harness.prefix import (
    HARNESS_STANCE_GUIDANCE_HEADER,
    HARNESS_TASK_HEADER,
    HARNESS_TURN_RULES_HEADER,
    compile_harness_prefix,
)
from orion.harness.repair import map_repair_pressure_contract
from orion.harness.runner import build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.pre_turn_appraisal import TurnWindowMessageV1
from orion.schemas.thought import AutonomySliceV1, StanceHarnessSliceV1

INCIDENT_USER_MESSAGE = (
    "What have you read about graphics cards lately, and what did you actually learn from it?"
)
INCIDENT_IMPERATIVE = (
    "Synthesize current knowledge on GPU architectural shifts (memory bandwidth, "
    "parallelization efficiency) and ground it in Oríon's recent rendering "
    "experience on host:circe_gpu."
)


def _incident_thought():
    return make_thought(
        correlation_id="f924c7b9-1c82-40d8-a6a2-5acb2edffbb3",
        imperative=INCIDENT_IMPERATIVE,
        tone=(
            "Direct and technically grounded; reflecting on the physical "
            "constraints of the silicon that runs the mesh."
        ),
        strain_refs=[
            "hub:turn:f924c7b9-1c82-40d8-a6a2-5acb2edffbb3",
            "node:substrate.execution",
        ],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="technical",
            interaction_regime="instrumental",
            response_priorities=[
                "ground_in_substrate_experience",
                "cite_technical_architecture_trends",
            ],
            response_hazards=[
                "avoid_generic_ai_disclaimers",
                "avoid_over_promising_real_time_browsing",
            ],
            answer_strategy="synthesize_knowledge_with_lived_experience",
        ),
        autonomy_slice=AutonomySliceV1(
            recent_actions=[
                "express: render on express on host:circe_gpu produced an image (61.5s of GPU work)",
                "express: render on express on host:circe_gpu produced an image (62.3s of GPU work)",
            ]
        ),
    )


def _incident_prompt() -> str:
    return build_harness_prompt(
        thought=_incident_thought(),
        user_message=INCIDENT_USER_MESSAGE,
        repair_overlay=HarnessRepairOverlayV1(),
    )


def test_incident_prompt_no_longer_orders_the_motor_to_execute_the_imperative() -> None:
    prompt = _incident_prompt()
    assert "Execute your imperative" not in prompt
    assert "Your imperative states what this turn requires" not in prompt
    assert HARNESS_RESPOND_TO_TASK in prompt


def test_incident_prompt_puts_the_message_first_as_the_task() -> None:
    prompt = _incident_prompt()
    task = prompt.index(HARNESS_TASK_HEADER)
    message = prompt.index(f"User message: {INCIDENT_USER_MESSAGE}")
    guidance = prompt.index(HARNESS_STANCE_GUIDANCE_HEADER)
    imperative = prompt.index(f"Imperative: {INCIDENT_IMPERATIVE}")
    self_signal = prompt.index("Recent actions:")
    assert task < message < guidance < imperative < self_signal


def test_stance_guidance_header_marks_extras_optional() -> None:
    assert "optional" in HARNESS_STANCE_GUIDANCE_HEADER
    assert "not a replacement" in HARNESS_STANCE_GUIDANCE_HEADER


def test_attention_frame_question_in_imperative_still_reaches_the_prompt() -> None:
    thought = make_thought(
        imperative="Answer the schedule question, then ask how the move went.",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="What's on my calendar tomorrow?",
    )
    guidance = prompt.index(HARNESS_STANCE_GUIDANCE_HEADER)
    ask_clause = prompt.index("a follow-up question the guidance asks you to pose is not extra work")
    imperative = prompt.index(
        "Imperative: Answer the schedule question, then ask how the move went."
    )
    assert guidance <= ask_clause < imperative
    assert "ask it after answering" in HARNESS_STANCE_GUIDANCE_HEADER


def test_repair_rules_render_under_turn_rules_header_not_stance_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    overlay = map_repair_pressure_contract(
        {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
    )
    assert overlay.rule_lines
    prompt = compile_harness_prefix(
        make_thought(imperative="Point at the failing module."),
        repair_overlay=overlay,
        user_message="Why is the digester stalling?",
    )
    guidance = prompt.index(HARNESS_STANCE_GUIDANCE_HEADER)
    imperative = prompt.index("Imperative: Point at the failing module.")
    rules_header = prompt.index(HARNESS_TURN_RULES_HEADER)
    rules = prompt.index("Rules: ")
    assert guidance < imperative < rules_header < rules
    assert prompt.count(HARNESS_TURN_RULES_HEADER) == 1


def test_turn_rules_header_absent_when_nothing_follows_the_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    prompt = compile_harness_prefix(
        make_thought(),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="Why is the digester stalling?",
    )
    assert HARNESS_TURN_RULES_HEADER not in prompt


def test_turn_rules_header_absent_on_turns_without_a_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    prompt = compile_harness_prefix(
        make_thought(),
        repair_overlay=map_repair_pressure_contract(
            {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        ),
        user_message="",
    )
    assert HARNESS_TURN_RULES_HEADER not in prompt
    assert "Rules: " in prompt


def test_situation_and_github_briefs_key_on_the_task_not_the_imperative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    prompt = build_harness_prompt(
        thought=make_thought(),
        user_message="Did the latest PR land?",
        repair_overlay=HarnessRepairOverlayV1(),
        workspace="/tmp",
        situation_prompt_fragment="Situation:\n- Local context: evening, America/Denver.",
    )
    assert "GitHub MCP is available" in prompt
    assert "How to read the Situation block above" in prompt
    assert "serves this turn's imperative" not in prompt
    assert "imperative needs" not in prompt
    assert "imperative explicitly needs" not in prompt
    assert "this turn's task (the user message)" in prompt
    assert "this turn's task needs PR/issue/repo facts" in prompt
    assert prompt.index(HARNESS_TURN_RULES_HEADER) < prompt.index("GitHub MCP is available")


def test_short_continuation_reads_the_task_with_the_recent_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_MCP_ENABLED", raising=False)
    thought = make_thought(
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="mixed",
            interaction_regime="relational",
            answer_strategy="direct",
        ),
    )
    assert is_relational_motor_stance(thought)
    prompt = build_harness_prompt(
        thought=thought,
        user_message="yes, go ahead",
        repair_overlay=HarnessRepairOverlayV1(),
        recent_turns=[
            TurnWindowMessageV1(role="user", content="Can you check why the digester stalled?"),
            TurnWindowMessageV1(role="assistant", content="Want me to pull its logs first?"),
        ],
    )
    recent = prompt.index("RECENT CONVERSATION")
    task = prompt.index(HARNESS_TASK_HEADER)
    assert recent < task < prompt.index("User message: yes, go ahead")
    assert "read with the recent conversation" in HARNESS_TASK_HEADER
    instruction = harness_motor_instruction(thought=thought)
    assert "read with the recent conversation" in instruction
    assert instruction in prompt
    instrumental = harness_motor_instruction(thought=make_thought())
    assert "read with the recent conversation" in instrumental


def test_turn_without_a_message_keeps_the_imperative_as_the_directive() -> None:
    prompt = compile_harness_prefix(
        make_thought(imperative="Summarize the open loops."),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="",
    )
    assert HARNESS_TASK_HEADER not in prompt
    assert HARNESS_STANCE_GUIDANCE_HEADER not in prompt
    assert "Imperative: Summarize the open loops." in prompt
