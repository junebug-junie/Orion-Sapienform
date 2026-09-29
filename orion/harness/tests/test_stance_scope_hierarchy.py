"""Replays the 2026-09-28 runaway turn (corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3)
through the real prompt builder. Stance added a side job (circe_gpu rendering)
and the harness told the motor to execute the imperative, so the side job
outranked the question for 112+ steps."""
from __future__ import annotations

from orion.harness.operator_brief import HARNESS_RESPOND_TO_TASK
from orion.harness.prefix import (
    HARNESS_STANCE_GUIDANCE_HEADER,
    HARNESS_TASK_HEADER,
    compile_harness_prefix,
)
from orion.harness.runner import build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
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
    assert prompt.index(HARNESS_STANCE_GUIDANCE_HEADER) < prompt.index(
        "Imperative: Answer the schedule question, then ask how the move went."
    )


def test_turn_without_a_message_keeps_the_imperative_as_the_directive() -> None:
    prompt = compile_harness_prefix(
        make_thought(imperative="Summarize the open loops."),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="",
    )
    assert HARNESS_TASK_HEADER not in prompt
    assert HARNESS_STANCE_GUIDANCE_HEADER not in prompt
    assert "Imperative: Summarize the open loops." in prompt
