"""Self-prediction A1: stance's bet never reaches the motor or the judge.

A prediction the motor can see becomes an order and fulfils itself; one the judge grades
against becomes a compliance check. Either way the bet stops measuring self-knowledge.
Spec: docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md.
"""
from __future__ import annotations

import json

from orion.harness.finalize import build_finalize_reflect_context
from orion.harness.prefix import compile_harness_prefix, harness_motor_instruction
from orion.harness.runner import build_harness_prompt
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought
from orion.schemas.thought import TurnSelfPredictionV1

# Distinctive values that appear nowhere else in a rendered prompt.
_BET = TurnSelfPredictionV1(
    asks_question_p=0.37,
    fetches_outside_p=0.63,
    steps_lo=7771,
    steps_hi=8881,
    reply_chars_lo=4241,
    reply_chars_hi=9191,
)
_MARKERS = ("self_prediction", "asks_question_p", "steps_lo", "7771", "8881", "4241", "9191", "0.37", "0.63")


def _assert_clean(text: str) -> None:
    leaked = [m for m in _MARKERS if m in text]
    assert not leaked, f"self_prediction leaked: {leaked}"


def test_harness_prefix_and_motor_instruction_exclude_the_bet() -> None:
    thought = make_thought(self_prediction=_BET)
    _assert_clean(compile_harness_prefix(thought, repair_overlay=make_repair_overlay(), user_message="howdy"))
    _assert_clean(harness_motor_instruction(thought=thought, answer_contract=None))


def test_full_motor_prompt_excludes_the_bet() -> None:
    thought = make_thought(self_prediction=_BET)
    _assert_clean(build_harness_prompt(thought=thought, user_message="howdy", repair_overlay=make_repair_overlay()))


def test_finalize_judge_context_excludes_the_bet() -> None:
    thought = make_thought(self_prediction=_BET)
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="howdy",
    )
    _assert_clean(json.dumps(ctx, default=str))
    # The rest of the stance still reaches the judge.
    assert ctx["thought_event"]["imperative"] == thought.imperative
