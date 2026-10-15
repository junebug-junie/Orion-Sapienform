"""Self-sense rows record how many harness steps the answering turn took."""

from __future__ import annotations

from orion.evals.self_sense_runner import build_row

_COMMON = dict(
    run_id="r1",
    question_key="who_matters",
    question="q",
    http_text="Juniper matters.",
    trace_text=None,
    correlation_id="00000000-0000-4000-8000-000000000002",
    self_definition_version=1,
)


def test_motor_steps_lands_in_notes_so_a_trend_can_be_split_by_tool_use():
    assert "motor_steps=60" in build_row(**_COMMON, motor_steps=60).notes


def test_zero_steps_is_recorded_not_dropped():
    assert "motor_steps=0" in build_row(**_COMMON, motor_steps=0).notes


def test_unknown_steps_adds_no_note():
    assert "motor_steps" not in (build_row(**_COMMON).notes or "")
