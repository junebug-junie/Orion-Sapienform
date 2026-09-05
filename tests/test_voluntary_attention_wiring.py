"""Wiring: goal-context store + _apply_voluntary_attention over a real frame."""
from __future__ import annotations

import pytest

from orion.schemas.attention_frame import (
    AttentionFrameV1,
    CuriosityCandidateActionV1,
    OpenLoopV1,
    VoluntaryOverrideV1,
)
from orion.schemas.field_goal import FieldGoalProvenanceV1
from orion.substrate.attention_broadcast import _apply_voluntary_attention
from orion.substrate.attention import goal_context as gc


_GOAL_TARGET = "node:substrate.execution"


def _loop(id: str, salience: float, **rel) -> OpenLoopV1:
    return OpenLoopV1(id=id, description=f"loop {id}", salience=salience, **rel)


def _frame() -> AttentionFrameV1:
    # loop A: high bottom-up, NOT the goal's target. loop B: low bottom-up, but
    # sitting on the node _goal() targets -- so only B is relevant.
    #
    # Steered by source_refs since 2026-09-05: relevance() joins
    # GoalContext.target_id to OpenLoopV1.source_refs. It previously read
    # loop.concept_value and ignored the goal entirely; setting concept_value
    # here now makes BOTH loops equally (ir)relevant, no override can fire, and
    # this file -- the only end-to-end wiring test of the feature -- goes red.
    # The node id must stay in sync with _goal()'s field_target_id default.
    loops = [
        _loop("A", 0.80, source_refs=["node:substrate.unrelated"]),
        _loop("B", 0.30, source_refs=[_GOAL_TARGET]),
    ]
    actions = [
        CuriosityCandidateActionV1(action_type="watch", open_loop_id="A", score=0.80),
        CuriosityCandidateActionV1(action_type="watch", open_loop_id="B", score=0.30),
    ]
    return AttentionFrameV1(open_loops=loops, candidate_actions=actions, selected_action=actions[0])


def _goal(field_target_id=_GOAL_TARGET, priority=0.9, status="proposed") -> FieldGoalProvenanceV1:
    return FieldGoalProvenanceV1(
        artifact_id="goal-1", subject="attention", model_layer="field_attention",
        entity_id=field_target_id, kind="memory.field_goals.proposed.v1",
        field_target_id=field_target_id, target_kind="node",
        salience_score=priority, source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1", priority=priority,
        proposal_status=status, provenance={"intake_channel": "internal.attention_runtime"},
    )


@pytest.fixture(autouse=True)
def _clear_goal():
    gc.clear_active_goal()
    yield
    gc.clear_active_goal()


def test_flag_off_frame_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "false")
    gc.set_active_goal(_goal())
    frame = _apply_voluntary_attention(_frame())
    assert frame.voluntary_override is None
    assert all(loop.top_down_bias == 0.0 for loop in frame.open_loops)
    assert frame.selected_action.open_loop_id == "A"  # bottom-up winner


def test_flag_on_no_goal_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "true")
    frame = _apply_voluntary_attention(_frame())
    assert frame.voluntary_override is None
    assert frame.selected_action.open_loop_id == "A"


def test_goal_overrides_low_salience_loop(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    gc.set_active_goal(_goal(priority=0.9))
    frame = _apply_voluntary_attention(_frame())
    # b(B) = 0.9*0.95 = 0.855; combined(B) = 0.30 + 0.6*0.855 = 0.813 > 0.80 (A).
    assert frame.voluntary_override is not None
    assert frame.voluntary_override.chosen_loop_id == "B"
    assert frame.voluntary_override.beat_loop_id == "A"
    assert frame.selected_action.open_loop_id == "B"  # re-pointed to winner
    assert frame.effort_budget_used > 0.0
    b_loop = next(l for l in frame.open_loops if l.id == "B")
    assert 0.0 <= b_loop.combined_salience <= 1.0 and b_loop.top_down_bias > 0.0


def test_topdown_applies_regardless_of_stale_salience_v2_flag(monkeypatch) -> None:
    """2026-07-31: `_apply_voluntary_attention` used to also gate on
    `ORION_ATTENTION_SALIENCE_V2_ENABLED` ("only layer top-down when v2 is
    the active selection basis") because `select_actions` could otherwise
    rank by a different, legacy weighted-sum formula. That legacy formula
    is deleted (`scoring.py::score_loop()` has exactly one formula now,
    unconditionally), so the two bases can never disagree regardless of
    this flag -- the gate itself was removed as a zombie check whose
    rationale no longer applies. Top-down layering is now gated ONLY by
    `ORION_ATTENTION_TOPDOWN_ENABLED`; this flag's value must have no
    effect on it (formerly the exact opposite of this assertion)."""
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "false")
    gc.set_active_goal(_goal(priority=0.9))
    frame = _apply_voluntary_attention(_frame())
    # b(B) = 0.9*0.95 = 0.855; combined(B) = 0.30 + 0.6*0.855 = 0.813 > 0.80 (A).
    assert frame.voluntary_override is not None
    assert frame.voluntary_override.chosen_loop_id == "B"
    assert frame.selected_action.open_loop_id == "B"


def test_terminal_goal_clears_store() -> None:
    gc.set_active_goal(_goal(priority=0.9, status="active"))
    assert gc.get_active_goal() is not None
    # same goal (artifact_id) goes terminal -> store clears.
    gc.set_active_goal(_goal(priority=0.9, status="completed"))
    assert gc.get_active_goal() is None


def test_strong_bottom_up_beats_weak_goal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    gc.set_active_goal(_goal(priority=0.2))  # weak
    frame = _apply_voluntary_attention(_frame())
    # b(B)=0.2*0.95=0.19; combined(B)=0.30+0.6*0.19=0.414 < 0.80 (A). No flip.
    assert frame.voluntary_override is None
    assert frame.selected_action.open_loop_id == "A"


def test_goal_store_ignores_non_active_status() -> None:
    gc.set_active_goal(_goal(status="archived"))  # terminal -> ignored
    assert gc.get_active_goal() is None
    gc.set_active_goal(_goal(status="proposed"))
    assert gc.get_active_goal() is not None


def test_override_roundtrips() -> None:
    ov = VoluntaryOverrideV1(chosen_loop_id="B", beat_loop_id="A", chosen_bottom_up=0.3,
                             beat_bottom_up=0.8, applied_bias=0.5, effort_spent=0.5)
    assert VoluntaryOverrideV1.model_validate(ov.model_dump()).chosen_loop_id == "B"
