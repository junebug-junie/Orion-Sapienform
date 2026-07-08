"""Wiring: goal-context store + _apply_voluntary_attention over a real frame."""
from __future__ import annotations

import pytest

from orion.schemas.attention_frame import (
    AttentionFrameV1,
    CuriosityCandidateActionV1,
    OpenLoopV1,
    VoluntaryOverrideV1,
)
from orion.core.schemas.drives import GoalProposalV1
from orion.substrate.attention_broadcast import _apply_voluntary_attention
from orion.substrate.attention import goal_context as gc


def _loop(id: str, salience: float, **rel) -> OpenLoopV1:
    return OpenLoopV1(id=id, description=f"loop {id}", salience=salience, **rel)


def _frame() -> AttentionFrameV1:
    # loop A: high bottom-up, no goal relevance. loop B: low bottom-up, high predictive_value.
    loops = [
        _loop("A", 0.80, predictive_value=0.0),
        _loop("B", 0.30, predictive_value=0.95),
    ]
    actions = [
        CuriosityCandidateActionV1(action_type="watch", open_loop_id="A", score=0.80),
        CuriosityCandidateActionV1(action_type="watch", open_loop_id="B", score=0.30),
    ]
    return AttentionFrameV1(open_loops=loops, candidate_actions=actions, selected_action=actions[0])


def _goal(drive="predictive", priority=0.9, status="proposed") -> GoalProposalV1:
    return GoalProposalV1(
        artifact_id="goal-1", subject="orion", model_layer="self-model",
        entity_id="self:orion", kind="autonomy.goal.proposed.v1",
        goal_statement="pursue predictive", proposal_signature="sig-1",
        drive_origin=drive, priority=priority,
        proposal_status=status, provenance={"intake_channel": "x"},
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
    gc.set_active_goal(_goal(drive="predictive", priority=0.9))
    frame = _apply_voluntary_attention(_frame())
    # b(B) = 0.9*0.95 = 0.855; combined(B) = 0.30 + 0.6*0.855 = 0.813 > 0.80 (A).
    assert frame.voluntary_override is not None
    assert frame.voluntary_override.chosen_loop_id == "B"
    assert frame.voluntary_override.beat_loop_id == "A"
    assert frame.selected_action.open_loop_id == "B"  # re-pointed to winner
    assert frame.effort_budget_used > 0.0
    b_loop = next(l for l in frame.open_loops if l.id == "B")
    assert 0.0 <= b_loop.combined_salience <= 1.0 and b_loop.top_down_bias > 0.0


def test_strong_bottom_up_beats_weak_goal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "true")
    gc.set_active_goal(_goal(drive="predictive", priority=0.2))  # weak
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
