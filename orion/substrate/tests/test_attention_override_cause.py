"""Why no voluntary override fired -- the cause, not just the absence.

Pins the claim this patch sells: before it, `attention_reason` was
`bottom_up_salience` on 19,408 of 19,408 live rows over seven days and the row
recorded that no override happened while recording nothing about *which* of
five real exits produced that. These tests assert that each exit is now
distinguishable, because a reason string that cannot separate the exits would
be exactly as useless as the silence it replaced.

Own helpers deliberately, not an append to test_attention_self_model.py: that
module already defines `_broadcast`/`_override` and appending here would
shadow them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    CuriosityCandidateActionV1,
    OpenLoopV1,
)
from orion.substrate.attention import goal_context as goal_context_mod
from orion.substrate.attention import top_down as top_down_mod
from orion.substrate.attention.top_down import GoalContext
from orion.substrate.attention_broadcast import _apply_voluntary_attention
from orion.substrate.attention_self_model import (
    describe_override_absence,
    reduce_attention_self_model,
)
from orion.schemas.attention_self_model import AttentionSelfModelV1

NOW = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


def _loop(loop_id: str, *, salience: float, concept_value: float = 0.0) -> OpenLoopV1:
    return OpenLoopV1(
        id=loop_id,
        description=f"loop {loop_id}",
        salience=salience,
        concept_value=concept_value,
    )


def _frame(*loops: OpenLoopV1, actions: list[CuriosityCandidateActionV1] | None = None) -> AttentionFrameV1:
    return AttentionFrameV1(
        generated_at=NOW,
        open_loops=list(loops),
        candidate_actions=list(actions or []),
    )


def _enable(monkeypatch: pytest.MonkeyPatch, *, goal: GoalContext | None) -> None:
    """Patch the two module attributes `_apply_voluntary_attention` imports.

    Patched on the real source modules rather than via `sys.modules`, so an
    import-path change in the function under test surfaces as a failure here
    instead of being silently absorbed by a stubbed module object.
    """
    monkeypatch.setattr(top_down_mod, "top_down_enabled", lambda: True)
    monkeypatch.setattr(goal_context_mod, "get_active_goal", lambda: goal)


# Bottom-up would pick loop-b (0.50 > 0.30); concept_value only on loop-a means
# top-down bias lands entirely there and flips the winner. gain=0.6,
# effort_max=1.0 -> combined(a) = 0.30 + 0.6*1.0 = 0.90 > 0.50.
def _flipping_loops() -> tuple[OpenLoopV1, OpenLoopV1]:
    return _loop("loop-a", salience=0.30, concept_value=1.0), _loop("loop-b", salience=0.50)


class TestProducerExits:
    """Each of the five real exits in `_apply_voluntary_attention`."""

    def test_flag_off_records_top_down_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(top_down_mod, "top_down_enabled", lambda: False)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert frame.voluntary_override is None
        assert frame.voluntary_override_absent_reason == "top_down_disabled"

    def test_no_goal_records_no_active_goal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=None)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert frame.voluntary_override_absent_reason == "no_active_goal"

    def test_no_loops_records_no_open_loops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1"))
        frame = _apply_voluntary_attention(_frame())
        assert frame.voluntary_override_absent_reason == "no_open_loops"

    def test_no_goal_and_no_loops_are_not_the_same_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard these two came from was a single `or`. Collapsed, they were
        indistinguishable -- and they mean different things: "Orion wanted
        nothing" vs "there was nothing to want"."""
        _enable(monkeypatch, goal=None)
        no_goal = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        _enable(monkeypatch, goal=GoalContext(priority=1.0))
        no_loops = _apply_voluntary_attention(_frame())
        assert no_goal.voluntary_override_absent_reason != no_loops.voluntary_override_absent_reason

    def test_bias_that_does_not_flip_the_winner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1"))
        # concept_value on the loop that ALREADY wins bottom-up -> bias applied,
        # effort spent, winner unchanged.
        frame = _apply_voluntary_attention(
            _frame(_loop("loop-a", salience=0.50, concept_value=1.0), _loop("loop-b", salience=0.30))
        )
        assert frame.voluntary_override is None
        assert frame.voluntary_override_absent_reason == "bias_did_not_flip_winner"
        assert frame.effort_budget_used > 0.0, "a goal really did run the combiner"

    def test_flip_without_a_candidate_action_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1"))
        frame = _apply_voluntary_attention(_frame(*_flipping_loops()))
        assert frame.voluntary_override is None
        assert frame.voluntary_override_absent_reason == "winner_had_no_action"

    def test_successful_override_sets_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1"))
        action = CuriosityCandidateActionV1(action_type="watch", open_loop_id="loop-a")
        frame = _apply_voluntary_attention(_frame(*_flipping_loops(), actions=[action]))
        assert frame.voluntary_override is not None, "fixture must actually flip the winner"
        assert frame.voluntary_override_absent_reason is None
        assert frame.selected_action is action

    def test_swallowed_exception_is_not_a_quiet_tick(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The never-raises guard must not make a defect look like calm."""
        def _boom() -> bool:
            raise RuntimeError("combiner exploded")

        monkeypatch.setattr(top_down_mod, "top_down_enabled", _boom)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert frame.voluntary_override_absent_reason == "combiner_error"


class TestReducerSurfacesTheCause:
    def _projection(self, frame: AttentionFrameV1, *, generated_at: datetime = NOW):
        return AttentionBroadcastProjectionV1(
            generated_at=generated_at,
            frame=frame,
            selected_action_type="watch",
            selected_open_loop_id="loop-a",
            selected_description="a dispatched open loop",
            coalition_stability_score=0.8,
        )

    def test_reason_and_three_numbers_reach_the_self_model(self) -> None:
        frame = _frame(_loop("loop-a", salience=0.5), _loop("loop-b", salience=0.2))
        frame.voluntary_override_absent_reason = "bias_did_not_flip_winner"
        frame.effort_budget_used = 0.42
        frame.open_loops[0].top_down_bias = 0.7
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert model.voluntary_override_absent_reason == "bias_did_not_flip_winner"
        assert model.top_down_effort_used == pytest.approx(0.42)
        assert model.top_down_bias_max == pytest.approx(0.7)
        assert model.open_loop_count == 2

    def test_absent_lane_reports_unreadable_not_a_cause(self) -> None:
        model = reduce_attention_self_model(None, None, now=NOW)
        assert model.voluntary_override_absent_reason == "broadcast_lane_unreadable"
        assert model.top_down_effort_used is None
        assert model.top_down_bias_max is None
        assert model.open_loop_count is None

    def test_stale_lane_does_not_carry_a_stale_reason_forward(self) -> None:
        """A stale frame's reason answers an earlier tick's question. Reporting
        it here would let an absent reading assert a cause for this tick."""
        stale_at = NOW - timedelta(hours=3)
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.voluntary_override_absent_reason = "no_active_goal"
        frame.effort_budget_used = 0.9
        model = reduce_attention_self_model(
            self._projection(frame, generated_at=stale_at), None, now=NOW
        )
        assert model.voluntary_override_absent_reason == "broadcast_lane_unreadable"
        assert model.top_down_effort_used is None

    def test_narrative_names_the_real_reason(self) -> None:
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.voluntary_override_absent_reason = "top_down_disabled"
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert model.attention_reason == "bottom_up_salience"
        assert "switched off" in model.reason_narrative

    def test_the_old_hardcoded_cause_claim_is_gone(self) -> None:
        """Regression. The previous narrative ended "no active goal override at
        this tick" on every bottom-up row -- asserting a cause the reducer had
        never checked, true for only one of five exits."""
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.voluntary_override_absent_reason = "bias_did_not_flip_winner"
        frame.open_loops[0].top_down_bias = 0.8
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert "no active goal override at this tick" not in model.reason_narrative


class TestDescribeOverrideAbsence:
    def test_irrelevant_goal_and_losing_goal_read_differently(self) -> None:
        """`bias_did_not_flip_winner` covers two different facts about Orion:
        a goal relevant to nothing, and a goal that pushed and lost. The reason
        string alone merges them; the numbers separate them."""
        irrelevant = AttentionSelfModelV1(
            voluntary_override_absent_reason="bias_did_not_flip_winner",
            top_down_bias_max=0.0,
            top_down_effort_used=0.0,
            open_loop_count=4,
        )
        pushed_and_lost = AttentionSelfModelV1(
            voluntary_override_absent_reason="bias_did_not_flip_winner",
            top_down_bias_max=0.9,
            top_down_effort_used=0.9,
            open_loop_count=4,
        )
        assert describe_override_absence(irrelevant) != describe_override_absence(pushed_and_lost)
        assert "relevant to none" in describe_override_absence(irrelevant)
        assert "did not" in describe_override_absence(pushed_and_lost)

    def test_missing_reason_reports_absence_never_a_cause(self) -> None:
        """Rows written before this patch carry no reason. That must read as
        unrecoverable, not as any particular exit."""
        legacy = AttentionSelfModelV1(voluntary_override_absent_reason=None)
        text = describe_override_absence(legacy)
        assert "unrecoverable" in text
        assert "no goal was active" not in text

    def test_every_reason_value_has_its_own_sentence(self) -> None:
        """A reason set that collapses to one sentence would be as useless as
        the silence it replaced."""
        reasons = [
            "top_down_disabled",
            "no_active_goal",
            "no_open_loops",
            "bias_did_not_flip_winner",
            "winner_had_no_action",
            "combiner_error",
            "broadcast_lane_unreadable",
        ]
        rendered = {
            describe_override_absence(
                AttentionSelfModelV1(voluntary_override_absent_reason=r, top_down_bias_max=0.5)
            )
            for r in reasons
        }
        assert len(rendered) == len(reasons)
