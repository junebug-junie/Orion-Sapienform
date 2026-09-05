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
    VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY,
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


_GOAL_TARGET = "node:substrate.wanted"


def _loop(loop_id: str, *, salience: float, goal_relevant: bool = False) -> OpenLoopV1:
    """Build a loop; ``goal_relevant`` puts it on the node the test goal targets.

    Was ``concept_value=`` until 2026-09-05. relevance() no longer reads any
    per-loop score field -- it joins ``GoalContext.target_id`` against
    ``source_refs`` -- so steering these tests by concept_value silently made
    every loop equally relevant and no override could fire.
    """
    return OpenLoopV1(
        id=loop_id,
        description=f"desc {loop_id}",
        salience=salience,
        source_refs=[_GOAL_TARGET] if goal_relevant else ["node:substrate.unrelated"],
    )


def _frame(*loops: OpenLoopV1, actions: list[CuriosityCandidateActionV1] | None = None) -> AttentionFrameV1:
    return AttentionFrameV1(
        generated_at=NOW,
        open_loops=list(loops),
        candidate_actions=list(actions or []),
    )


def _reason(frame: AttentionFrameV1) -> str | None:
    """The producer's reason, read where it actually lives."""
    return frame.debug.get(VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY)


def _enable(monkeypatch: pytest.MonkeyPatch, *, goal: GoalContext | None) -> None:
    """Patch the two module attributes `_apply_voluntary_attention` imports.

    Patched on the real source modules rather than via `sys.modules`, so an
    import-path change in the function under test surfaces as a failure here
    instead of being silently absorbed by a stubbed module object.
    """
    monkeypatch.setattr(top_down_mod, "top_down_enabled", lambda: True)
    monkeypatch.setattr(goal_context_mod, "get_active_goal", lambda: goal)


# Bottom-up would pick loop-b (0.50 > 0.30); only loop-a sits on the goal's target, so
# top-down bias lands entirely there and flips the winner. gain=0.6,
# effort_max=1.0 -> combined(a) = 0.30 + 0.6*1.0 = 0.90 > 0.50.
def _flipping_loops() -> tuple[OpenLoopV1, OpenLoopV1]:
    return _loop("loop-a", salience=0.30, goal_relevant=True), _loop("loop-b", salience=0.50)


class TestProducerExits:
    """Each of the five real exits in `_apply_voluntary_attention`."""

    def test_flag_off_records_top_down_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(top_down_mod, "top_down_enabled", lambda: False)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert frame.voluntary_override is None
        assert _reason(frame) == "top_down_disabled"

    def test_no_goal_records_no_active_goal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=None)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert _reason(frame) == "no_active_goal"

    def test_no_loops_records_no_open_loops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        frame = _apply_voluntary_attention(_frame())
        assert _reason(frame) == "no_open_loops"

    def test_no_goal_and_no_loops_are_not_the_same_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard these two came from was a single `or`. Collapsed, they were
        indistinguishable -- and they mean different things: "Orion wanted
        nothing" vs "there was nothing to want"."""
        _enable(monkeypatch, goal=None)
        no_goal = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        _enable(monkeypatch, goal=GoalContext(priority=1.0, target_id=_GOAL_TARGET))
        no_loops = _apply_voluntary_attention(_frame())
        assert _reason(no_goal) != _reason(no_loops)

    def test_goal_targeting_the_already_winning_loop_is_not_a_defeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Split out 2026-09-05. This scenario -- the goal wanting the loop that
        already wins -- used to record `bias_did_not_flip_winner`, the same
        string as a goal that pushed a different loop and lost. Measured live,
        that conflation was most of the signal: the goal usually wants `chat`,
        which often wins on salience anyway, so a low override rate looked like
        Orion losing when it was Orion agreeing."""
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        # the goal targets the loop that ALREADY wins bottom-up -> bias applied,
        # effort spent, winner unchanged.
        frame = _apply_voluntary_attention(
            _frame(_loop("loop-a", salience=0.50, goal_relevant=True), _loop("loop-b", salience=0.30))
        )
        assert frame.voluntary_override is None
        assert _reason(frame) == "goal_target_already_winning"
        assert frame.effort_budget_used > 0.0, "a goal really did run the combiner"

    def test_goal_about_nothing_competing_is_not_a_defeat_either(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The third flavour: the goal's target is not in the competition at
        all, so every bias is 0.0. Measured live at 33 of 43 non-firing ticks --
        the single largest slice, and the one most wrongly read as "goals keep
        losing"."""
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        frame = _apply_voluntary_attention(
            _frame(_loop("loop-a", salience=0.50), _loop("loop-b", salience=0.30))
        )
        assert frame.voluntary_override is None
        assert _reason(frame) == "goal_matched_no_loop"

    def test_a_real_defeat_still_reads_as_a_defeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bias lands on a loop that then loses -- the ONLY one of the three
        that is a genuine competitive loss. If the split ever mislabels this as
        agreement it would erase the evidence that Orion tried and failed."""
        _enable(monkeypatch, goal=GoalContext(priority=0.05, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        frame = _apply_voluntary_attention(
            _frame(_loop("loop-a", salience=0.95), _loop("loop-b", salience=0.30, goal_relevant=True))
        )
        assert frame.voluntary_override is None
        assert _reason(frame) == "bias_did_not_flip_winner"

    def test_flip_without_a_candidate_action_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        frame = _apply_voluntary_attention(_frame(*_flipping_loops()))
        assert frame.voluntary_override is None
        assert _reason(frame) == "winner_had_no_action"

    def test_successful_override_sets_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        action = CuriosityCandidateActionV1(action_type="watch", open_loop_id="loop-a")
        frame = _apply_voluntary_attention(_frame(*_flipping_loops(), actions=[action]))
        assert frame.voluntary_override is not None, "fixture must actually flip the winner"
        assert _reason(frame) is None
        assert frame.selected_action is action

    def test_a_crash_INSIDE_the_combiner_is_not_goal_irrelevance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The combiner's own never-raise guard (top_down.py Rule 8) falls back
        to `_pure_bottom_up`, which returns a POPULATED result with every bias
        0.0 and a real winner -- byte-identical to a legitimate "the goal was
        relevant to nothing" outcome. Before `TopDownResult.failed`, a crash in
        here was recorded as `bias_did_not_flip_winner` and narrated as a
        confident claim about goal quality. Found by code review 2026-09-04."""
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))

        def _boom(goal: object, loop: object) -> float:
            raise RuntimeError("relevance exploded")

        monkeypatch.setattr(top_down_mod, "relevance", _boom)
        frame = _apply_voluntary_attention(
            _frame(_loop("loop-a", salience=0.5, goal_relevant=True), _loop("loop-b", salience=0.3))
        )
        assert _reason(frame) == "combiner_error"
        assert _reason(frame) != "bias_did_not_flip_winner"

    def test_success_clears_a_reason_the_caller_already_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`frame.debug` is caller-supplied and survives, so clearing on the
        success path is load-bearing rather than defensive. Deleting the clear
        must fail this test -- an earlier version asserted only against the
        schema default and could not fail at all (code review 2026-09-04)."""
        _enable(monkeypatch, goal=GoalContext(priority=1.0, goal_artifact_id="g-1", target_id=_GOAL_TARGET))
        action = CuriosityCandidateActionV1(action_type="watch", open_loop_id="loop-a")
        frame = _frame(*_flipping_loops(), actions=[action])
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "no_active_goal"
        result = _apply_voluntary_attention(frame)
        assert result.voluntary_override is not None
        assert _reason(result) is None, "a fired override must claim no absence"

    def test_swallowed_exception_outside_the_combiner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the never-raises surface: an exception raised
        outside `TopDownBiasCombiner.apply` (here, from `top_down_enabled`)."""
        def _boom() -> bool:
            raise RuntimeError("combiner exploded")

        monkeypatch.setattr(top_down_mod, "top_down_enabled", _boom)
        frame = _apply_voluntary_attention(_frame(_loop("loop-a", salience=0.4)))
        assert _reason(frame) == "combiner_error"


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
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "bias_did_not_flip_winner"
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
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "no_active_goal"
        frame.effort_budget_used = 0.9
        model = reduce_attention_self_model(
            self._projection(frame, generated_at=stale_at), None, now=NOW
        )
        assert model.voluntary_override_absent_reason == "broadcast_lane_unreadable"
        assert model.top_down_effort_used is None

    def test_no_loops_reports_no_measurement_not_a_zero(self) -> None:
        """A fresh lane carrying zero loops took no relevance measurement.
        `max(default=0.0)` would assert a reading that never happened, and an
        aggregate filtering `top_down_bias_max == 0` would then count empty
        ticks as "the goal was irrelevant". Code review 2026-09-04: this path
        had no test, and mutating the default to 1.0 changed nothing."""
        frame = _frame()
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "no_open_loops"
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert model.open_loop_count == 0
        assert model.top_down_bias_max is None, "no loops means no measurement, not zero"

    def test_a_faint_bias_is_not_reported_as_no_relevance(self) -> None:
        """`top_down_bias_max` is stored unrounded. Rounded to 4 places a real
        bias of 0.00004 becomes 0.0, flipping the narrative into claiming the
        goal was relevant to nothing -- false, and it is the sentence Orion
        reads back about themselves."""
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.open_loops[0].top_down_bias = 0.00004
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "bias_did_not_flip_winner"
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert model.top_down_bias_max == pytest.approx(0.00004)
        assert "relevant to none" not in model.reason_narrative

    def test_narrative_names_the_real_reason(self) -> None:
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "top_down_disabled"
        model = reduce_attention_self_model(self._projection(frame), None, now=NOW)
        assert model.attention_reason == "bottom_up_salience"
        assert "switched off" in model.reason_narrative

    def test_the_old_hardcoded_cause_claim_is_gone(self) -> None:
        """Regression. The previous narrative ended "no active goal override at
        this tick" on every bottom-up row -- asserting a cause the reducer had
        never checked, true for only one of five exits."""
        frame = _frame(_loop("loop-a", salience=0.5))
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = "bias_did_not_flip_winner"
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
        assert "Legacy row" in describe_override_absence(irrelevant), (
            "a pre-split row's cause is INFERRED from the bias number, not "
            "recorded -- the narrative must say so rather than presenting it "
            "with the same confidence as a classified row"
        )
        assert "then lost" in describe_override_absence(pushed_and_lost)

    def test_the_three_absence_flavours_read_differently(self) -> None:
        """The whole point of the split: these mean different things about
        whether Orion has any say, and must not share prose."""
        def _m(reason, bias):
            return AttentionSelfModelV1(
                voluntary_override_absent_reason=reason,
                top_down_bias_max=bias, top_down_effort_used=bias, open_loop_count=4,
            )
        no_overlap = describe_override_absence(_m("goal_matched_no_loop", 0.0))
        already_won = describe_override_absence(_m("goal_target_already_winning", 1.0))
        lost = describe_override_absence(_m("bias_did_not_flip_winner", 0.9))

        assert len({no_overlap, already_won, lost}) == 3
        # Each must actually say its own thing, not merely differ by a number.
        assert "different things" in no_overlap
        assert "nothing to override" in already_won
        assert "competitive defeat" in lost
        # And the two that are NOT defeats must not read as defeats.
        assert "defeat" not in no_overlap
        assert "defeat" not in already_won

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
