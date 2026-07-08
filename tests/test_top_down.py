"""Unit tests for voluntary attention override (biased competition)."""

from __future__ import annotations

from orion.schemas.attention_frame import OpenLoopV1, VoluntaryOverrideV1
from orion.substrate.attention.top_down import (
    GoalContext,
    LoopScore,
    TopDownBiasCombiner,
    TopDownConfig,
    TopDownResult,
    relevance,
)


def _loop(id: str, **relevance_fields: float) -> OpenLoopV1:
    """Construct an OpenLoopV1 with the given id and relevance field overrides."""
    return OpenLoopV1(id=id, description="d", **relevance_fields)


def _all_bounded(res: TopDownResult) -> bool:
    for ls in res.per_loop.values():
        if not (0.0 <= ls.top_down_bias <= 1.0):
            return False
        if not (0.0 <= ls.combined_salience <= 1.0):
            return False
    return True


def test_1_no_goal_is_pure_bottom_up():
    loops = [_loop("a"), _loop("b"), _loop("c")]
    bottom_up = {"a": 0.2, "b": 0.9, "c": 0.5}
    res = TopDownBiasCombiner().apply(goal=None, loops=loops, bottom_up=bottom_up)

    assert all(ls.top_down_bias == 0.0 for ls in res.per_loop.values())
    assert res.winner_loop_id == "b"  # argmax bottom_up
    assert res.override is None
    assert res.effort_used == 0.0


def test_2_goal_flips_low_salience_loop_to_winner():
    # loop "hi" is the bottom-up winner; loop "goal" is low bottom-up but highly
    # relevant to a predictive goal.
    loops = [
        _loop("hi", predictive_value=0.0),
        _loop("goal", predictive_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(drive_origin="predictive", priority=0.9, goal_artifact_id="g1")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.winner_loop_id == "goal"
    assert res.override is not None
    assert res.override.beat_loop_id == "hi"
    assert res.override.chosen_loop_id == "goal"
    assert res.override.goal_artifact_id == "g1"
    assert res.override.goal_drive_origin == "predictive"


def test_3_effort_budget_exhausted_second_loop_gets_zero():
    loops = [
        _loop("top", predictive_value=1.0),
        _loop("second", predictive_value=0.9),
    ]
    bottom_up = {"top": 0.3, "second": 0.3}
    goal = GoalContext(drive_origin="predictive", priority=1.0)
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=0.2)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.per_loop["top"].applied_bias > 0.0
    assert res.per_loop["second"].applied_bias == 0.0
    # top-b loop's applied bias is capped by the small effort budget.
    assert abs(res.per_loop["top"].applied_bias - 0.2) < 1e-9


def test_4_strong_salience_beats_weak_goal_no_override():
    loops = [
        _loop("salient", predictive_value=0.0),
        _loop("weak_goal", predictive_value=0.2),
    ]
    bottom_up = {"salient": 0.95, "weak_goal": 0.1}
    # weak goal: low priority, small relevance -> small bias, cannot flip winner.
    goal = GoalContext(drive_origin="predictive", priority=0.2)
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.winner_loop_id == "salient"
    assert res.override is None


def test_5_relevance_mapping_table():
    goal_and_field = [
        ("predictive", "predictive_value"),
        ("relational", "relational_relevance"),
        ("continuity", "continuity_relevance"),
        ("autonomy", "autonomy_value"),
        ("coherence", "concept_value"),
        ("capability", "concept_value"),
    ]
    for drive_origin, field in goal_and_field:
        loop = _loop("x", **{field: 0.8})
        goal = GoalContext(drive_origin=drive_origin, priority=1.0)
        assert relevance(goal, loop) > 0.0, drive_origin
        # A loop with that field zero (and others zero) reads 0.
        zero_loop = _loop("y")
        assert relevance(goal, zero_loop) == 0.0, drive_origin

    # Unknown drive_origin falls back to concept_value.
    unknown_goal = GoalContext(drive_origin="banana", priority=1.0)
    loop = _loop("z", concept_value=0.7)
    assert relevance(unknown_goal, loop) == 0.7
    # ...and reads 0 when concept_value is 0 even if other fields are high.
    other_high = _loop("w", predictive_value=1.0)
    assert relevance(unknown_goal, other_high) == 0.0


def test_6_agency_gates_effort():
    loops = [
        _loop("hi", predictive_value=0.0),
        _loop("goal", predictive_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(drive_origin="predictive", priority=0.9)
    cfg = TopDownConfig(gain=0.6, effort_max=1.0, scale_by_agency=True)

    # agency_readiness=0 -> E=0 -> no bias applied, no override.
    res0 = TopDownBiasCombiner(cfg).apply(
        goal=goal, loops=loops, bottom_up=bottom_up, agency_readiness=0.0
    )
    assert res0.effort_used == 0.0
    assert all(ls.applied_bias == 0.0 for ls in res0.per_loop.values())
    assert res0.override is None
    assert res0.winner_loop_id == "hi"

    # agency_readiness=1 -> full effort -> override can fire.
    res1 = TopDownBiasCombiner(cfg).apply(
        goal=goal, loops=loops, bottom_up=bottom_up, agency_readiness=1.0
    )
    assert res1.override is not None
    assert res1.winner_loop_id == "goal"


def test_7_scores_clamped_to_unit_interval():
    # s=0.9 + gain*applied could exceed 1 -> must clamp.
    loops = [_loop("a", predictive_value=1.0), _loop("b", predictive_value=0.0)]
    bottom_up = {"a": 0.9, "b": 0.1}
    goal = GoalContext(drive_origin="predictive", priority=1.0)
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )
    assert _all_bounded(res)
    # 0.9 + 0.6*1.0 = 1.5 -> clamped to 1.0
    assert res.per_loop["a"].combined_salience == 1.0


def test_8_no_goal_combined_equals_bottom_up():
    loops = [_loop("a"), _loop("b"), _loop("c")]
    bottom_up = {"a": 0.2, "b": 0.9, "c": 0.5}
    res = TopDownBiasCombiner().apply(goal=None, loops=loops, bottom_up=bottom_up)
    for lid, s in bottom_up.items():
        assert res.per_loop[lid].combined_salience == s
    # missing-from-bottom_up loop -> 0.0
    loops2 = loops + [_loop("d")]
    res2 = TopDownBiasCombiner().apply(goal=None, loops=loops2, bottom_up=bottom_up)
    assert res2.per_loop["d"].combined_salience == 0.0


def test_9_override_roundtrips_through_pydantic():
    loops = [
        _loop("hi", predictive_value=0.0),
        _loop("goal", predictive_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(drive_origin="predictive", priority=0.9, goal_artifact_id="g1")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )
    assert res.override is not None
    restored = VoluntaryOverrideV1.model_validate(res.override.model_dump())
    assert restored == res.override


def test_never_raises_on_bad_input():
    # bottom_up missing keys, empty loops, weird agency -> no exception.
    res = TopDownBiasCombiner().apply(
        goal=GoalContext(drive_origin="predictive", priority=0.5),
        loops=[],
        bottom_up={},
        agency_readiness=5.0,
    )
    assert isinstance(res, TopDownResult)
    assert res.override is None
    assert res.winner_loop_id is None
