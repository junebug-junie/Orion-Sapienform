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
    # relevant (concept_value) to the active goal.
    loops = [
        _loop("hi", concept_value=0.0),
        _loop("goal", concept_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9, goal_artifact_id="g1")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.winner_loop_id == "goal"
    assert res.override is not None
    assert res.override.beat_loop_id == "hi"
    assert res.override.chosen_loop_id == "goal"
    assert res.override.goal_artifact_id == "g1"
    # drive_origin was removed from GoalContext (Wave 2b) -- the schema field
    # survives on VoluntaryOverrideV1 for other consumers but this producer no
    # longer has a value to set it with, so it stays at its schema default.
    assert res.override.goal_drive_origin is None


def test_3_effort_budget_exhausted_second_loop_gets_zero():
    loops = [
        _loop("top", concept_value=1.0),
        _loop("second", concept_value=0.9),
    ]
    bottom_up = {"top": 0.3, "second": 0.3}
    goal = GoalContext(priority=1.0)
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=0.2)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.per_loop["top"].applied_bias > 0.0
    assert res.per_loop["second"].applied_bias == 0.0
    # top-b loop's applied bias is capped by the small effort budget.
    assert abs(res.per_loop["top"].applied_bias - 0.2) < 1e-9


def test_4_strong_salience_beats_weak_goal_no_override():
    loops = [
        _loop("salient", concept_value=0.0),
        _loop("weak_goal", concept_value=0.2),
    ]
    bottom_up = {"salient": 0.95, "weak_goal": 0.1}
    # weak goal: low priority, small relevance -> small bias, cannot flip winner.
    goal = GoalContext(priority=0.2)
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.winner_loop_id == "salient"
    assert res.override is None


def test_5_relevance_reads_concept_value_only():
    # Wave 2b removed the drive_origin -> relevance-field mapping table
    # (predictive_value/relational_relevance/continuity_relevance/
    # autonomy_value/concept_value keyed by drive_origin). relevance() now
    # always reads concept_value -- the one dimension that was already the
    # fallback for unmapped/unknown drive origins -- and ignores the other
    # four legacy fields entirely, even when they're populated.
    goal = GoalContext(priority=1.0)
    loop = _loop(
        "x",
        concept_value=0.8,
        predictive_value=1.0,
        relational_relevance=1.0,
        continuity_relevance=1.0,
        autonomy_value=1.0,
    )
    assert relevance(goal, loop) == 0.8

    zero_concept_loop = _loop(
        "y",
        predictive_value=1.0,
        relational_relevance=1.0,
        continuity_relevance=1.0,
        autonomy_value=1.0,
    )
    assert relevance(goal, zero_concept_loop) == 0.0


def test_6_agency_gates_effort():
    loops = [
        _loop("hi", concept_value=0.0),
        _loop("goal", concept_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9)
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
    loops = [_loop("a", concept_value=1.0), _loop("b", concept_value=0.0)]
    bottom_up = {"a": 0.9, "b": 0.1}
    goal = GoalContext(priority=1.0)
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
        _loop("hi", concept_value=0.0),
        _loop("goal", concept_value=1.0),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9, goal_artifact_id="g1")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )
    assert res.override is not None
    restored = VoluntaryOverrideV1.model_validate(res.override.model_dump())
    assert restored == res.override


def test_never_raises_on_bad_input():
    # bottom_up missing keys, empty loops, weird agency -> no exception.
    res = TopDownBiasCombiner().apply(
        goal=GoalContext(priority=0.5),
        loops=[],
        bottom_up={},
        agency_readiness=5.0,
    )
    assert isinstance(res, TopDownResult)
    assert res.override is None
    assert res.winner_loop_id is None
