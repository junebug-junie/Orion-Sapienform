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


def _loop(id: str, target: str | None = None, **fields: float) -> OpenLoopV1:
    """Construct an OpenLoopV1; ``target`` sets the source node relevance joins on.

    relevance() matches ``GoalContext.target_id`` against ``source_refs`` (both
    hold substrate node ids). The legacy ``concept_value``/``predictive_value``
    style kwargs still pass through so tests can prove they are IGNORED.
    """
    refs = [target] if target else []
    return OpenLoopV1(id=id, description="d", source_refs=refs, **fields)


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
    # the goal's own target (source_refs matches GoalContext.target_id).
    loops = [
        _loop("hi"),
        _loop("goal", target="node:wanted"),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9, goal_artifact_id="g1", target_id="node:wanted")
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
        _loop("top", target="node:same"),
        _loop("second", target="node:same"),
    ]
    # Both loops match the goal, so relevance -- and therefore bias -- is 1.0
    # for both; the effort budget, not relevance, is what must starve the
    # second one. Rule 4 orders by (-bias, -bottom_up, id), so with bias tied
    # the bottom_up values below are what make "top" processed first. They are
    # deliberately unequal: leaving them tied would decide the order by id
    # string alone, which is an accident, not the behavior under test.
    bottom_up = {"top": 0.4, "second": 0.3}
    goal = GoalContext(priority=1.0, target_id="node:same")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=0.2)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.per_loop["top"].applied_bias > 0.0
    assert res.per_loop["second"].applied_bias == 0.0
    # top-b loop's applied bias is capped by the small effort budget.
    assert abs(res.per_loop["top"].applied_bias - 0.2) < 1e-9


def test_4_strong_salience_beats_weak_goal_no_override():
    loops = [
        _loop("salient"),
        _loop("weak_goal", target="node:wanted"),
    ]
    bottom_up = {"salient": 0.95, "weak_goal": 0.1}
    # weak goal: relevant, but low priority -> small bias, cannot flip winner.
    goal = GoalContext(priority=0.2, target_id="node:wanted")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=1.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )

    assert res.winner_loop_id == "salient"
    assert res.override is None


def test_5_relevance_joins_goal_target_to_loop_source_refs():
    """relevance() reads the GOAL, and ignores the legacy score fields.

    Before 2026-09-05 the body was ``return _clamp01(loop.concept_value)`` --
    it took ``goal`` and never read it, so it could not discriminate between
    goals even in principle. These assertions fail against that version.
    """
    goal = GoalContext(priority=1.0, target_id="node:wanted")

    match = _loop("x", target="node:wanted")
    assert relevance(goal, match) == 1.0

    other = _loop("y", target="node:something_else")
    assert relevance(goal, other) == 0.0

    # A loop with no source at all is never relevant.
    assert relevance(goal, _loop("z")) == 0.0

    # A goal that names no target pushes on NOTHING -- not on everything.
    # This is the deliberate "push on nothing" choice; the old constant-0.55
    # behavior is what made override impossible.
    assert relevance(GoalContext(priority=1.0), match) == 0.0


def test_5b_legacy_score_fields_are_ignored():
    """The five legacy relevance fields must not influence bias any more.

    ``concept_value`` in particular was the old input, and it was fabricated:
    scoring.py floored it to a constant 0.55 for every substrate loop.
    """
    goal = GoalContext(priority=1.0, target_id="node:wanted")
    loud_but_irrelevant = _loop(
        "x",
        target="node:other",
        concept_value=1.0,
        predictive_value=1.0,
        relational_relevance=1.0,
        continuity_relevance=1.0,
        autonomy_value=1.0,
    )
    assert relevance(goal, loud_but_irrelevant) == 0.0

    quiet_but_relevant = _loop("y", target="node:wanted", concept_value=0.0)
    assert relevance(goal, quiet_but_relevant) == 1.0


def test_5c_uniform_relevance_cannot_flip_a_winner():
    """The exact defect this patch fixes, pinned as a regression test.

    When every candidate scores the same relevance, bias is identical across
    loops, ``combined = salience + gain*applied`` shifts them all equally, and
    the bottom-up winner always survives. That was production for the entire
    life of the feature (proved live: 5 real loops, priority 0.1..1.0, override
    fired 0/6). If a future change reintroduces a constant relevance, this test
    is what should catch it.
    """
    loops = [_loop("loud", target="node:same"), _loop("quiet", target="node:same")]
    bottom_up = {"loud": 0.8, "quiet": 0.2}
    goal = GoalContext(priority=1.0, target_id="node:same")
    res = TopDownBiasCombiner(TopDownConfig(gain=0.6, effort_max=10.0)).apply(
        goal=goal, loops=loops, bottom_up=bottom_up
    )
    biases = {ls.top_down_bias for ls in res.per_loop.values()}
    assert biases == {1.0}, "both loops match the goal -> identical bias"
    assert res.winner_loop_id == "loud"
    assert res.override is None


def test_6_agency_gates_effort():
    loops = [
        _loop("hi"),
        _loop("goal", target="node:wanted"),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9, target_id="node:wanted")
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
    loops = [_loop("a", target="node:wanted"), _loop("b")]
    bottom_up = {"a": 0.9, "b": 0.1}
    goal = GoalContext(priority=1.0, target_id="node:wanted")
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
        _loop("hi"),
        _loop("goal", target="node:wanted"),
    ]
    bottom_up = {"hi": 0.6, "goal": 0.1}
    goal = GoalContext(priority=0.9, goal_artifact_id="g1", target_id="node:wanted")
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
