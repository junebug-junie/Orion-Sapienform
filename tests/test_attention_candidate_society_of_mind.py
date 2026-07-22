"""Deterministic unit tests for Candidate B's rank-aggregation shadow module
(`orion/attention/field_attention/candidate_society_of_mind.py`).

No DB, no network, no bus. Pure-function tests only: the three scorer
functions (magnitude/novelty/dwell) and the Borda-count combiner, exercised
against synthetic fixtures for clean-rank, tie-handling, partial-ballot, and
disagreement scenarios per this candidate's own task brief.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.attention.field_attention.candidate_society_of_mind import (
    DWELL_TICKS_SATURATION,
    BordaResult,
    aggregate_borda,
    dwell_scorer,
    magnitude_scorer,
    novelty_scorer,
    scorer_top1,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1

UTC = timezone.utc
BASE = datetime(2026, 7, 21, 0, 0, 0, tzinfo=UTC)


def _target(target_id: str, salience: float, kind: str = "node") -> FieldAttentionTargetV1:
    return FieldAttentionTargetV1(
        target_id=target_id,
        target_kind=kind,
        salience_score=salience,
        pressure_score=salience,
        novelty_score=0.0,
        urgency_score=0.0,
        confidence_score=0.0,
    )


def _frame(targets: list[FieldAttentionTargetV1]) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:test:policy.v1",
        generated_at=BASE,
        source_field_tick_id="tick:test",
        source_field_generated_at=BASE,
        overall_salience=max((t.salience_score for t in targets), default=0.0),
        dominant_targets=targets,
        node_targets=[t for t in targets if t.target_kind == "node"],
        capability_targets=[t for t in targets if t.target_kind == "capability"],
    )


# ===========================================================================
# magnitude_scorer -- real passthrough/validation, no I/O.
# ===========================================================================


def test_magnitude_scorer_passthrough() -> None:
    result = magnitude_scorer({"node:substrate.biometrics": 0.42, "node:substrate.execution": 0.0})
    assert result == {"node:substrate.biometrics": 0.42, "node:substrate.execution": 0.0}


def test_magnitude_scorer_clamps_out_of_range() -> None:
    result = magnitude_scorer({"node:substrate.biometrics": 1.5, "node:substrate.execution": -0.3})
    assert result == {"node:substrate.biometrics": 1.0, "node:substrate.execution": 0.0}


def test_magnitude_scorer_drops_non_finite_and_non_numeric() -> None:
    result = magnitude_scorer(
        {
            "node:substrate.biometrics": float("nan"),
            "node:substrate.execution": float("inf"),
            "node:substrate.transport": "garbage",  # type: ignore[dict-item]
            "node:substrate.chat": 0.3,
        }
    )
    assert result == {"node:substrate.chat": 0.3}


def test_magnitude_scorer_empty_input() -> None:
    assert magnitude_scorer({}) == {}


# ===========================================================================
# novelty_scorer -- thin wrapper around the already-live novelty_for_target().
# ===========================================================================


def test_novelty_scorer_diffs_against_previous_frame() -> None:
    previous = _frame([_target("node:athena", 0.2)])
    result = novelty_scorer(["node:athena"], {"node:athena": 0.9}, previous)
    assert result["node:athena"] == pytest.approx(0.7)


def test_novelty_scorer_no_previous_frame_is_zero() -> None:
    result = novelty_scorer(["node:athena"], {"node:athena": 0.9}, None)
    assert result == {"node:athena": 0.0}


def test_novelty_scorer_missing_current_salience_defaults_zero() -> None:
    previous = _frame([_target("node:athena", 0.5)])
    result = novelty_scorer(["node:athena"], {}, previous)
    assert result["node:athena"] == pytest.approx(0.5)


def test_novelty_scorer_target_absent_from_previous_frame_is_zero_prior() -> None:
    previous = _frame([_target("node:athena", 0.5)])
    result = novelty_scorer(["node:atlas"], {"node:atlas": 0.3}, previous)
    assert result["node:atlas"] == pytest.approx(0.3)


# ===========================================================================
# dwell_scorer -- real duration signal, disclosed live-degeneracy caveat.
# ===========================================================================


def test_dwell_scorer_scores_all_attended_targets_equally() -> None:
    result = dwell_scorer(["node:substrate.harness_closure", "node:athena"], dwell_ticks=60)
    expected = min(1.0, 60.0 / DWELL_TICKS_SATURATION)
    assert result == {
        "node:substrate.harness_closure": pytest.approx(expected),
        "node:athena": pytest.approx(expected),
    }


def test_dwell_scorer_saturates_at_cap() -> None:
    result = dwell_scorer(["node:athena"], dwell_ticks=4899, saturation_ticks=120)
    assert result == {"node:athena": 1.0}


def test_dwell_scorer_empty_coalition_returns_empty_dict() -> None:
    """The real, load-bearing 2026-07-21 finding: attended_node_ids is []
    in 99.9% of live substrate_coalition_dwell_log rows. This must return an
    empty ballot (no opinion), not a score for a nonexistent target."""
    assert dwell_scorer([], dwell_ticks=183) == {}


def test_dwell_scorer_zero_or_negative_ticks_returns_empty_dict() -> None:
    assert dwell_scorer(["node:athena"], dwell_ticks=0) == {}
    assert dwell_scorer(["node:athena"], dwell_ticks=-1) == {}


def test_dwell_scorer_deduplicates_attended_ids() -> None:
    result = dwell_scorer(["node:athena", "node:athena", "node:atlas"], dwell_ticks=30)
    assert set(result.keys()) == {"node:athena", "node:atlas"}
    assert result["node:athena"] == result["node:atlas"]


# ===========================================================================
# scorer_top1 -- deterministic tie-break helper.
# ===========================================================================


def test_scorer_top1_highest_score_wins() -> None:
    assert scorer_top1({"a": 0.2, "b": 0.9, "c": 0.5}) == "b"


def test_scorer_top1_tie_break_alphabetical() -> None:
    assert scorer_top1({"z": 0.5, "a": 0.5}) == "a"


def test_scorer_top1_empty_is_none() -> None:
    assert scorer_top1({}) is None


# ===========================================================================
# aggregate_borda -- the rank-aggregation combiner itself.
# ===========================================================================


def test_aggregate_borda_clean_rank_all_scorers_agree() -> None:
    """3 targets, all 3 scorers rank them identically A > B > C -- classic
    Borda: A gets 3*2=6, B gets 3*1=3, C gets 3*0=0."""
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    result = aggregate_borda({"magnitude": scores, "novelty": scores, "dwell": scores})
    assert result.totals == {"a": 6.0, "b": 3.0, "c": 0.0}
    assert result.ranking == ("a", "b", "c")
    assert result.winner == "a"
    assert result.disagreement is False


def test_aggregate_borda_tie_within_one_scorer_shares_average_points() -> None:
    """b and c tie for 2nd/3rd place (positions 0 and 1 in ascending order,
    each worth 0 and 1 point) -> both get the average, 0.5."""
    scores = {"a": 0.9, "b": 0.4, "c": 0.4}
    result = aggregate_borda({"only": scores})
    assert result.totals["a"] == 2.0
    assert result.totals["b"] == pytest.approx(0.5)
    assert result.totals["c"] == pytest.approx(0.5)


def test_aggregate_borda_disagreement_detected_with_real_examples() -> None:
    """Each scorer independently prefers a different target -- the direct
    test for whether rank-aggregation ever disagrees, per the task brief."""
    magnitude = {"a": 0.9, "b": 0.1, "c": 0.1}
    novelty = {"a": 0.1, "b": 0.9, "c": 0.1}
    dwell = {"a": 0.1, "b": 0.1, "c": 0.9}
    result = aggregate_borda({"magnitude": magnitude, "novelty": novelty, "dwell": dwell})
    assert result.per_scorer_top1 == {"magnitude": "a", "novelty": "b", "dwell": "c"}
    assert result.disagreement is True
    # Fully symmetric 3-way disagreement: each scorer's own bottom two targets
    # tie at 0.1 (average of positions 0/1 = 0.5 each) and its top pick gets
    # 2 -- so every target ends up with the same total (2 + 0.5 + 0.5 = 3.0)
    # -- the deterministic target_id tiebreak decides the reported winner.
    assert result.totals == {"a": 3.0, "b": 3.0, "c": 3.0}
    assert result.winner == "a"


def test_aggregate_borda_partial_ballot_absent_target_treated_as_last() -> None:
    """`dwell` only has an opinion about "a" (the rest of the tick's real
    targets never entered the coalition) -- "b"/"c" must be tied for dwell's
    own last place, never silently excluded or averaged favorably."""
    magnitude = {"a": 0.5, "b": 0.9, "c": 0.1}
    dwell = {"a": 0.8}  # only "a" was ever in a real coalition this tick
    result = aggregate_borda({"magnitude": magnitude, "dwell": dwell})
    # magnitude alone: b=2, a=1, c=0. dwell alone: a=2, {b,c} tied last -> 0.5 each.
    assert result.totals == {"a": 3.0, "b": 2.5, "c": 0.5}
    assert result.ranking == ("a", "b", "c")


def test_aggregate_borda_scorer_with_empty_ballot_does_not_affect_ranking_or_disagreement() -> None:
    """A scorer with an entirely empty ballot still nominally casts a
    "everyone tied last" vote (both targets get the same averaged points
    added), which shifts absolute totals uniformly but never the relative
    *ranking* -- "a" still wins, and an empty ballot never registers a top1
    pick, so it cannot contribute to `disagreement` either."""
    magnitude = {"a": 0.9, "b": 0.1}
    empty_dwell: dict[str, float] = {}
    result = aggregate_borda({"magnitude": magnitude, "dwell": empty_dwell})
    assert result.totals == {"a": 1.5, "b": 0.5}
    assert result.ranking == ("a", "b")
    assert result.per_scorer_top1 == {"magnitude": "a", "dwell": None}
    assert result.disagreement is False


def test_aggregate_borda_explicit_universe_includes_unscored_target() -> None:
    """"z" is in the requested universe but no scorer ever mentioned it --
    it sorts below every real (even low) score, not merely tied with the
    scorer's own lowest real pick: -inf < 0.1, so "z" is strictly last."""
    magnitude = {"a": 0.9, "b": 0.1}
    result = aggregate_borda({"magnitude": magnitude}, universe=["a", "b", "z"])
    assert result.universe == ("a", "b", "z")
    assert result.totals == {"a": 2.0, "b": 1.0, "z": 0.0}
    assert result.ranking == ("a", "b", "z")


def test_aggregate_borda_no_scorers_empty_universe() -> None:
    result = aggregate_borda({})
    assert result == BordaResult(
        universe=(), totals={}, ranking=(), winner=None, per_scorer_top1={}, disagreement=False
    )


def test_aggregate_borda_single_target_universe_gets_zero_points() -> None:
    result = aggregate_borda({"magnitude": {"a": 0.7}})
    assert result.totals == {"a": 0.0}
    assert result.winner == "a"
