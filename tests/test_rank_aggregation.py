"""Direct-import smoke tests for `orion.attention.rank_aggregation`.

Extracted 2026-07-31 from `candidate_society_of_mind.py` (whose own
`tests/test_attention_candidate_society_of_mind.py` continues to exercise
this exact code via re-export -- see that file). These tests import the
shared module directly so a regression here is caught even if a future
change stops re-exporting from the field-attention module.
"""

from __future__ import annotations

from orion.attention.rank_aggregation import (
    BordaResult,
    aggregate_borda,
    scorer_top1,
)


def test_scorer_top1_empty():
    assert scorer_top1({}) is None


def test_scorer_top1_deterministic_tiebreak():
    assert scorer_top1({"b": 0.5, "a": 0.5, "c": 0.1}) == "a"


def test_aggregate_borda_two_scorers_clean_winner():
    result = aggregate_borda(
        {
            "strength": {"x": 0.9, "y": 0.2},
            "breadth": {"x": 0.8, "y": 0.1},
        }
    )
    assert isinstance(result, BordaResult)
    assert result.winner == "x"
    assert result.disagreement is False


def test_aggregate_borda_partial_ballot_treated_as_last():
    """A scorer with no opinion about a target ranks it tied-last for that
    scorer, never favored over a target it actually scored."""
    result = aggregate_borda(
        {
            "strength": {"x": 0.9},
            "breadth": {"y": 0.9},
        },
        universe=["x", "y"],
    )
    # Each scorer only voted for one target; the other is tied-last for it.
    # Both x and y get exactly one scorer's top vote (1.0) + one scorer's
    # absent vote (0.0) -> tied overall.
    assert result.totals["x"] == result.totals["y"]


def test_aggregate_borda_single_candidate_scores_zero():
    """With exactly one candidate, Borda has nothing to rank against --
    every scorer's point allocation for n=1 is 0.0 (see
    `_borda_points_for_scorer`'s n==1 special case)."""
    result = aggregate_borda({"strength": {"solo": 0.95}})
    assert result.totals == {"solo": 0.0}
    assert result.winner == "solo"


def test_aggregate_borda_disagreement_flag():
    result = aggregate_borda(
        {
            "strength": {"x": 0.9, "y": 0.1},
            "breadth": {"x": 0.1, "y": 0.9},
        }
    )
    assert result.disagreement is True
