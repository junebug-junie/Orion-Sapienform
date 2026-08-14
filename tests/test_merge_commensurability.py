"""R3 of the phase-5 roadmap: detect a low-resolution input masking a
high-resolution one in a max() merge.

Every expected value is exact and hand-derived. No assertion compares against
a constant imported from the module under test -- that pattern is a tautology
and was how R2's first suite let 23 of 30 mutants through.
"""
from datetime import datetime, timezone

import pytest

from orion.field.commensurability import (
    VERDICT_ALL_TIED,
    VERDICT_CONTESTED,
    VERDICT_DOMINATED,
    VERDICT_INSUFFICIENT,
    VERDICT_WALKOVER,
    analyse_merge,
    observe_channel_to_dimension,
    observe_source_to_channel,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


def _obs(winner, **contributions):
    return (winner, dict(contributions))


def _series(n_coarse_wins, n_fine_wins, coarse_distinct, fine_distinct):
    """Build observations where `coarse` wins n_coarse_wins times and `fine`
    wins n_fine_wins, carrying exactly the requested distinct-value counts."""
    out = []
    total = n_coarse_wins + n_fine_wins
    for i in range(total):
        winner = "coarse" if i < n_coarse_wins else "fine"
        out.append(
            _obs(
                winner,
                coarse=round(0.9 + (i % coarse_distinct) * 0.001, 6),
                fine=round(0.1 + (i % fine_distinct) * 0.001, 6),
            )
        )
    return out


def test_a_coarse_winner_masking_a_fine_loser_is_the_finding() -> None:
    """The shape of both confirmed live instances.

    Hand-computed: 100 observations, no ties.
      coarse wins 60 -> win_share 60/100 = 0.60 (>= 0.5)
      coarse carries 2 distinct values; fine carries 50
      50 >= 2 * 10 -> fine is masked
    """
    result = analyse_merge("prediction_error", "source_to_channel",
                           _series(60, 40, coarse_distinct=2, fine_distinct=50))

    assert result.sample_count == 100
    assert result.tied_count == 0
    assert result.winner == "coarse"
    assert result.winner_share == pytest.approx(0.60)
    assert result.winner_distinct_values == 2
    assert [m.name for m in result.masked] == ["fine"]
    assert result.masked[0].distinct_values == 50
    assert result.masked[0].win_share == pytest.approx(0.40)
    assert result.verdict == VERDICT_DOMINATED
    assert result.is_finding is True


def test_the_most_informative_contender_winning_is_not_a_finding() -> None:
    """Same win split, resolutions swapped. A rich signal winning is exactly
    what should happen -- the detector must not fire on it.

    Hand-computed: coarse wins 60 with 50 distinct; fine carries 2 distinct.
      2 >= 50 * 10 is false -> nothing masked
    """
    result = analyse_merge("healthy", "source_to_channel",
                           _series(60, 40, coarse_distinct=50, fine_distinct=2))
    assert result.winner == "coarse"
    assert result.winner_distinct_values == 50
    assert result.masked == ()
    assert result.verdict == VERDICT_CONTESTED
    assert result.is_finding is False


def test_mask_ratio_boundary_is_inclusive_at_ten_times() -> None:
    """Pins MASK_RATIO at 10.0 and the comparison as `>=`, with a fixture on
    each side. Winner carries 3 distinct values.
      loser 30 distinct -> 30 >= 3*10 -> masked
      loser 29 distinct -> 29 >= 30 is false -> not masked
    """
    at = analyse_merge("m", "source_to_channel",
                       _series(60, 40, coarse_distinct=3, fine_distinct=30))
    assert at.winner_distinct_values == 3
    assert [m.distinct_values for m in at.masked] == [30]
    assert at.verdict == VERDICT_DOMINATED

    below = analyse_merge("m", "source_to_channel",
                          _series(60, 40, coarse_distinct=3, fine_distinct=29))
    assert below.masked == ()
    assert below.verdict == VERDICT_CONTESTED


def test_domination_share_boundary_is_inclusive_at_one_half() -> None:
    """Pins DOMINATION_SHARE at 0.5 and the comparison as `>=`.
      50 of 100 wins -> 0.50, exactly at the threshold -> dominated
      49 of 100     -> 0.49 -> contested, and `coarse` is not even top
    """
    at = analyse_merge("m", "source_to_channel",
                       _series(50, 50, coarse_distinct=2, fine_distinct=50))
    assert at.winner_share == pytest.approx(0.50)
    assert at.verdict == VERDICT_DOMINATED

    below = analyse_merge("m", "source_to_channel",
                          _series(49, 51, coarse_distinct=2, fine_distinct=50))
    assert below.winner == "fine"
    assert below.winner_share == pytest.approx(0.51)
    assert below.verdict == VERDICT_CONTESTED


def test_a_plurality_winner_below_the_threshold_is_not_domination() -> None:
    """Three contenders, so the top one can lead without a majority. This is
    the only shape that pins the LOWER side of DOMINATION_SHARE: the previous
    boundary fixture had no masked contender at a sub-majority share, so
    lowering the threshold to 0.2 changed nothing and survived mutation.

    Hand-computed: 100 observations, no ties.
      coarse wins 40 -> 0.40, the plurality but not a majority
      mid    wins 30, fine wins 30
      coarse carries 2 distinct values, fine carries 50 -> fine WOULD be
      masked (50 >= 2*10) if the share test passed, so the verdict turns
      entirely on 0.40 < 0.5.
    """
    observations = []
    for i in range(100):
        winner = "coarse" if i < 40 else ("mid" if i < 70 else "fine")
        observations.append(
            _obs(
                winner,
                coarse=round(0.9 + (i % 2) * 0.001, 6),
                mid=round(0.5 + (i % 5) * 0.001, 6),
                fine=round(0.1 + (i % 50) * 0.001, 6),
            )
        )
    result = analyse_merge("m", "source_to_channel", observations)

    assert result.winner == "coarse"
    assert result.winner_share == pytest.approx(0.40)
    assert result.winner_distinct_values == 2
    # A masked contender exists, so only the share test can hold the verdict.
    assert max(c.distinct_values for c in result.contenders if c.name != "coarse") == 50
    assert result.verdict == VERDICT_CONTESTED
    assert result.is_finding is False


def test_ties_are_excluded_from_the_win_share_not_from_the_information() -> None:
    """A tie names a winner by dict order. Counting those as wins is how a
    domination that never happened gets manufactured -- live, social_pressure
    is tied on 100% of ticks.

    Hand-computed: 100 observations, 60 tied. Only 40 are decided, and coarse
    wins 30 of them -> 30/40 = 0.75, NOT 30/100 = 0.30. The tied ticks still
    contribute their values to both contenders' distinct counts.
    """
    observations = []
    for i in range(60):
        observations.append(_obs(None, coarse=0.5, fine=round(0.1 + i * 0.001, 6)))
    for i in range(30):
        observations.append(_obs("coarse", coarse=0.9, fine=round(0.2 + i * 0.001, 6)))
    for i in range(10):
        observations.append(_obs("fine", coarse=0.1, fine=round(0.3 + i * 0.001, 6)))

    result = analyse_merge("m", "source_to_channel", observations)
    assert result.sample_count == 100
    assert result.tied_count == 60
    assert result.winner == "coarse"
    assert result.winner_share == pytest.approx(0.75)
    # coarse contributed exactly three distinct values: 0.5, 0.9, 0.1
    assert result.winner_distinct_values == 3
    # fine contributed 60 + 30 + 10 = 100 distinct values, all counted
    fine = next(c for c in result.contenders if c.name == "fine")
    assert fine.distinct_values == 100
    assert result.verdict == VERDICT_DOMINATED


def test_an_all_tied_merge_names_no_winner() -> None:
    observations = [_obs(None, a=0.5, b=0.5) for _ in range(50)]
    result = analyse_merge("m", "source_to_channel", observations)
    assert result.tied_count == 50
    assert result.winner is None
    assert result.verdict == VERDICT_ALL_TIED
    assert result.is_finding is False


def test_a_single_contender_is_a_walkover_not_a_defect() -> None:
    """6 of 7 dimensions are single-channel today. That is information, not a
    bug, and must never be reported as a finding."""
    observations = [_obs("only", only=round(0.5 + i * 0.001, 6)) for i in range(50)]
    result = analyse_merge("m", "channel_to_dimension", observations)
    assert result.winner == "only"
    assert result.winner_share == 1.0
    assert result.verdict == VERDICT_WALKOVER
    assert result.is_finding is False


def test_short_windows_refuse_a_verdict() -> None:
    """Pins MIN_MERGE_SAMPLES at 30: 29 observations refuse, 30 decide.
    A short window makes any contender look dominant."""
    short = analyse_merge("m", "source_to_channel",
                          _series(29, 0, coarse_distinct=2, fine_distinct=50))
    assert short.sample_count == 29
    assert short.verdict == VERDICT_INSUFFICIENT
    assert short.is_finding is False

    enough = analyse_merge("m", "source_to_channel",
                           _series(30, 0, coarse_distinct=2, fine_distinct=50))
    assert enough.sample_count == 30
    assert enough.verdict == VERDICT_DOMINATED


def test_empty_observations_refuse_rather_than_crash() -> None:
    result = analyse_merge("m", "source_to_channel", [])
    assert result.sample_count == 0
    assert result.contenders == ()
    assert result.winner is None
    assert result.verdict == VERDICT_INSUFFICIENT


# --------------------------------------------------------------------------
# observation builders, against real FieldStateV1 shapes
# --------------------------------------------------------------------------

def _state(node_vectors, capability_vectors=None):
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_r3",
        node_vectors=node_vectors,
        capability_vectors=capability_vectors or {},
    )


def test_source_merge_uses_max_for_a_pressure_channel() -> None:
    """Hand-computed: cpu_pressure 0.3 vs 0.7 -> node:b wins on max()."""
    obs = observe_source_to_channel(
        _state({"node:a": {"cpu_pressure": 0.3}, "node:b": {"cpu_pressure": 0.7}})
    )
    winner, contributions = obs["cpu_pressure"]
    assert winner == "node:b"
    assert contributions == {"node:a": 0.3, "node:b": 0.7}


def test_source_merge_uses_min_for_a_higher_is_better_channel() -> None:
    """`availability` is HIGHER_IS_BETTER, so the merge takes min() -- the
    WORST reading wins, because a healthy node must not mask a degraded one.
    Hand-computed: 0.9 vs 0.4 -> node:b wins."""
    obs = observe_source_to_channel(
        _state({"node:a": {"availability": 0.9}, "node:b": {"availability": 0.4}})
    )
    winner, contributions = obs["availability"]
    assert winner == "node:b"
    assert contributions == {"node:a": 0.9, "node:b": 0.4}


def test_source_merge_reports_a_tie_as_no_winner() -> None:
    obs = observe_source_to_channel(
        _state({"node:a": {"cpu_pressure": 0.5}, "node:b": {"cpu_pressure": 0.5}})
    )
    winner, contributions = obs["cpu_pressure"]
    assert winner is None
    assert len(contributions) == 2


def test_dimension_merge_reads_r1_provenance_including_the_tie_flag() -> None:
    """Hand-computed: repair_pressure 0.4 and conversation_load 0.9 both map to
    social_pressure -> conversation_load wins, uniquely."""
    obs = observe_channel_to_dimension(
        _state({"node:a": {"repair_pressure": 0.4, "conversation_load": 0.9}})
    )
    winner, contributions = obs["social_pressure"]
    assert winner == "conversation_load"
    assert contributions == {"repair_pressure": 0.4, "conversation_load": 0.9}

    # Equal values -> R1 marks the winner non-unique -> no winner here.
    tied = observe_channel_to_dimension(
        _state({"node:a": {"repair_pressure": 0.7, "conversation_load": 0.7}})
    )
    tied_winner, _ = tied["social_pressure"]
    assert tied_winner is None
