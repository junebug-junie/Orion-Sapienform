"""Hop 0: sustained metacog trend -> governed arena candidate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.metacog.proposal import (
    COGNITIVE_HOP_PROPOSAL_SOURCE,
    trend_result_to_candidate,
)
from orion.metacog.trend_reducer import (
    MetacogTrendReading,
    MetacogTrendResultV1,
    MetacogTrendStateV1,
    replay,
)

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


def _result(*, sustained: bool, cold: bool = False, z: float | None = 2.0,
            consecutive: int = 3, count: int = 40) -> MetacogTrendResultV1:
    return MetacogTrendResultV1(
        state=MetacogTrendStateV1(
            ewma=0.1, variance=0.02, count=count, consecutive_elevated=consecutive
        ),
        latest_zscore=z,
        is_cold_start=cold,
        is_elevated_this_tick=sustained,
        is_sustained_trend=sustained,
    )


def _candidate(**kw):
    return trend_result_to_candidate(
        _result(**kw), series_name="repair_pressure",
        run_started_at=NOW, fallback_target_id="tick:abc",
    )


def test_a_sustained_trend_becomes_a_candidate() -> None:
    c = _candidate(sustained=True)
    assert c is not None
    assert c.source == COGNITIVE_HOP_PROPOSAL_SOURCE
    assert "consecutive" in c.description
    # "This is a trend, not a spike" is the whole point of hop 0.
    assert "trend, not a spike" in c.description


def test_a_single_spike_is_not_a_train_of_thought() -> None:
    """`is_sustained_trend` is the reducer's "has this kept happening" verdict.
    One elevated tick must not manufacture a chain -- that would make hop 0 a
    point-in-time trigger, which is exactly what this arc exists to replace."""
    assert _candidate(sustained=False) is None


def test_cold_start_never_proposes() -> None:
    assert _candidate(sustained=True, cold=True) is None
    assert _candidate(sustained=True, z=None) is None


def test_none_result_degrades_rather_than_raising() -> None:
    assert trend_result_to_candidate(
        None, series_name="x", run_started_at=NOW, fallback_target_id="t"
    ) is None


def test_candidate_can_never_auto_dispatch() -> None:
    """The hop-chain design's named danger case is a chain that always
    self-scores above the preemption threshold and is never actually
    interrupted. An unconditional operator_review gate means the worst case is
    a noisy proposal queue, not autonomous action."""
    c = _candidate(sustained=True, z=99.0)  # maximally salient
    assert c.required_policy_gate == "operator_review"
    assert c.priority_score <= 1.0


def test_urgency_stays_below_priority_because_a_trend_is_not_a_spike() -> None:
    """A sustained background condition must not outrank a genuine acute one in
    the arena."""
    c = _candidate(sustained=True)
    assert c.urgency_score < c.priority_score


def test_priority_scales_with_run_length_not_raw_zscore() -> None:
    """Measured: a flat input run collapses the EWMA variance toward its 1e-6
    floor, so the first excursion after one scores z=463 -- and the real
    repair_pressure series is exactly that shape. A z-proportional salience
    would put hop 0 in the arena at priority 1.0 every time it fired, which is
    the design doc's named danger case."""
    huge_z_short_run = _candidate(sustained=True, z=463.0, consecutive=3)
    modest_z_long_run = _candidate(sustained=True, z=2.0, consecutive=6)
    assert huge_z_short_run.priority_score < modest_z_long_run.priority_score
    assert huge_z_short_run.priority_score == pytest.approx(0.5)
    assert modest_z_long_run.priority_score == pytest.approx(1.0)
    # The z-score is still on the record, just not driving the score.
    assert any(r.startswith("zscore:463") for r in huge_z_short_run.reasons)


def test_confidence_is_inverted_from_z_and_can_fall() -> None:
    """Matches the arena's native dimension_confidence() semantic: confidence is
    "how well does this match its own baseline", not "how sure am I there is a
    trend". Critically it must be able to move BOTH ways -- a monotonically
    growing confidence is an always-saturated metric."""
    near_baseline = _candidate(sustained=True, z=0.3)
    far_from_baseline = _candidate(sustained=True, z=3.0)
    assert near_baseline.confidence_score > far_from_baseline.confidence_score
    # Must be able to FALL, and to reach both ends -- the first version used
    # state.count/100, which only grows and pinned at 1.0 permanently after
    # ~100 rows (~4.5 days at this series' rate).
    assert far_from_baseline.confidence_score == 0.0
    assert _candidate(sustained=True, z=0.0).confidence_score == 1.0


def test_chain_lineage_is_present_and_marks_hop_zero() -> None:
    c = _candidate(sustained=True)
    reasons = " ".join(c.reasons)
    assert "cognitive_hop" in reasons
    assert "hop_index:0" in reasons
    assert "parent_hop_id:none" in reasons
    assert any(r.startswith("chain_id:chain:repair_pressure:") for r in c.reasons)


def test_chain_id_is_stable_across_one_continuous_trend() -> None:
    """The hop-chain design requires chain_id "stable across a chain's life" so
    an eventual outcome can be attributed back to the observation that started
    the chain. The first version keyed on the LATEST reading, so one continuous
    escalating trend minted a new identity every tick (consec 3/4/5/6 gave 4
    distinct ids) -- making one chain look like N unrelated proposals to
    feedback/consolidation, and leaving hop 1 nothing stable to point
    parent_hop_id at."""
    ids = {
        trend_result_to_candidate(
            _result(sustained=True, consecutive=c), series_name="repair_pressure",
            run_started_at=NOW, fallback_target_id="t",
        ).proposal_id
        for c in (3, 4, 5, 6)
    }
    assert len(ids) == 1, f"one trend must be one chain, got {len(ids)}"


def test_chain_id_is_deterministic_for_replay() -> None:
    """Still deterministic: the onset is a stored timestamp, not wall-clock, so
    reconstructing from stored rows reproduces the same id."""
    assert _candidate(sustained=True).proposal_id == _candidate(sustained=True).proposal_id

    different_run = trend_result_to_candidate(
        _result(sustained=True), series_name="repair_pressure",
        run_started_at=NOW + timedelta(hours=3), fallback_target_id="tick:abc",
    )
    assert different_run.proposal_id != _candidate(sustained=True).proposal_id


def test_reducer_detects_escalation_not_mere_elevation() -> None:
    """Pins the measured characterization that was previously comment-only.

    A value that jumps high and HOLDS produces exactly one sustained tick,
    because the EWMA baseline adopts the new level as normal within a few ticks.
    A rising ramp sustains. This is the most consequential property of hop 0 --
    it will not open a chain about a condition that has been quietly bad for an
    hour -- so it must fail loudly if the reducer is ever re-shaped."""
    def series(levels):
        return [
            MetacogTrendReading(at=NOW + timedelta(minutes=i), level=v, confidence=0.65)
            for i, v in enumerate(levels)
        ]

    calm = [0.087] * 30
    step_hold = replay(series(calm + [0.55] * 12))
    ramp = replay(series(calm + [0.15, 0.25, 0.38, 0.52, 0.68]))

    assert sum(1 for r in step_hold if r.is_sustained_trend) == 1, "a held step is not a trend"
    assert sum(1 for r in ramp if r.is_sustained_trend) >= 3, "escalation must sustain"


def test_end_to_end_from_raw_readings_through_the_reducer() -> None:
    """Drives the real reducer, not a hand-built result: a calm baseline
    followed by a sustained excursion must produce exactly one candidate."""
    calm = [
        MetacogTrendReading(at=NOW + timedelta(minutes=i), level=0.087, confidence=0.65)
        for i in range(30)
    ]
    assert trend_result_to_candidate(
        replay(calm)[-1], series_name="repair_pressure",
        run_started_at=calm[-1].at, fallback_target_id="t",
    ) is None, "a flat calm series is not a trend"

    # A RISING series, not a step: measured above, a step spikes once and then
    # normalizes as the EWMA adopts it. Escalation is what sustains.
    elevated = calm + [
        MetacogTrendReading(at=NOW + timedelta(minutes=30 + i), level=level, confidence=0.65)
        for i, level in enumerate([0.15, 0.25, 0.38, 0.52, 0.68])
    ]
    c = trend_result_to_candidate(
        replay(elevated)[-1], series_name="repair_pressure",
        run_started_at=elevated[-1].at, fallback_target_id="t",
    )
    assert c is not None, "a sustained real excursion must reach the arena"
    assert c.source == COGNITIVE_HOP_PROPOSAL_SOURCE
