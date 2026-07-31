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
        observed_at=NOW, fallback_target_id="tick:abc",
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
        None, series_name="x", observed_at=NOW, fallback_target_id="t"
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


def test_confidence_tracks_baseline_size_not_excursion_size() -> None:
    """A 3-sigma reading off 20 samples is less trustworthy than the same
    reading off 200 -- same distinction the arena's own proposal_confidence()
    makes for native candidates."""
    thin = _candidate(sustained=True, z=3.0, count=20)
    thick = _candidate(sustained=True, z=3.0, count=200)
    assert thin.confidence_score < thick.confidence_score
    assert thick.confidence_score == 1.0


def test_chain_lineage_is_present_and_marks_hop_zero() -> None:
    c = _candidate(sustained=True)
    reasons = " ".join(c.reasons)
    assert "cognitive_hop" in reasons
    assert "hop_index:0" in reasons
    assert "parent_hop_id:none" in reasons
    assert any(r.startswith("chain_id:chain:repair_pressure:") for r in c.reasons)


def test_chain_id_is_deterministic_for_replay() -> None:
    """The repo's replay convention reconstructs historical state from stored
    events; a uuid4 chain id would make the same trend firing a different chain
    on every replay."""
    a = _candidate(sustained=True)
    b = _candidate(sustained=True)
    assert a.proposal_id == b.proposal_id

    later = trend_result_to_candidate(
        _result(sustained=True), series_name="repair_pressure",
        observed_at=NOW + timedelta(seconds=1), fallback_target_id="tick:abc",
    )
    assert later.proposal_id != a.proposal_id


def test_end_to_end_from_raw_readings_through_the_reducer() -> None:
    """Drives the real reducer, not a hand-built result: a calm baseline
    followed by a sustained excursion must produce exactly one candidate."""
    calm = [
        MetacogTrendReading(at=NOW + timedelta(minutes=i), level=0.087, confidence=0.65)
        for i in range(30)
    ]
    assert trend_result_to_candidate(
        replay(calm)[-1], series_name="repair_pressure",
        observed_at=calm[-1].at, fallback_target_id="t",
    ) is None, "a flat calm series is not a trend"

    # A RISING series, not a step: measured above, a step spikes once and then
    # normalizes as the EWMA adopts it. Escalation is what sustains.
    elevated = calm + [
        MetacogTrendReading(at=NOW + timedelta(minutes=30 + i), level=level, confidence=0.65)
        for i, level in enumerate([0.15, 0.25, 0.38, 0.52, 0.68])
    ]
    c = trend_result_to_candidate(
        replay(elevated)[-1], series_name="repair_pressure",
        observed_at=elevated[-1].at, fallback_target_id="t",
    )
    assert c is not None, "a sustained real excursion must reach the arena"
    assert c.source == COGNITIVE_HOP_PROPOSAL_SOURCE
