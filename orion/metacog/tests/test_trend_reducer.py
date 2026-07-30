from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.metacog.trend_reducer import (
    MetacogTrendReading,
    MetacogTrendStateV1,
    apply_reading,
    replay,
)


def _reading(i: int, level: float, confidence: float = 0.65) -> MetacogTrendReading:
    return MetacogTrendReading(
        at=datetime(2026, 7, 24, tzinfo=timezone.utc) + timedelta(minutes=i),
        level=level,
        confidence=confidence,
    )


def test_first_reading_is_cold_start_with_no_zscore() -> None:
    state = MetacogTrendStateV1()
    result = apply_reading(state, _reading(0, 0.3))
    assert result.is_cold_start is True
    assert result.latest_zscore is None
    assert result.is_elevated_this_tick is False
    assert result.state.count == 1


def test_stays_cold_start_below_min_samples() -> None:
    readings = [_reading(i, 0.3) for i in range(10)]
    results = replay(readings, min_samples=20)
    assert all(r.is_cold_start for r in results)
    assert all(not r.is_elevated_this_tick for r in results)


def test_flat_baseline_never_reads_elevated() -> None:
    # A perfectly constant series should never trip elevated, even past
    # the cold-start window -- there is no real deviation to be anomalous
    # against.
    readings = [_reading(i, 0.2) for i in range(50)]
    results = replay(readings, min_samples=20)
    warmed_up = [r for r in results if not r.is_cold_start]
    assert len(warmed_up) > 0
    assert all(not r.is_elevated_this_tick for r in warmed_up)
    assert all(not r.is_sustained_trend for r in warmed_up)


def test_real_spike_after_warmup_reads_elevated() -> None:
    baseline = [_reading(i, 0.15 + (0.01 if i % 2 == 0 else -0.01)) for i in range(30)]
    spike = [_reading(30 + i, 0.95) for i in range(5)]
    results = replay(baseline + spike, min_samples=20)
    spike_results = results[-5:]
    assert any(r.is_elevated_this_tick for r in spike_results)


def test_sustained_trend_requires_consecutive_hits_not_a_single_spike() -> None:
    baseline = [_reading(i, 0.15) for i in range(25)]
    single_spike = [_reading(25, 0.95)] + [_reading(26 + i, 0.15) for i in range(5)]
    results = replay(baseline + single_spike, min_samples=20, sustained_hits=3)
    assert not any(r.is_sustained_trend for r in results)

    sustained_spike = [_reading(25 + i, 0.95) for i in range(5)]
    results2 = replay(baseline + sustained_spike, min_samples=20, sustained_hits=3)
    assert any(r.is_sustained_trend for r in results2)


def test_consecutive_elevated_resets_on_a_non_elevated_tick() -> None:
    baseline = [_reading(i, 0.15) for i in range(25)]
    pattern = [
        _reading(25, 0.95),
        _reading(26, 0.95),
        _reading(27, 0.15),  # break the streak
        _reading(28, 0.95),
    ]
    results = replay(baseline + pattern, min_samples=20, sustained_hits=3)
    # The streak breaks at index 27 (0.15), so consecutive_elevated resets to 0
    # there, and by index 28 it can only be back to 1 -- never reaching 3
    # again within this short tail.
    assert results[-1].state.consecutive_elevated <= 1
    assert not results[-1].is_sustained_trend


def test_state_is_immutable_and_resumable() -> None:
    state = MetacogTrendStateV1()
    result1 = apply_reading(state, _reading(0, 0.3))
    # Original state object must be untouched (frozen dataclass, pure function).
    assert state.count == 0
    # Resuming from a persisted state gives the same result as continuing
    # the same in-memory replay.
    result2 = apply_reading(result1.state, _reading(1, 0.4))
    resumed_state = MetacogTrendStateV1(**result1.state.__dict__)
    result2_resumed = apply_reading(resumed_state, _reading(1, 0.4))
    assert result2 == result2_resumed


def test_confidence_is_carried_through_but_not_used_in_fold_math() -> None:
    # Two readings with identical level but different confidence must fold
    # identically -- confidence-gating is the caller's job (see module
    # docstring), not this function's.
    state = MetacogTrendStateV1()
    result_low_conf = apply_reading(state, MetacogTrendReading(
        at=datetime(2026, 7, 24, tzinfo=timezone.utc), level=0.3, confidence=0.0,
    ))
    result_high_conf = apply_reading(state, MetacogTrendReading(
        at=datetime(2026, 7, 24, tzinfo=timezone.utc), level=0.3, confidence=0.9,
    ))
    assert result_low_conf.state.ewma == result_high_conf.state.ewma
    assert result_low_conf.state.variance == result_high_conf.state.variance


def test_min_samples_boundary_is_exact() -> None:
    # new_count < min_samples is cold-start; new_count == min_samples is the
    # first tick allowed to leave cold-start. Exercise the exact boundary,
    # not just wide margins either side of it.
    readings = [_reading(i, 0.2 + 0.001 * i) for i in range(20)]
    results = replay(readings, min_samples=20)
    assert results[18].is_cold_start is True   # count reaches 19 here
    assert results[19].is_cold_start is False  # count reaches 20 here


def test_resuming_with_different_config_changes_classification_with_no_new_evidence() -> None:
    # Documents a real, disclosed limitation (see MetacogTrendStateV1's
    # docstring): the checkpointed state does not carry the config it was
    # warmed up under, so resuming with a different min_samples reclassifies
    # the SAME accumulated state without any new reading being folded in.
    # This is expected today (no live caller exists yet to own a guard
    # against it) -- this test pins the current behavior so a future change
    # to either direction is a deliberate decision, not a silent regression.
    readings = [_reading(i, 0.3) for i in range(10)]
    state = MetacogTrendStateV1()
    for r in readings:
        result = apply_reading(state, r, min_samples=20)
        state = result.state
    assert result.is_cold_start is True  # correctly still cold-start under min_samples=20

    # Resume the SAME state, only the config differs, zero new evidence.
    same_state_new_config = apply_reading(state, _reading(10, 0.3), min_samples=5)
    assert same_state_new_config.is_cold_start is False  # flips with no new real evidence


def test_replay_matches_manual_incremental_application() -> None:
    readings = [_reading(i, 0.1 * (i % 5)) for i in range(15)]
    batch_results = replay(readings, min_samples=5)

    state = MetacogTrendStateV1()
    manual_results = []
    for r in readings:
        result = apply_reading(state, r, min_samples=5)
        manual_results.append(result)
        state = result.state

    assert batch_results == manual_results
