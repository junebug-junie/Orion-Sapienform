from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.field.pressure import (
    RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    collect_field_channel_pressures,
)
from orion.schemas.field_state import FieldStateV1

from app.digestion.perturbation import apply_perturbations
from app.ingest.state_deltas import Perturbation

BASE = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
TICK_SECONDS = 2.0


def _empty_state(tick_id: str) -> FieldStateV1:
    return FieldStateV1(generated_at=BASE, tick_id=tick_id)


def _tick(state: FieldStateV1, n_labels: int, *, tick_index: int, prefix: str = "p") -> None:
    """Apply n_labels new distinct perturbations at BASE + tick_index * TICK_SECONDS."""
    at = BASE + timedelta(seconds=TICK_SECONDS * tick_index)
    perturbations = [
        Perturbation(
            node_id="node:athena",
            channel="cortex_exec_step_load",
            intensity=0.1,
            label=f"{prefix}_{tick_index}_{i}",
        )
        for i in range(n_labels)
    ]
    apply_perturbations(state, perturbations, now=at)


def test_first_tick_has_no_baseline_yet() -> None:
    # Regression guard for the bus_synaptic_prediction_error class of bug:
    # a metric that varies but is never validated against its true rest
    # state. Here the true "no signal yet" state is an ABSENT channel, not
    # a fake 0.0 or a fake 1.0 -- there's no baseline on the very first
    # observation (orion.bus.ewma.compute_ewma_update's own documented
    # behavior).
    state = _empty_state("tick_0")
    _tick(state, 5, tick_index=0)
    assert state.recent_perturbation_zscore is None
    assert state.recent_perturbation_ewma_n == 1
    channels, _ = collect_field_channel_pressures(state)
    assert "recent_perturbation_count" not in channels


def test_cold_start_samples_below_min_do_not_produce_a_reading() -> None:
    # Confirmed by hand (2026-07-28): the shared EWMA utility's z-score is
    # wildly unreliable for the first few samples (z=1000 on the second
    # observation for a steady 1-unit/tick ramp, since variance is
    # estimated from a single prior point). Below
    # RECENT_PERTURBATION_EWMA_MIN_SAMPLES, the channel must stay absent
    # even though a numeric zscore already exists on state.
    state = _empty_state("tick_cold")
    for i in range(RECENT_PERTURBATION_EWMA_MIN_SAMPLES - 1):
        _tick(state, 5 + i, tick_index=i)
    assert state.recent_perturbation_ewma_n == RECENT_PERTURBATION_EWMA_MIN_SAMPLES - 1
    assert state.recent_perturbation_zscore is not None
    channels, _ = collect_field_channel_pressures(state)
    assert "recent_perturbation_count" not in channels


def test_steady_baseline_does_not_saturate_even_past_the_old_fixed_caps() -> None:
    # The actual live bug this fixes: real steady-state traffic (observed
    # live 2026-07-28: ~100-118 distinct labels/60s window) already blew
    # past both the old count/10.0 (selectors.py) and count/20.0
    # (pressure.py) fixed caps, pinning both permanently at 1.0.
    #
    # Simulated here as a constant rate of 3 new distinct labels every 2s
    # tick -- the window itself naturally converges to ~93 entries once
    # entries start expiring past the 60s mark (tick ~30), well past both
    # old caps. Run long enough (150 ticks = 300s of simulated time) for the
    # EWMA baseline to catch up to that new steady window size too --
    # hand-verified via direct simulation that the reading is already <0.1
    # by this point, monotonically decreasing from its ramp-up peak. Once
    # past cold start, "always ~93" must read as this mesh's genuine normal
    # (near 0), not as permanent maximum surprise.
    state = _empty_state("tick_steady")
    for i in range(150):
        _tick(state, 3, tick_index=i)
    assert state.recent_perturbation_ewma_n >= RECENT_PERTURBATION_EWMA_MIN_SAMPLES
    assert len(state.recent_perturbations) > 20  # comfortably past both old fixed caps (10, 20)
    channels, _ = collect_field_channel_pressures(state)
    assert channels["recent_perturbation_count"] < 0.3


def test_genuine_burst_above_established_baseline_is_flagged() -> None:
    # Establish a GENUINELY settled baseline first, not just "past the
    # RECENT_PERTURBATION_EWMA_MIN_SAMPLES cold-start gate" -- 5-9 samples in
    # is nowhere near enough for either the 60s recent_perturbations window
    # or the alpha=0.02 EWMA to have converged (hand-verified: a 20-tick
    # "baseline" here reads ~0.97 with NO burst at all, purely from the
    # window/EWMA still ramping up from empty -- that variant of this test
    # would have passed even with the burst line deleted, so it wasn't
    # actually testing burst detection). 150 ticks matches
    # test_steady_baseline_does_not_saturate_even_past_the_old_fixed_caps's
    # already-verified convergence point (~0.095 by then).
    state = _empty_state("tick_burst_after_baseline")
    for i in range(150):
        _tick(state, 3, tick_index=i)
    pre_burst_channels, _ = collect_field_channel_pressures(state)
    pre_burst_reading = pre_burst_channels["recent_perturbation_count"]
    assert pre_burst_reading < 0.3  # confirms the baseline really is settled

    _tick(state, 300, tick_index=150, prefix="burst")

    channels, _ = collect_field_channel_pressures(state)
    assert channels["recent_perturbation_count"] > 0.5
    assert channels["recent_perturbation_count"] > pre_burst_reading


def test_burst_does_not_pin_the_reading_forever() -> None:
    # The actual regression the old mechanism could never satisfy: after a
    # burst, once activity returns to normal, the reading must come back
    # down (the EWMA baseline has partially absorbed the burst, and the
    # burst's own labels eventually age out of the 60s window), not stay
    # pinned at its post-burst high forever. Hand-verified via direct
    # simulation: with this baseline/burst/recovery shape, the reading is
    # back at its pre-burst level (in this case exactly 0.0) within 80s of
    # the burst ending.
    state = _empty_state("tick_burst_recovers")
    for i in range(150):
        _tick(state, 3, tick_index=i)
    burst_index = 150
    _tick(state, 300, tick_index=burst_index, prefix="burst")
    channels_at_burst, _ = collect_field_channel_pressures(state)
    burst_reading = channels_at_burst.get("recent_perturbation_count", 0.0)
    assert burst_reading > 0.9  # confirms the burst itself was actually flagged

    for j in range(1, 40):
        _tick(state, 3, tick_index=burst_index + j)
    channels_after, _ = collect_field_channel_pressures(state)
    recovered_reading = channels_after.get("recent_perturbation_count", 0.0)

    assert recovered_reading < burst_reading
