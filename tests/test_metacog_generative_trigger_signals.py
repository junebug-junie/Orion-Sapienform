"""Unit tests for the two generative (non-rupture) metacog trigger detectors.

Covers orion/substrate/metacog_trigger_signals.py's `detect_confidence_recovery`
("insight") and `detect_flow_regime` ("flow") against synthetic tick sequences
shaped like the real `substrate_attention_self_model` history they were
calibrated on (see docs/superpowers/specs/2026-07-28-collapse-mirror-generative-
triggers-design.md).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from orion.substrate.metacog_trigger_signals import (
    ConfidenceSample,
    detect_confidence_recovery,
    detect_flow_regime,
)

TICK = timedelta(seconds=30)
T0 = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)

# Live defaults from services/orion-equilibrium-service/app/settings.py.
LOW = 0.70
HIGH = 0.90
MAX_CROSS = 15
CONFIRM = 2
FLOOR = 0.90
MAX_STDEV = 0.02
MIN_TICKS = 20
EXPECTED_TICK_SEC = 30.0
SPAN_TOLERANCE = 2.0
# Derived exactly as services/orion-equilibrium-service/app/service.py does.
MAX_CROSS_SPAN_SEC = MAX_CROSS * EXPECTED_TICK_SEC * SPAN_TOLERANCE
MAX_FLOW_SPAN_SEC = (MIN_TICKS - 1) * EXPECTED_TICK_SEC * SPAN_TOLERANCE


def _samples(values: list[float]) -> list[ConfidenceSample]:
    """Oldest -> newest, one tick apart, matching the real ~30s cadence."""
    return [
        ConfidenceSample(generated_at=T0 + i * TICK, value=v)
        for i, v in enumerate(values)
    ]


def _recovery(values: list[float], samples=None, **kwargs):
    params = {
        "low_threshold": LOW,
        "high_threshold": HIGH,
        "max_ticks_to_cross": MAX_CROSS,
        "confirm_ticks": CONFIRM,
        "max_cross_span_sec": MAX_CROSS_SPAN_SEC,
    }
    params.update(kwargs)
    return detect_confidence_recovery(
        _samples(values) if samples is None else samples, **params
    )


def _flow(values: list[float], samples=None, **kwargs):
    params = {
        "floor": FLOOR,
        "max_stdev": MAX_STDEV,
        "min_ticks": MIN_TICKS,
        "max_span_sec": MAX_FLOW_SPAN_SEC,
    }
    params.update(kwargs)
    return detect_flow_regime(
        _samples(values) if samples is None else samples, **params
    )


# ===========================================================================
# insight: detect_confidence_recovery
# ===========================================================================


def test_real_shaped_gradual_recovery_fires() -> None:
    """The shape PR #1463 actually measured: a multi-tick gradual climb out of
    the low band, then holding high. This is the case a single-tick crossing
    gate would have gotten wrong."""
    event = _recovery([0.95, 0.68, 0.72, 0.80, 0.87, 0.91, 0.93])
    assert event is not None
    assert event.low_value == 0.68
    assert event.high_value == 0.91
    # low at index 1, high run starts at index 5
    assert event.ticks_to_cross == 4
    assert event.confirm_ticks == CONFIRM
    assert event.window_ticks == 7
    assert event.low_at == T0 + 1 * TICK
    assert event.high_at == T0 + 5 * TICK


def test_flat_calm_sequence_does_not_fire() -> None:
    """Sustained high confidence is `flow`, not `insight` -- with no preceding
    low band there was no surprise to resolve."""
    assert _recovery([0.93] * 10) is None


def test_single_tick_high_spike_mid_climb_does_not_fire() -> None:
    """The specific misfire the confirm requirement exists to prevent: one
    noisy tick pokes above the high band partway up the climb, then falls back.
    A single-tick `>= high` gate would have fired here."""
    assert _recovery([0.65, 0.75, 0.91, 0.84]) is None


def test_confirm_ticks_not_yet_satisfied_does_not_fire() -> None:
    """Only one tick has reached the high band so far -- a real recovery may be
    underway, but it is not yet confirmed."""
    assert _recovery([0.65, 0.80, 0.91]) is None
    # One more sustained tick and the same recovery does fire.
    assert _recovery([0.65, 0.80, 0.91, 0.92]) is not None


def test_low_too_long_ago_is_not_called_a_recovery() -> None:
    """A low from far outside max_ticks_to_cross must not be retroactively
    stitched to a present-day high band."""
    values = [0.60] + [0.80] * 20 + [0.95, 0.96]
    assert _recovery(values) is None
    # Same data, a max_ticks_to_cross wide enough to span it, and it fires.
    assert _recovery(values, max_ticks_to_cross=40) is not None


def test_never_dropped_into_low_band_does_not_fire() -> None:
    """Mirrors the real finding that the design doc's original 0.5 anchor never
    fired: a dip that never reaches the low threshold is not a surprise."""
    assert _recovery([0.88, 0.75, 0.80, 0.94, 0.95]) is None


def test_recovery_is_stable_while_an_unbroken_high_run_holds() -> None:
    """While the high run is *unbroken*, both anchors hold steady as more high
    ticks arrive. Narrow on purpose: this does NOT prove `high_at` is safe
    episode identity in general -- see
    test_low_at_is_stable_when_the_high_run_breaks_and_reforms for the case
    where it is not, which is why the service keys on `low_at`."""
    base = [0.95, 0.66, 0.78, 0.91, 0.92]
    first = _recovery(base)
    later = _recovery(base + [0.93, 0.94])
    assert first is not None and later is not None
    assert first.high_at == later.high_at
    assert first.ticks_to_cross == later.ticks_to_cross


def test_non_finite_value_fails_closed() -> None:
    """A NaN compares False against every threshold, so it must skip the window
    rather than be silently mis-evaluated."""
    assert _recovery([0.65, 0.80, math.nan, 0.95, 0.96]) is None


def test_too_few_samples_for_confirm_does_not_fire() -> None:
    assert _recovery([0.95]) is None


# ===========================================================================
# flow: detect_flow_regime
# ===========================================================================


def test_sustained_high_low_variance_fires() -> None:
    values = [0.93, 0.94, 0.93, 0.95, 0.94] * 4  # 20 ticks, tight band
    regime = _flow(values)
    assert regime is not None
    assert regime.tick_count == MIN_TICKS
    assert regime.min_value >= FLOOR
    assert regime.stdev_value <= MAX_STDEV
    assert regime.started_at == T0
    assert regime.ended_at == T0 + 19 * TICK


def test_one_dip_below_floor_blocks_flow() -> None:
    """The hard floor on the window *minimum* is the whole point: a single real
    dip cannot be averaged away, which `mean - k*stdev >= floor` would allow."""
    values = [0.93] * 19 + [0.85]
    assert _flow(values) is None
    # Same window with the dip removed does fire, proving the dip was the cause.
    assert _flow([0.93] * 20) is not None


def test_noisy_high_variance_window_blocks_flow() -> None:
    """All ticks above the floor but swinging -- high confidence, not calm."""
    values = [0.91, 0.99, 0.91, 0.99] * 5
    assert min(values) >= FLOOR
    assert _flow(values) is None


def test_declining_sequence_does_not_fire() -> None:
    values = [0.98 - 0.01 * i for i in range(20)]
    assert _flow(values) is None


def test_only_trailing_window_is_evaluated() -> None:
    """"Sustained" means the last N consecutive ticks -- an old rough patch
    before a genuinely calm run must not veto it."""
    values = [0.40, 0.99, 0.55] + [0.93] * MIN_TICKS
    regime = _flow(values)
    assert regime is not None
    assert regime.tick_count == MIN_TICKS
    assert regime.started_at == T0 + 3 * TICK


def test_too_few_ticks_does_not_fire() -> None:
    """A calm window shorter than min_ticks is not yet evidence of sustained
    calm -- fail closed rather than claiming a regime from 3 samples."""
    assert _flow([0.93] * (MIN_TICKS - 1)) is None


def test_non_finite_value_fails_closed_for_flow() -> None:
    assert _flow([0.93] * (MIN_TICKS - 1) + [math.inf]) is None


# ===========================================================================
# Contiguity / staleness regressions (review finding M1, 2026-07-30).
# Row adjacency is not tick adjacency: the reader drops rows whose confidence is
# missing/non-finite, so a "20 consecutive tick" window can really span hours.
# ===========================================================================


def _gappy(values: list[float], gap_after: int, gap: timedelta):
    """Samples where a real time gap opens after index `gap_after`, exactly as
    dropped rows would produce."""
    out, ts = [], T0
    for i, v in enumerate(values):
        out.append(ConfidenceSample(generated_at=ts, value=v))
        ts += gap if i == gap_after else TICK
    return out


def test_insight_rejects_a_low_hours_before_the_high_run() -> None:
    """Pre-fix this fired with ticks_to_cross=1, because only the *tick* bound
    existed -- making the docstring's "not an hours-old low" claim false."""
    samples = _gappy([0.66, 0.91, 0.92], gap_after=0, gap=timedelta(hours=5))
    assert _recovery([], samples=samples) is None
    # Identical values with a real 30s cadence do fire, proving the span bound
    # is what rejected it rather than the values.
    assert _recovery([0.66, 0.91, 0.92]) is not None


def test_insight_span_bound_is_enforced_in_seconds() -> None:
    values = [0.66, 0.80, 0.91, 0.92]
    assert _recovery(values) is not None
    assert _recovery(values, max_cross_span_sec=1.0) is None


def test_flow_rejects_a_window_that_only_looks_consecutive() -> None:
    """Pre-fix, 20 rows spanning 6h fired while reporting tick_count=20 as
    though it were 10 minutes of sustained calm."""
    samples = _gappy([0.93] * MIN_TICKS, gap_after=5, gap=timedelta(hours=6))
    assert _flow([], samples=samples) is None
    assert _flow([0.93] * MIN_TICKS) is not None


def test_flow_records_real_span_for_auditing() -> None:
    regime = _flow([0.93] * MIN_TICKS)
    assert regime is not None
    assert regime.span_sec == (MIN_TICKS - 1) * 30.0


def test_insight_records_real_cross_span_for_auditing() -> None:
    recovery = _recovery([0.66, 0.80, 0.91, 0.92])
    assert recovery is not None
    # low at index 0, high run starts at index 2 -> 2 real ticks.
    assert recovery.cross_span_sec == 60.0
    assert recovery.ticks_to_cross == 2


# ===========================================================================
# Double-fire regression (review finding M2, 2026-07-30).
# ===========================================================================


def test_low_at_is_stable_when_the_high_run_breaks_and_reforms() -> None:
    """The bug: a single sub-threshold tick mid-high-run re-anchors `high_at`,
    so de-duping on it published one real recovery twice (390s apart, clearing
    the 300s cooldown). `low_at` is the stable episode identity, so the service
    keys on it -- this locks that property in.

    Simulates the real sliding window the reader produces, not an append-only
    list, which is what the earlier stability test failed to do.
    """
    series = [0.95, 0.66, 0.80, 0.91, 0.92, 0.93, 0.89, 0.91, 0.92, 0.93, 0.94]
    all_samples = _samples(series)

    low_ats, high_ats = set(), set()
    for end in range(3, len(all_samples) + 1):
        window = all_samples[max(0, end - 20) : end]
        event = _recovery([], samples=window)
        if event is not None:
            low_ats.add(event.low_at)
            high_ats.add(event.high_at)

    # The whole point: high_at drifts (that was the double-fire), low_at doesn't.
    assert len(high_ats) > 1, "expected high_at to re-anchor when the run breaks"
    assert len(low_ats) == 1, f"low_at must identify one episode, got {low_ats}"
    assert low_ats == {T0 + 1 * TICK}


def test_degenerate_floor_documented_in_settings_blocks_everything() -> None:
    """floor=0.92 measured 0 qualifying windows against 2246 real 20-tick
    windows. Locks in that finding: a window whose real min is 0.91 must not
    qualify at that floor, which is why settings.py warns against raising it."""
    values = [0.91, 0.93] * 10
    assert _flow(values) is not None  # fires at the shipped floor of 0.90
    assert _flow(values, floor=0.92) is None
