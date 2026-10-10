"""Unit tests for the generative (non-rupture) insight metacog trigger detector.

Covers orion/substrate/metacog_trigger_signals.py's `detect_confidence_recovery`
("insight") against synthetic tick sequences
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
)

TICK = timedelta(seconds=30)
T0 = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)

# Live defaults from services/orion-equilibrium-service/app/settings.py.
LOW = 0.70
HIGH = 0.90
MAX_CROSS = 15
CONFIRM = 2
EXPECTED_TICK_SEC = 30.0
SPAN_TOLERANCE = 2.0
# Derived exactly as services/orion-equilibrium-service/app/service.py does.
MAX_CROSS_SPAN_SEC = MAX_CROSS * EXPECTED_TICK_SEC * SPAN_TOLERANCE


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
    """Sustained high confidence is not `insight` -- with no preceding
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
