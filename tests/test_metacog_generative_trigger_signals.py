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


def _samples(values: list[float]) -> list[ConfidenceSample]:
    """Oldest -> newest, one tick apart, matching the real ~30s cadence."""
    return [
        ConfidenceSample(generated_at=T0 + i * TICK, value=v)
        for i, v in enumerate(values)
    ]


def _recovery(values: list[float], **kwargs):
    params = {
        "low_threshold": LOW,
        "high_threshold": HIGH,
        "max_ticks_to_cross": MAX_CROSS,
        "confirm_ticks": CONFIRM,
    }
    params.update(kwargs)
    return detect_confidence_recovery(_samples(values), **params)


def _flow(values: list[float], **kwargs):
    params = {"floor": FLOOR, "max_stdev": MAX_STDEV, "min_ticks": MIN_TICKS}
    params.update(kwargs)
    return detect_flow_regime(_samples(values), **params)


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


def test_recovery_is_stable_while_high_run_holds() -> None:
    """`high_at` is what the service de-dupes on, so it must NOT drift as more
    high ticks arrive -- otherwise one real recovery would re-fire every poll."""
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


def test_degenerate_floor_documented_in_settings_blocks_everything() -> None:
    """floor=0.92 measured 0 qualifying windows against 2246 real 20-tick
    windows. Locks in that finding: a window whose real min is 0.91 must not
    qualify at that floor, which is why settings.py warns against raising it."""
    values = [0.91, 0.93] * 10
    assert _flow(values) is not None  # fires at the shipped floor of 0.90
    assert _flow(values, floor=0.92) is None
