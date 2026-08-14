"""R2 of the phase-5 roadmap: level, dispersion, saturation and refresh state
as separate readings over a declared window.

Fixtures are hand-computed and the arithmetic is written out longhand, so a
reader can re-derive each expected value without running the code.
"""
import pytest

from orion.field.regime import (
    LOADED_LEVEL,
    STEADY_DISPERSION,
    ChannelRegime,
    channel_regime,
)


def _flat(value: float, n: int = 20) -> list[float]:
    return [value] * n


def test_loaded_steady_is_the_reading_this_module_exists_for() -> None:
    """Busy near the top of the range, and not moving -- which a single
    verdict label collapses into "quiet", indistinguishable from a channel
    idling at 0.02.

    Hand-computed: 20 samples alternating 0.810/0.815.
      median      = 0.8125            (>= LOADED_LEVEL 0.70   -> loaded)
      pstdev      = 0.0025            (<= STEADY_DISPERSION 0.02 -> steady)
      saturation  = 0/20 = 0.0
      refresh     = producer_written (0.815/0.810 = 1.00617, not a decay rate)
    Mirrors live memory_pressure (0.8126, dispersion 0.00056).
    """
    values = [0.810, 0.815] * 10
    r = channel_regime("memory_pressure", values, window_seconds=1224.0)

    assert r.level == pytest.approx(0.8125)
    assert r.dispersion == pytest.approx(0.0025)
    assert r.level >= LOADED_LEVEL
    assert r.dispersion <= STEADY_DISPERSION
    assert r.saturation == 0.0
    assert r.refresh_state == "producer_written"
    assert r.regime == "loaded_steady"


def test_loaded_volatile_is_a_different_reading_at_the_same_level() -> None:
    """Same level band as the test above, but moving. The two must not
    collapse together -- that discrimination is the point.

    Hand-computed: 20 samples alternating 0.55/1.00 (never at a rail long
    enough to saturate: 10/20 = 0.5 exactly, and the check is > 0.5).
      median     = 0.775  (>= 0.70 -> loaded)
      pstdev     = 0.225  (>  0.02 -> not steady)
    """
    values = [0.55, 1.00] * 10
    r = channel_regime("power_pressure", values, window_seconds=1224.0)

    assert r.level == pytest.approx(0.775)
    assert r.dispersion == pytest.approx(0.225)
    assert r.saturation == 0.5
    assert r.regime == "loaded_volatile"


def test_decay_only_is_detected_at_the_real_rate() -> None:
    """The node:substrate.route signature: every step an exact 0.92
    multiplication, nothing refreshing it. "The numbers move" is not evidence
    the metric is alive.
    """
    values = [0.5]
    for _ in range(19):
        values.append(values[-1] * 0.92)

    r = channel_regime("reliability_pressure", values, window_seconds=1224.0)
    assert r.refresh_state == "decay_only"
    assert r.regime == "no_input"


def test_one_producer_write_disqualifies_decay_only() -> None:
    """Conservative direction: mislabelling a live channel as decay-only is
    the damaging error, so a single non-decay step is enough to reject it."""
    values = [0.5]
    for _ in range(9):
        values.append(values[-1] * 0.92)
    values.append(0.77)  # a real write
    for _ in range(9):
        values.append(values[-1] * 0.92)

    r = channel_regime("some_channel", values, window_seconds=1224.0)
    assert r.refresh_state == "producer_written"
    assert r.regime != "no_input"


def test_idle_chat_channel_reads_no_input_not_dead() -> None:
    """repair_pressure/conversation_load are chat-gated. With nobody talking
    to Orion they sit at a decayed subnormal -- which is CORRECT, not a
    defect. The readout must not call that dead, and must make no claim about
    whether the silence is right.

    Mirrors the live value exactly (2.964e-323 on 2026-08-13).
    """
    r = channel_regime("repair_pressure", _flat(2.964e-323), window_seconds=1224.0)

    assert r.refresh_state == "static"
    assert r.regime == "no_input"
    # Specifically NOT a verdict about correctness.
    assert r.regime != "dead"


def test_refresh_state_dominates_the_composed_label() -> None:
    """A number nothing wrote is not a reading about the world, regardless of
    how high it sits. A channel pinned at 1.0 with no writes is no_input, not
    loaded_steady."""
    r = channel_regime("availability", _flat(1.0), window_seconds=1224.0)
    assert r.level == 1.0
    assert r.saturation == 1.0
    assert r.refresh_state == "static"
    assert r.regime == "no_input"


def test_saturation_counts_both_rails() -> None:
    """Hand-computed: 20 samples, 5 at 0.0 and 5 at 1.0 are within
    RAIL_EPSILON of a bound; the other 10 (0.4/0.6) are not.
      saturation = 10/20 = 0.5
    """
    values = ([0.0] * 5) + ([1.0] * 5) + ([0.4, 0.6] * 5)
    r = channel_regime("gpu_pressure", values, window_seconds=1224.0)
    assert r.saturation == pytest.approx(0.5)


def test_saturated_label_requires_a_majority_at_the_rails() -> None:
    """Hand-computed: 20 samples, 15 at 1.0 -> 0.75 > 0.5, and the series
    changes by a non-decay ratio so refresh does not pre-empt it."""
    values = ([1.0, 0.5] * 2) + ([1.0] * 15) + [0.5]
    r = channel_regime("some_channel", values, window_seconds=1224.0)
    assert r.saturation > 0.5
    assert r.refresh_state == "producer_written"
    assert r.regime == "saturated"


def test_window_seconds_is_carried_not_inferred() -> None:
    """The unit trap this module exists to close: a 600-tick window is ~20
    minutes at the live 2.04s cadence, and nothing recorded that."""
    r = channel_regime("cpu_pressure", [0.1, 0.2] * 10, window_seconds=1224.0)
    assert r.window_seconds == 1224.0
    assert r.sample_count == 20


def test_drift_and_relative_readings_are_none_without_a_baseline() -> None:
    """Absent, never a fabricated 0.0 (which would read as "no drift") or 0.5
    (which would read as "perfectly typical")."""
    r = channel_regime("cpu_pressure", [0.1, 0.2] * 10, window_seconds=1224.0)
    assert r.drift is None
    assert r.level_percentile is None
    assert r.dispersion_ratio is None


def test_relative_readings_are_computed_when_a_baseline_is_given() -> None:
    """Hand-computed: window is a flat 0.9; baseline is 100 samples spread
    0.0..0.99 (every value below 0.9 except the top 10).
      level_percentile = fraction of baseline strictly below 0.9 = 90/100
    """
    baseline = [i / 100 for i in range(100)]
    r = channel_regime("cpu_pressure", _flat(0.9), window_seconds=1224.0,
                       baseline=baseline)
    assert r.level_percentile == pytest.approx(0.90)
    assert r.drift is not None
    assert r.dispersion_ratio == pytest.approx(0.0)


def test_short_windows_refuse_a_regime_but_still_report_refresh() -> None:
    """Below MIN_REGIME_SAMPLES dispersion and drift are artifacts, so the
    composed label refuses -- but refresh_state needs only two points and
    stays readable."""
    values = [0.5, 0.46]  # exactly one 0.92 step
    r = channel_regime("some_channel", values, window_seconds=4.0)
    assert r.sample_count == 2
    assert r.refresh_state == "decay_only"
    assert r.regime == "insufficient_samples"


def test_empty_series_is_insufficient_not_zero() -> None:
    r = channel_regime("absent_channel", [], window_seconds=1224.0)
    assert isinstance(r, ChannelRegime)
    assert r.sample_count == 0
    assert r.regime == "insufficient_samples"
    assert r.refresh_state == "insufficient_samples"
    assert r.drift is None


def test_calm_is_low_and_steady_with_a_live_producer() -> None:
    """Hand-computed: alternating 0.10/0.11.
      median = 0.105 (< 0.70 -> not loaded)
      pstdev = 0.005 (<= 0.02 -> steady)
      0.11/0.10 = 1.1, not a decay rate -> producer_written
    """
    r = channel_regime("disk_pressure", [0.10, 0.11] * 10, window_seconds=1224.0)
    assert r.level == pytest.approx(0.105)
    assert r.dispersion == pytest.approx(0.005)
    assert r.regime == "calm"
