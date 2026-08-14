"""R2 of the phase-5 roadmap: level, dispersion, saturation and refresh state
as separate readings over a declared window.

An earlier version of this suite asserted things like `r.level >= LOADED_LEVEL`
after importing LOADED_LEVEL from the module under test. That is a tautology:
it cannot fail for ANY value of the constant. A mutation harness found 23 of 30
mutants surviving, including every threshold and `drift = 0.0` hardcoded.

So: every expected value below is an exact number, hand-derived and written out
longhand, and each threshold has a fixture on both sides of its boundary.
"""
from datetime import datetime, timedelta, timezone

import pytest

from orion.field.regime import (
    MIN_DECAY_CHANGES,
    ChannelRegime,
    channel_regime,
)

T0 = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


def _flat(value: float, n: int = 20) -> list[float]:
    return [value] * n


# --------------------------------------------------------------------------
# the reading this module exists for
# --------------------------------------------------------------------------

def test_loaded_steady_is_the_reading_this_module_exists_for() -> None:
    """Busy near the top of the range and not moving -- which a single verdict
    collapses into "quiet", indistinguishable from a channel idling at 0.02.

    Hand-computed, 20 samples alternating 0.810/0.815:
      median = (0.810 + 0.815) / 2 = 0.8125
      pstdev = 0.0025 exactly (alternating +/- d about the mean gives d)
      saturation_low = saturation_high = 0
    Mirrors live memory_pressure (0.8126, dispersion 0.00056).
    """
    r = channel_regime("memory_pressure", [0.810, 0.815] * 10, window_seconds=1224.0)

    assert r.level == pytest.approx(0.8125)
    assert r.pressure_equivalent_level == pytest.approx(0.8125)
    assert r.polarity_inverted is False
    assert r.dispersion == pytest.approx(0.0025)
    assert r.saturation_low == 0.0
    assert r.saturation_high == 0.0
    assert r.regime == "loaded_steady"


def test_loaded_volatile_is_a_different_reading_at_the_same_level() -> None:
    """Same level band, moving. Hand-computed, 20 samples alternating
    0.55/0.99: median 0.77, pstdev 0.22, no sample within RAIL_EPSILON of a
    bound (0.99 is 0.01 away, RAIL_EPSILON is 1e-3)."""
    r = channel_regime("power_pressure", [0.55, 0.99] * 10, window_seconds=1224.0)

    assert r.level == pytest.approx(0.77)
    assert r.dispersion == pytest.approx(0.22)
    assert r.saturation_high == 0.0
    assert r.regime == "loaded_volatile"


# --------------------------------------------------------------------------
# thresholds: a fixture on each side of every boundary
# --------------------------------------------------------------------------

def test_steady_dispersion_boundary() -> None:
    """Pins STEADY_DISPERSION at 0.02, not merely "somewhere in [0.005, 0.225)".
    Alternating 0.5 +/- d gives pstdev exactly d."""
    steady = channel_regime("c", [0.5 - 0.019, 0.5 + 0.019] * 10, window_seconds=1.0)
    assert steady.dispersion == pytest.approx(0.019)
    assert steady.regime == "calm"

    volatile = channel_regime("c", [0.5 - 0.021, 0.5 + 0.021] * 10, window_seconds=1.0)
    assert volatile.dispersion == pytest.approx(0.021)
    assert volatile.regime == "quiet_volatile"


def test_loaded_level_boundary_is_inclusive_at_0_70() -> None:
    """Pins LOADED_LEVEL at 0.70 and the comparison as `>=`."""
    at = channel_regime("c", [0.70 - 0.001, 0.70 + 0.001] * 10, window_seconds=1.0)
    assert at.level == pytest.approx(0.70)
    assert at.regime == "loaded_steady"

    below = channel_regime("c", [0.69 - 0.001, 0.69 + 0.001] * 10, window_seconds=1.0)
    assert below.level == pytest.approx(0.69)
    assert below.regime == "calm"


def test_loaded_comparison_is_inclusive_exactly_at_the_threshold() -> None:
    """Pins `>=` vs `>` at exact equality, which is the ONLY input that
    distinguishes them. A flat 0.70 series has median exactly the same float
    literal as LOADED_LEVEL, so this is a real equality, not an approximation.
    Timestamps supplied so refresh_state does not pre-empt the label.
    """
    stamps = [T0 + timedelta(seconds=i) for i in range(20)]
    r = channel_regime("c", _flat(0.70), window_seconds=1.0, updated_at=stamps)
    assert r.level == 0.70
    assert r.dispersion == 0.0
    assert r.refresh_state == "producer_written"
    assert r.regime == "loaded_steady"   # `>` would give "calm"


def test_steady_comparison_is_inclusive_exactly_at_the_threshold() -> None:
    """Pins `<=` vs `<` on dispersion at exact equality.

    Hand-computed: [0.0, 0.04] alternating has mean 0.02 and each sample
    deviates by exactly 0.02, and this pair is one of the few that yields
    pstdev == 0.02 EXACTLY in float (0.48/0.52 gives 0.020000000000000018,
    which is greater and would not test the boundary).
      median = 0.02, saturation_low = 10/20 = 0.5 (not > 0.5, so no pinning)
    """
    r = channel_regime("c", [0.0, 0.04] * 10, window_seconds=1.0)
    assert r.dispersion == 0.02
    assert r.saturation_low == 0.5
    assert r.regime == "calm"   # `<` would give "quiet_volatile"


def test_pinning_requires_a_strict_majority_not_exactly_half() -> None:
    """Pins `> 0.5` vs `>= 0.5`. Hand-computed: 10 of 20 samples at the low
    rail -> saturation_low exactly 0.5, median 0.25, pstdev 0.25."""
    low = channel_regime("c", [0.0] * 10 + [0.5] * 10, window_seconds=1.0)
    assert low.saturation_low == 0.5
    assert low.level == 0.25
    assert low.regime == "quiet_volatile"   # `>=` would give "pinned_min"

    # Same boundary on the OTHER rail. Needed separately: a fixture with only
    # saturation_low at 0.5 leaves the pinned_max comparison unexercised, and
    # a mutation of just that line survived the first version of this test.
    # Hand-computed: 10 of 20 at 1.0 -> saturation_high 0.5, median 0.75,
    # pstdev 0.25 -> loaded but not steady.
    high = channel_regime("c", [1.0] * 10 + [0.5] * 10, window_seconds=1.0)
    assert high.saturation_high == 0.5
    assert high.level == 0.75
    assert high.regime == "loaded_volatile"   # `>=` would give "pinned_max"


def test_a_non_decay_ratio_is_not_treated_as_decay() -> None:
    """Pins KNOWN_DECAY_RATES to 0.92 alone. A clean geometric series at 0.5
    per step is emphatically a producer doing something, not the staleness
    loop -- adding 0.5 to the rate tuple must not be invisible."""
    values = [0.9]
    for _ in range(19):
        values.append(values[-1] * 0.5)
    r = channel_regime("c", values, window_seconds=1.0)
    assert r.refresh_state == "producer_written"


def test_rail_epsilon_boundary() -> None:
    """Pins RAIL_EPSILON at 1e-3. 0.0005 is inside the low rail, 0.002 is not."""
    inside = channel_regime("c", _flat(0.0005), window_seconds=1.0)
    assert inside.saturation_low == 1.0

    outside = channel_regime("c", _flat(0.002), window_seconds=1.0)
    assert outside.saturation_low == 0.0

    # Exactly AT the epsilon -- the only input separating `<` from `<=`.
    # 1e-3 here is the same float literal as RAIL_EPSILON.
    at = channel_regime("c", _flat(1e-3), window_seconds=1.0)
    assert at.saturation_low == 0.0


def test_min_regime_samples_boundary_at_8() -> None:
    """Pins MIN_REGIME_SAMPLES at 8: dispersion is None below it and a real
    number at it."""
    seven = channel_regime("c", [0.1, 0.9] * 3 + [0.1], window_seconds=1.0)
    assert seven.sample_count == 7
    assert seven.dispersion is None
    assert seven.regime == "insufficient_samples"

    eight = channel_regime("c", [0.1, 0.9] * 4, window_seconds=1.0)
    assert eight.sample_count == 8
    assert eight.dispersion == pytest.approx(0.4)
    assert eight.regime != "insufficient_samples"


def test_level_is_the_median_not_the_mean() -> None:
    """The docstring claims median deliberately, for robustness to the 1.0
    spikes several channels show. Every fixture in the previous suite was
    symmetric, so mean == median throughout and the choice was untested.

    Hand-computed: nine 0.1s and one 1.0.
      median = (0.1 + 0.1) / 2 = 0.1
      mean   = (9*0.1 + 1.0) / 10 = 0.19
    """
    r = channel_regime("c", [0.1] * 9 + [1.0], window_seconds=1.0)
    assert r.level == pytest.approx(0.1)
    assert r.level != pytest.approx(0.19)


# --------------------------------------------------------------------------
# polarity
# --------------------------------------------------------------------------

def test_higher_is_better_channel_is_polarity_corrected() -> None:
    """`confidence` at 0.8678 means HEALTHY. An earlier version composed on raw
    level and called it loaded_volatile in 159 of 208 live windows.

    Hand-computed: flat 0.8678 -> level 0.8678 (native, healthy),
    pressure_equivalent_level = 1 - 0.8678 = 0.1322 (low pressure), so not
    loaded.
    """
    r = channel_regime("confidence", _flat(0.8678), window_seconds=1224.0)

    assert r.polarity_inverted is True
    assert r.level == pytest.approx(0.8678)
    assert r.pressure_equivalent_level == pytest.approx(0.1322)
    assert r.regime != "loaded_steady"
    assert r.regime != "loaded_volatile"


def test_pressure_channel_is_not_inverted() -> None:
    r = channel_regime("cpu_pressure", _flat(0.8678), window_seconds=1224.0)
    assert r.polarity_inverted is False
    assert r.level == pytest.approx(0.8678)
    assert r.pressure_equivalent_level == pytest.approx(0.8678)


# --------------------------------------------------------------------------
# refresh state: timestamp path is authoritative
# --------------------------------------------------------------------------

def test_timestamp_path_is_authoritative_and_says_so() -> None:
    """A repeated timestamp means the producer did not write in this window --
    a fact, not an inference from value ratios."""
    stamps = [T0] * 20
    r = channel_regime("availability", _flat(1.0), window_seconds=1224.0,
                       updated_at=stamps)
    assert r.refresh_evidence == "timestamp"
    assert r.refresh_state == "no_write_in_window"
    assert r.regime == "no_new_input"


def test_advancing_timestamp_beats_a_decay_looking_value_series() -> None:
    """The value series is a textbook 0.92 decay, so the fallback would call it
    decay_only. The timestamp says a producer wrote every tick, and the
    timestamp wins."""
    values = [0.5]
    for _ in range(19):
        values.append(values[-1] * 0.92)
    stamps = [T0 + timedelta(seconds=2 * i) for i in range(20)]

    r = channel_regime("c", values, window_seconds=1224.0, updated_at=stamps)
    assert r.refresh_evidence == "timestamp"
    assert r.refresh_state == "producer_written"


def test_all_none_timestamps_fall_back_to_value_inference() -> None:
    """Capability vectors carry no timestamps at all; that must not be read as
    'no writes'."""
    values = [0.1, 0.9] * 10
    r = channel_regime("c", values, window_seconds=1.0, updated_at=[None] * 20)
    assert r.refresh_evidence == "value_ratio"
    assert r.refresh_state == "producer_written"


# --------------------------------------------------------------------------
# refresh state: fallback inference and its limits
# --------------------------------------------------------------------------

def test_decay_only_is_detected_at_the_real_rate() -> None:
    values = [0.5]
    for _ in range(19):
        values.append(values[-1] * 0.92)
    r = channel_regime("reliability_pressure", values, window_seconds=1224.0)
    assert r.refresh_state == "decay_only"
    assert r.regime == "no_new_input"


def test_one_producer_write_disqualifies_decay_only() -> None:
    values = [0.5]
    for _ in range(9):
        values.append(values[-1] * 0.92)
    values.append(0.77)
    for _ in range(9):
        values.append(values[-1] * 0.92)
    r = channel_regime("c", values, window_seconds=1224.0)
    assert r.refresh_state == "producer_written"


def test_a_single_eight_percent_step_is_not_decay() -> None:
    """1.0 -> 0.92 is an ordinary two-decimal move, and many channels sit
    pinned at 1.0. Below MIN_DECAY_CHANGES the fallback must not assert
    decay_only off one coincidental ratio."""
    r = channel_regime("c", [1.0] * 10 + [0.92] * 10, window_seconds=1.0)
    assert r.refresh_state == "producer_written"

    # ...and exactly MIN_DECAY_CHANGES real decay steps is enough.
    values = [0.5]
    for _ in range(MIN_DECAY_CHANGES):
        values.append(values[-1] * 0.92)
    values += [values[-1]] * 10
    assert channel_regime("c", values, window_seconds=1.0).refresh_state == "decay_only"


def test_subnormal_decay_is_a_documented_blind_spot() -> None:
    """Pins the KNOWN limitation rather than pretending it does not exist: a
    genuine 0.92 decay below ~5e-318 reads producer_written, because float
    quantization puts b/a outside DECAY_RATIO_EPSILON.

    If this ever starts passing as decay_only, the module docstring's
    subnormal carve-out is stale and must be updated.
    """
    values = [1e-320]
    for _ in range(19):
        values.append(values[-1] * 0.92)
    r = channel_regime("repair_pressure", values, window_seconds=1224.0)
    assert r.refresh_state == "producer_written"
    assert r.refresh_evidence == "value_ratio"


def test_idle_chat_channel_reads_no_new_input_not_dead() -> None:
    """repair_pressure/conversation_load are chat-gated. With nobody talking to
    Orion they sit at a decayed subnormal, which is CORRECT -- not a defect.
    Mirrors the live value exactly (2.964e-323 on 2026-08-13)."""
    r = channel_regime("repair_pressure", _flat(2.964e-323), window_seconds=1224.0)
    assert r.refresh_state == "static"
    assert r.regime == "no_new_input"


def test_refresh_state_dominates_the_composed_label() -> None:
    """A number nothing wrote is not a reading about the world, however high."""
    r = channel_regime("cpu_pressure", _flat(1.0), window_seconds=1224.0)
    assert r.level == 1.0
    assert r.saturation_high == 1.0
    assert r.refresh_state == "static"
    assert r.regime == "no_new_input"


# --------------------------------------------------------------------------
# saturation: split by rail
# --------------------------------------------------------------------------

def test_saturation_is_split_by_rail() -> None:
    """Hand-computed: 20 samples, 5 at 0.0 and 5 at 1.0, 10 at 0.4/0.6.
      saturation_low  = 5/20 = 0.25
      saturation_high = 5/20 = 0.25
    """
    values = [0.0] * 5 + [1.0] * 5 + [0.4, 0.6] * 5
    r = channel_regime("gpu_pressure", values, window_seconds=1224.0)
    assert r.saturation_low == pytest.approx(0.25)
    assert r.saturation_high == pytest.approx(0.25)


def test_a_pressure_channel_at_the_floor_is_not_alarming() -> None:
    """failure_pressure reporting no failures is the healthiest possible state.
    A combined-rail draft labelled it "saturated" -- the most alarming word in
    the vocabulary -- and 495 of 505 live saturated labels were this case.

    Hand-computed: 12 zeros, 8 at 0.3 -> saturation_low = 12/20 = 0.6 > 0.5.
    """
    r = channel_regime("failure_pressure", [0.0] * 12 + [0.3] * 8,
                       window_seconds=1224.0,
                       updated_at=[T0 + timedelta(seconds=i) for i in range(20)])
    assert r.saturation_low == pytest.approx(0.6)
    assert r.regime == "pinned_min"


def test_a_pressure_channel_pinned_high_is_alarming() -> None:
    """Hand-computed: 12 at 1.0, 8 at 0.3 -> saturation_high = 0.6 > 0.5."""
    r = channel_regime("failure_pressure", [1.0] * 12 + [0.3] * 8,
                       window_seconds=1224.0,
                       updated_at=[T0 + timedelta(seconds=i) for i in range(20)])
    assert r.saturation_high == pytest.approx(0.6)
    assert r.regime == "pinned_max"


def test_pinned_rails_invert_for_a_higher_is_better_channel() -> None:
    """`confidence` pinned at 1.0 is the GOOD rail; pinned at 0.0 is the bad
    one. The label must follow pressure-equivalent meaning, not the raw rail."""
    good = channel_regime("confidence", [1.0] * 12 + [0.5] * 8,
                          window_seconds=1224.0,
                          updated_at=[T0 + timedelta(seconds=i) for i in range(20)])
    assert good.saturation_high == pytest.approx(0.6)
    assert good.regime == "pinned_min"

    bad = channel_regime("confidence", [0.0] * 12 + [0.5] * 8,
                         window_seconds=1224.0,
                         updated_at=[T0 + timedelta(seconds=i) for i in range(20)])
    assert bad.saturation_low == pytest.approx(0.6)
    assert bad.regime == "pinned_max"


# --------------------------------------------------------------------------
# relative readings
# --------------------------------------------------------------------------

def test_drift_is_computed_exactly_not_merely_non_none() -> None:
    """The previous suite asserted only `r.drift is not None`, so hardcoding
    `drift = 0.0` -- the exact fabrication this module forbids -- survived, as
    did dropping the abs() and dropping the division.

    Hand-derived: window is a flat 0.9; baseline is 0.00, 0.01, ... 0.99.
      fmean(baseline) = 0.495
      pstdev(baseline) = sqrt((100^2 - 1)/12) * 0.01 = 0.2886607004772212
      drift = |0.9 - 0.495| / 0.2886607004772212 = 1.40303130745004
      level_percentile = #{b < 0.9} / 100 = 90/100 = 0.90
    """
    baseline = [i / 100 for i in range(100)]
    r = channel_regime("c", _flat(0.9), window_seconds=1.0, baseline=baseline)

    assert r.drift == pytest.approx(1.40303130745004, abs=1e-12)
    assert r.level_percentile == pytest.approx(0.90)
    assert r.dispersion_ratio == pytest.approx(0.0)


def test_drift_is_a_magnitude_and_stays_positive_below_the_baseline() -> None:
    """Pins the abs(). Both previous drift fixtures sat ABOVE the baseline
    mean, so dropping abs() changed nothing and survived mutation.

    Hand-derived: window flat 0.1, baseline mean 0.495, so the raw difference
    is NEGATIVE (-0.395).
      drift = |0.1 - 0.495| / 0.2886607004772212 = 1.36838855911794
    """
    baseline = [i / 100 for i in range(100)]
    r = channel_regime("c", _flat(0.1), window_seconds=1.0, baseline=baseline)
    assert r.drift == pytest.approx(1.36838855911794, abs=1e-12)
    assert r.drift > 0


def test_dispersion_ratio_is_a_ratio_not_a_copy() -> None:
    """The old fixture used a flat window, so dispersion == 0 and
    `dispersion_ratio = dispersion` (no division) was indistinguishable.

    Hand-derived: window alternates 0.5 +/- 0.05 -> pstdev 0.05.
    Baseline 0.00..0.99 -> pstdev 0.2886607004772212.
      ratio = 0.05 / 0.2886607004772212 = 0.17321374166049874

    NOTE: the first draft of this assertion said 0.1732051, which is
    sqrt(3)/10 -- pattern-matched instead of divided. It was caught only
    because the assertion is exact; an `is not None` check would have taken it.
    """
    baseline = [i / 100 for i in range(100)]
    r = channel_regime("c", [0.45, 0.55] * 10, window_seconds=1.0, baseline=baseline)
    assert r.dispersion == pytest.approx(0.05)
    assert r.dispersion_ratio == pytest.approx(0.17321374166049874, abs=1e-12)
    assert r.dispersion_ratio != pytest.approx(r.dispersion)


def test_relative_readings_are_none_without_a_baseline() -> None:
    r = channel_regime("c", [0.1, 0.2] * 10, window_seconds=1.0)
    assert r.drift is None
    assert r.level_percentile is None
    assert r.dispersion_ratio is None


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------

def test_empty_series_reports_no_numbers_at_all() -> None:
    """The previous version of this test asserted sample_count and labels but
    never the numeric fields, so mutating the empty-path level to 99.0 or
    saturation to 1.0 survived -- despite the test's own name."""
    r = channel_regime("absent", [], window_seconds=1224.0)
    assert isinstance(r, ChannelRegime)
    assert r.sample_count == 0
    assert r.level == 0.0
    assert r.dispersion is None
    assert r.saturation_low == 0.0
    assert r.saturation_high == 0.0
    assert r.drift is None
    assert r.refresh_state == "insufficient_samples"
    assert r.regime == "insufficient_samples"


def test_single_sample_reports_no_dispersion() -> None:
    """One sample cannot be "perfectly steady"."""
    r = channel_regime("c", [0.5], window_seconds=1.0)
    assert r.sample_count == 1
    assert r.dispersion is None
    assert r.refresh_state == "insufficient_samples"


def test_window_seconds_is_carried_not_inferred() -> None:
    """The unit trap: 600 ticks is ~20 minutes at the live 2.0417s cadence, and
    nothing recorded that."""
    r = channel_regime("c", [0.1, 0.2] * 10, window_seconds=1224.0)
    assert r.window_seconds == 1224.0
    assert r.sample_count == 20
