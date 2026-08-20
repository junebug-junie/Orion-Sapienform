"""Unit tests for Candidate A (`orion/attention/field_attention/
candidate_precision_weighted.py`) -- precision-weighted prediction-error salience,
Feldman & Friston 2010. Shadow-only pure function; see the module docstring and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md` for the design record. No I/O, no fixtures needed."""

from __future__ import annotations

import pytest

from orion.attention.field_attention.candidate_precision_weighted import (
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    PRECISION_VARIANCE_FLOOR,
    PrecisionEwmaBaseline,
    advance_precision_baseline,
    cross_domain_variance_floor,
    normalize_across_targets,
    precision_weighted_salience,
    precision_weighted_salience_from_baseline,
)


def test_empty_history_yields_zero_everything() -> None:
    result = precision_weighted_salience([])
    assert result.salience == 0.0
    assert result.precision == 0.0
    assert result.variance == 0.0
    assert result.current_error == 0.0
    assert result.n_samples == 0
    assert result.variance_floored is False


def test_single_sample_history_floors_variance_and_uses_that_sample_as_current() -> None:
    """One real observation has zero variance by definition -- treated as the
    smallest-possible-sample-size floor case, not a separate undefined case."""
    result = precision_weighted_salience([0.05])
    assert result.n_samples == 1
    assert result.current_error == pytest.approx(0.05)
    assert result.variance == 0.0
    assert result.variance_floored is True
    assert result.precision == pytest.approx(1.0 / PRECISION_VARIANCE_FLOOR)
    assert result.salience == pytest.approx(0.05 / PRECISION_VARIANCE_FLOOR)


def test_near_zero_variance_edge_case_floors_precision_not_diverges() -> None:
    """A target whose error has been almost perfectly constant -- the concrete
    'variance-near-zero instability risk' named in the design doc. Must not raise
    ZeroDivisionError or produce +inf."""
    history = [0.03, 0.03, 0.03, 0.03, 0.0300001]
    result = precision_weighted_salience(history)
    assert result.n_samples == 5
    assert result.variance == pytest.approx(0.0, abs=1e-9)
    assert result.variance_floored is True
    assert result.precision == pytest.approx(1.0 / PRECISION_VARIANCE_FLOOR)
    assert result.salience == pytest.approx(
        (1.0 / PRECISION_VARIANCE_FLOOR) * abs(0.0300001)
    )
    import math

    assert math.isfinite(result.precision)
    assert math.isfinite(result.salience)


def test_healthy_variance_produces_finite_bounded_precision() -> None:
    """A real, non-degenerate history (modeled on the live biometrics spread found
    2026-07-21: real values roughly in [0.0, 0.17]) should NOT trip the variance
    floor, and should produce a precision that is large but not floor-ceiling-pinned."""
    history = [0.001, 0.08, 0.003, 0.12, 0.0005, 0.09, 0.002, 0.15]
    result = precision_weighted_salience(history)
    assert result.n_samples == 8
    assert result.variance > PRECISION_VARIANCE_FLOOR
    assert result.variance_floored is False
    assert result.current_error == pytest.approx(0.15)
    expected_precision = 1.0 / result.variance
    assert result.precision == pytest.approx(expected_precision)
    assert result.salience == pytest.approx(expected_precision * 0.15)
    # Sanity: precision should be well below the floor-ceiling for a healthy series.
    assert result.precision < 1.0 / PRECISION_VARIANCE_FLOOR


def test_current_error_is_the_last_element_not_the_max_or_mean() -> None:
    history = [0.5, 0.5, 0.01]
    result = precision_weighted_salience(history)
    assert result.current_error == pytest.approx(0.01)


def test_zero_current_error_yields_zero_salience_regardless_of_precision() -> None:
    history = [0.02, 0.03, 0.0]
    result = precision_weighted_salience(history)
    assert result.current_error == 0.0
    assert result.salience == 0.0
    assert result.precision > 0.0  # precision itself is unaffected by the current value


def test_negative_values_are_handled_via_absolute_value_of_current_error() -> None:
    """prediction_error instruments in orion/substrate/prediction_error.py are all
    non-negative by construction (min(1.0, mean/threshold) or a [0,1] mismatch rate),
    but this function does not assume that -- abs() is applied defensively."""
    history = [0.1, -0.2, 0.15, -0.3]
    result = precision_weighted_salience(history)
    assert result.current_error == pytest.approx(-0.3)
    assert result.salience == pytest.approx(result.precision * 0.3)
    assert result.salience >= 0.0


def test_result_is_a_frozen_dataclass_not_mutable() -> None:
    result = precision_weighted_salience([0.1, 0.2])
    with pytest.raises(Exception):
        result.salience = 999.0  # type: ignore[misc]


# -- normalize_across_targets --------------------------------------------------
# Added in review (2026-07-22): precision_weighted_salience()'s raw output is
# unbounded and dominated by each target's own historical variance scale, which
# is not a valid drop-in for FieldAttentionTargetV1.salience_score (schema-bound
# to [0,1]) and is not meaningfully comparable across targets without this step.


def test_normalize_across_targets_empty_input() -> None:
    assert normalize_across_targets({}) == {}


def test_normalize_across_targets_maps_min_to_zero_and_max_to_one() -> None:
    result = normalize_across_targets({"a": 10.0, "b": 400.0, "c": 57100.0})
    assert result["c"] == pytest.approx(1.0)
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx((400.0 - 10.0) / (57100.0 - 10.0))


def test_normalize_across_targets_preserves_relative_rank() -> None:
    raw = {"low": 5.0, "mid": 500.0, "high": 50000.0}
    result = normalize_across_targets(raw)
    assert result["low"] < result["mid"] < result["high"]


def test_normalize_across_targets_output_always_in_unit_interval() -> None:
    raw = {"a": 0.0, "b": 3.3, "c": 1e6, "d": 42.0}
    result = normalize_across_targets(raw)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_normalize_across_targets_all_equal_scores_get_one_not_zero() -> None:
    """A real tie must not be misrepresented as 'nothing here matters' -- there is
    no basis to floor a genuine tie to 0.0."""
    result = normalize_across_targets({"a": 42.0, "b": 42.0, "c": 42.0})
    assert result == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_normalize_across_targets_single_target_gets_one() -> None:
    """Degenerate case of the tie rule: the only real competitor this tick gets
    maximal (not zero, not arbitrary) attention -- it's the only real candidate."""
    assert normalize_across_targets({"only": 12345.6}) == {"only": 1.0}


def test_normalize_across_targets_near_equal_scores_within_epsilon_treated_as_tie() -> None:
    result = normalize_across_targets({"a": 1.0, "b": 1.0 + 1e-13})
    assert result == {"a": 1.0, "b": 1.0}


def test_normalize_across_targets_does_not_mutate_input() -> None:
    raw = {"a": 1.0, "b": 2.0}
    normalize_across_targets(raw)
    assert raw == {"a": 1.0, "b": 2.0}


def test_normalize_across_targets_end_to_end_with_real_precision_weighted_salience() -> None:
    """Integration-shaped: run two targets through the real
    precision_weighted_salience() pure function, then normalize the raw results --
    exercises both functions together the way a real caller would, not in isolation."""
    quiet_target = precision_weighted_salience([0.03, 0.03, 0.03, 0.03, 0.031])
    noisy_target = precision_weighted_salience([0.01, 0.08, 0.02, 0.12, 0.15])
    raw = {"quiet": quiet_target.salience, "noisy": noisy_target.salience}
    normalized = normalize_across_targets(raw)
    assert set(normalized) == {"quiet", "noisy"}
    assert all(0.0 <= v <= 1.0 for v in normalized.values())


# -- advance_precision_baseline / precision_weighted_salience_from_baseline ---
# 2026-07-30 fix (Sentience Striving Program officer review, see
# orion/sentience_striving_program/README.md §12): the live incident this section
# regression-tests is that the OLD path (precision_weighted_salience() fed by a
# freshly re-queried ~30-minute rolling window every tick) let a target's real
# "n_samples" silently reset to whatever currently survived that window instead of
# accumulating. The persisted EWMA baseline below is what actually fixes that.


def test_advance_precision_baseline_empty_new_values_returns_same_object() -> None:
    """A tick with no new real receipts must be a true no-op -- the caller uses
    object identity to decide whether a DB write is even needed."""
    baseline = PrecisionEwmaBaseline(ewma=0.1, variance=0.01, observation_count=5, last_value=0.1)
    result = advance_precision_baseline(baseline, [], alpha=0.2, min_variance=1e-5)
    assert result is baseline


def test_advance_precision_baseline_cold_start_first_observation() -> None:
    baseline = PrecisionEwmaBaseline()
    result = advance_precision_baseline(baseline, [0.05], alpha=0.2, min_variance=1e-5)
    assert result.observation_count == 1
    assert result.last_value == pytest.approx(0.05)
    assert result.ewma == pytest.approx(0.05)  # compute_ewma_update's own first-obs contract
    assert result.variance == 0.0  # no fluctuation observed yet


def test_advance_precision_baseline_observation_count_is_cumulative_not_windowed() -> None:
    """The exact property that fixes the live incident: folding N real values in,
    one at a time across separate calls (simulating separate ticks), accumulates
    observation_count -- it never resets just because a caller's fetch this tick
    was small."""
    baseline = PrecisionEwmaBaseline()
    for value in [0.01, 0.02]:
        baseline = advance_precision_baseline(baseline, [value], alpha=0.2, min_variance=1e-5)
    assert baseline.observation_count == 2
    # A later tick that fetches a large batch of new real receipts (e.g. after a
    # backlog) keeps accumulating on top, not starting over.
    baseline = advance_precision_baseline(
        baseline, [0.03, 0.04, 0.05], alpha=0.2, min_variance=1e-5
    )
    assert baseline.observation_count == 5
    assert baseline.last_value == pytest.approx(0.05)


def test_advance_precision_baseline_multiple_new_values_processed_in_order() -> None:
    baseline = PrecisionEwmaBaseline()
    baseline = advance_precision_baseline(
        baseline, [0.1, 0.2, 0.9], alpha=0.2, min_variance=1e-5
    )
    assert baseline.observation_count == 3
    assert baseline.last_value == pytest.approx(0.9)  # the most recent, not max/mean


def test_precision_weighted_salience_from_baseline_cold_start_is_empty() -> None:
    result = precision_weighted_salience_from_baseline(
        PrecisionEwmaBaseline(), min_variance=1e-5
    )
    assert result.n_samples == 0
    assert result.salience == 0.0
    assert result.precision == 0.0


def test_precision_weighted_salience_from_baseline_uses_observation_count_as_n_samples() -> None:
    baseline = PrecisionEwmaBaseline(ewma=0.1, variance=0.02, observation_count=37, last_value=0.3)
    result = precision_weighted_salience_from_baseline(baseline, min_variance=1e-5)
    assert result.n_samples == 37
    assert result.current_error == pytest.approx(0.3)
    assert result.precision == pytest.approx(1.0 / 0.02)
    assert result.salience == pytest.approx((1.0 / 0.02) * 0.3)


def test_precision_weighted_salience_from_baseline_floors_near_zero_variance() -> None:
    baseline = PrecisionEwmaBaseline(ewma=0.03, variance=1e-9, observation_count=10, last_value=0.031)
    result = precision_weighted_salience_from_baseline(baseline, min_variance=1e-5)
    assert result.variance_floored is True
    assert result.precision == pytest.approx(1.0 / 1e-5)
    import math

    assert math.isfinite(result.precision)


def test_precision_weighted_salience_from_baseline_does_not_falsely_pin_at_low_n() -> None:
    """This is the concrete regression for the live incident: a baseline that has
    only accumulated 2 real observations must still honestly report n_samples=2
    (not silently inflate it), because it is the caller's/consumer's job
    (goal_provenance.py's confidence gate) to distinguish 'real but thin' from
    'real and trustworthy' -- this function's job is only to report the truth of
    what the baseline has actually seen so far."""
    baseline = PrecisionEwmaBaseline()
    for value in [0.0028, 0.0053]:  # the real live chat_session pair, 2026-07-30
        baseline = advance_precision_baseline(baseline, [value], alpha=0.2, min_variance=1e-5)
    result = precision_weighted_salience_from_baseline(baseline, min_variance=1e-5)
    assert result.n_samples == 2  # honest, not artificially inflated or reset


def test_advance_and_score_synthetic_regime_shift_no_permanent_floor_pinning() -> None:
    """A synthetic series with a real regime shift (long calm period, then a real
    sustained change) should NOT stay permanently pinned at the variance floor once
    enough real observations have accumulated -- confirms the EWMA baseline can
    genuinely reflect a live process's real statistics over time, not just freeze
    at whatever its first few samples looked like."""
    calm = [0.05] * 15
    shifted = [0.05, 0.4, 0.05, 0.45, 0.05, 0.5, 0.05, 0.42]
    baseline = PrecisionEwmaBaseline()
    for value in calm:
        baseline = advance_precision_baseline(
            baseline, [value], alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
            min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
        )
    calm_result = precision_weighted_salience_from_baseline(
        baseline, min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
    )
    assert calm_result.variance_floored is True  # genuinely constant so far -- real, not a bug

    for value in shifted:
        baseline = advance_precision_baseline(
            baseline, [value], alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
            min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
        )
    shifted_result = precision_weighted_salience_from_baseline(
        baseline, min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
    )
    assert shifted_result.variance_floored is False
    assert shifted_result.variance > calm_result.variance
    assert baseline.observation_count == 15 + 8  # real cumulative count, not reset


class TestCrossDomainVarianceFloor:
    """2026-08-20 fix, Sentience Striving Program item 4: replaces the single
    global `NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE` floor with one derived
    live from a target's real competing siblings this tick -- see
    `cross_domain_variance_floor`'s own docstring for the live incident
    (`node:substrate.route` winning goal-provenance dominance by construction)
    this fixes. Uses real live-observed magnitudes from that investigation
    (2026-08-20, `substrate_node_prediction_error_baseline`) as fixtures, not
    synthetic round numbers, so the test pins the actual bug rather than an
    idealized version of it."""

    def _baseline(self, *, variance: float, observation_count: int = 1000) -> PrecisionEwmaBaseline:
        return PrecisionEwmaBaseline(
            ewma=0.1, variance=variance, observation_count=observation_count, last_value=0.1
        )

    def test_reproduces_the_live_incident_route_no_longer_dwarfs_everyone(self) -> None:
        """The exact live values pulled 2026-08-20: route's real variance had
        underflowed to ~2.9e-39 while its four siblings sat at 0.0127-0.193.
        Under the old fixed 1e-5 floor, route's precision (1/1e-5=100,000) beat
        every sibling's organic precision (~5-79) by 1,000x+. Under the new
        floor, route's effective floor should land at the siblings' own median
        variance instead, bringing its precision back into the same order of
        magnitude as its real competitors."""
        baselines = {
            "node:substrate.route": self._baseline(variance=2.9e-39),
            "node:substrate.biometrics": self._baseline(variance=0.0135),
            "node:substrate.bus_synaptic": self._baseline(variance=0.0127),
            "node:substrate.execution": self._baseline(variance=0.117),
            "node:substrate.chat": self._baseline(variance=0.193),
        }
        floor = cross_domain_variance_floor(
            baselines, "node:substrate.route", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        # median of the OTHER four (0.0127, 0.0135, 0.117, 0.193) = (0.0135+0.117)/2
        assert floor == pytest.approx((0.0135 + 0.117) / 2.0)
        old_precision = 1.0 / NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        new_precision = 1.0 / floor
        assert new_precision < old_precision / 1000  # the dwarfing is actually gone

    def test_excludes_the_target_itself_from_its_own_floor(self) -> None:
        """A target's floor must come from its competitors, not partly from
        itself -- otherwise an unusually-quiet target could marginally suppress
        its own floor."""
        baselines = {
            "a": self._baseline(variance=1e-9),  # would corrupt its own median if included
            "b": self._baseline(variance=0.02),
            "c": self._baseline(variance=0.04),
        }
        floor = cross_domain_variance_floor(
            baselines, "a", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == pytest.approx(0.03)  # median of {0.02, 0.04}, "a" excluded

    def test_falls_back_to_global_constant_with_no_real_competitors(self) -> None:
        """Cold start / every sibling at zero observations: no live data to
        derive a better floor from, so this must fall back to the original
        global constant rather than inventing a value from nothing."""
        baselines = {
            "a": self._baseline(variance=1e-9),
            "b": PrecisionEwmaBaseline(),  # observation_count=0, cold start
        }
        floor = cross_domain_variance_floor(
            baselines, "a", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE

    def test_never_floors_below_the_global_constant(self) -> None:
        """If every real competitor is itself unusually quiet (median below the
        global floor), the global floor still wins -- this function only ever
        raises the effective floor above the global constant, never lowers it."""
        baselines = {
            "a": self._baseline(variance=1e-9),
            "b": self._baseline(variance=1e-8),
            "c": self._baseline(variance=1e-7),
        }
        floor = cross_domain_variance_floor(
            baselines, "a", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE

    def test_odd_number_of_competitors_uses_true_median(self) -> None:
        baselines = {
            "a": self._baseline(variance=1e-9),
            "b": self._baseline(variance=0.01),
            "c": self._baseline(variance=0.05),
            "d": self._baseline(variance=0.09),
        }
        floor = cross_domain_variance_floor(
            baselines, "a", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == pytest.approx(0.05)

    def test_correlated_degeneracy_falls_back_instead_of_reproducing_the_bug(self) -> None:
        """Code review, 2026-08-20: if a MAJORITY of a target's real siblings are
        themselves degenerate (at/below min_variance) at the same tick -- a real
        possibility, not hypothetical: the same near-constant-signal mechanism
        that flattened `node:substrate.route` can flatten any domain during a
        correlated quiet period, e.g. a deploy freeze -- a plain median would
        itself be degenerate, and `max(min_variance, degenerate_median)` would
        reproduce the exact 100,000-precision pathology this function exists to
        fix. 3 of 4 siblings degenerate here (the reviewer's exact scenario):
        without the guard, median of the sorted 4 lands on two degenerate
        values and stays degenerate. With the guard, this must fall back to the
        global constant instead."""
        baselines = {
            "route": self._baseline(variance=2.9e-39),
            "sib1": self._baseline(variance=1e-9),
            "sib2": self._baseline(variance=1e-8),
            "sib3": self._baseline(variance=1e-7),
            "healthy": self._baseline(variance=0.1),  # the lone real one
        }
        floor = cross_domain_variance_floor(
            baselines, "route", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE  # fallback, not a degenerate median

    def test_exact_tie_of_degenerate_and_real_siblings_also_falls_back(self) -> None:
        """A 2-of-4 tie is not a trustworthy majority either -- this must not
        silently trust a coin-flip split between real and degenerate siblings."""
        baselines = {
            "route": self._baseline(variance=2.9e-39),
            "sib1": self._baseline(variance=1e-9),
            "sib2": self._baseline(variance=1e-8),
            "healthy1": self._baseline(variance=0.05),
            "healthy2": self._baseline(variance=0.1),
        }
        floor = cross_domain_variance_floor(
            baselines, "route", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        assert floor == NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE

    def test_strict_majority_of_real_siblings_still_derives_a_floor(self) -> None:
        """3 of 4 siblings real (the mirror of the degenerate-majority case) must
        still derive a live floor, not fall back -- the guard should only trip
        when degenerate siblings are the majority, not whenever any exist."""
        baselines = {
            "route": self._baseline(variance=2.9e-39),
            "sib1": self._baseline(variance=1e-9),  # the lone degenerate one
            "healthy1": self._baseline(variance=0.02),
            "healthy2": self._baseline(variance=0.05),
            "healthy3": self._baseline(variance=0.1),
        }
        floor = cross_domain_variance_floor(
            baselines, "route", min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE
        )
        # median of {1e-9, 0.02, 0.05, 0.1} = (0.02+0.05)/2
        assert floor == pytest.approx((0.02 + 0.05) / 2.0)
