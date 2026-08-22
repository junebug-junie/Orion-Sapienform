"""Spending a finite allowance on the actions worth running.

The allocator had one hard problem: every action Orion has measures ~zero,
so ranking by value-per-cost ranks noise and an absolute bar on value refuses
everything. The resolution is that expected free energy has two terms and only
the pragmatic one was being used. These tests pin the epistemic one, and pin
the gates that keep it from becoming a licence to do anything at all.
"""

from __future__ import annotations

import math
import random

import pytest

from orion.autonomy.allocator import (
    HARM_CONFIDENCE_SIGMAS,
    Candidate,
    allocate,
    expected_information_gain_nats,
)
from orion.autonomy.prediction import (
    DEFAULT_OBSERVATION_VARIANCE,
    EffectPosterior,
    score_observation,
)


class TestExpectedInformationGain:
    def test_the_closed_form_is_what_it_claims(self):
        # 0.5 * ln(1 + 0.25/0.04) = 0.5 * ln(7.25) = 0.5 * 1.9810015 = 0.9905007
        assert expected_information_gain_nats(0.25) == pytest.approx(0.9905007, abs=1e-6)
        # 0.5 * ln(1 + 0.001/0.04) = 0.5 * ln(1.025) = 0.5 * 0.0246926 = 0.0123463
        assert expected_information_gain_nats(0.001) == pytest.approx(0.0123463, abs=1e-6)

    def test_it_matches_the_actual_update_by_simulation(self):
        """The derivation, checked against the real scorer rather than trusted.

        E[KL] over the predictive distribution collapses to
        0.5*ln(1 + sigma^2/tau^2) because the mean-shift term integrates to
        exactly sigma^2/(sigma^2+tau^2) and the variance term to
        tau^2/(sigma^2+tau^2), which sum to 1 and cancel the -1.
        """
        random.seed(11)
        for variance in (0.25, 0.05, 0.01, 0.001):
            prior = EffectPosterior(mean=0.0, variance=variance, n=5)
            sd = math.sqrt(variance + DEFAULT_OBSERVATION_VARIANCE)
            trials = 20000
            observed = sum(
                score_observation(prior, random.gauss(0.0, sd))[1] for _ in range(trials)
            ) / trials
            assert observed == pytest.approx(
                expected_information_gain_nats(variance), rel=0.02
            )

    def test_a_cold_action_is_worth_eighty_times_a_known_one(self):
        """Redundancy stops paying with no anti-repetition rule written
        anywhere -- the property that made Bayesian surprise the right
        currency at the start of this arc."""
        cold = expected_information_gain_nats(0.25)
        known = expected_information_gain_nats(0.001)
        assert cold / known == pytest.approx(80.2, abs=0.5)

    def test_certainty_is_worth_nothing(self):
        assert expected_information_gain_nats(0.0) == 0.0

    def test_degenerate_inputs_are_refused(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                expected_information_gain_nats(bad)
        with pytest.raises(ValueError):
            expected_information_gain_nats(0.1, observation_variance=0.0)


def _c(did, var=0.25, cost=5.0, contrast=None, sd=None, direction=None):
    return Candidate(
        dispatch_id=did,
        dispatch_kind="inspect",
        target_id="t",
        posterior_variance=var,
        cost_sec=cost,
        contrast=contrast,
        contrast_sd=sd,
        claimed_direction=direction,
    )


class TestTheHarmGate:
    """An action that cannot lose is not competing."""

    def test_confidently_opposite_to_its_claim_is_refused(self):
        """The live prune: claims `decrease`, measured contrast +0.037 with
        sd 0.0037 -- ten sigma on the wrong side of zero."""
        c = _c("prune", var=0.001, contrast=0.037, sd=0.0037, direction="decrease")
        assert c.confidently_harmful
        result = allocate([c], allowance_sec=100.0, min_nats_per_sec=0.0)
        assert result.admitted == ()
        assert result.refused[0][1] == "confidently_harmful"

    def test_harm_beats_information(self):
        """Maximally informative AND confidently harmful -> still refused.
        Learning is not a reason to do damage."""
        c = _c("bad", var=0.25, contrast=0.5, sd=0.01, direction="decrease")
        assert c.expected_nats > 0.9
        assert allocate([c], allowance_sec=100.0, min_nats_per_sec=0.0).admitted == ()

    def test_a_wide_error_bar_is_not_confidently_anything(self):
        """Every live contrast currently sits inside its own error bar. The
        gate must refuse almost nothing today and bite as evidence accrues."""
        c = _c("noisy", contrast=0.037, sd=0.05, direction="decrease")
        assert not c.confidently_harmful

    def test_the_boundary_is_exactly_two_sigma(self):
        assert HARM_CONFIDENCE_SIGMAS == 2.0
        just_inside = _c("a", contrast=0.020, sd=0.010, direction="decrease")
        just_outside = _c("b", contrast=0.021, sd=0.010, direction="decrease")
        assert not just_inside.confidently_harmful  # 0.020 - 0.020 = 0.0, not > 0
        assert just_outside.confidently_harmful     # 0.021 - 0.020 > 0

    def test_direction_matters(self):
        assert _c("x", contrast=-0.5, sd=0.01, direction="increase").confidently_harmful
        assert not _c("y", contrast=-0.5, sd=0.01, direction="decrease").confidently_harmful

    def test_no_control_coverage_is_not_evidence_of_safety(self):
        """None means unknown, and unknown cannot be gated on -- but it must
        not be silently read as harmless either. It simply does not fire."""
        assert not _c("unmeasured", contrast=None, sd=None, direction="decrease").confidently_harmful

    def test_a_no_change_claim_cannot_be_harmful_in_this_sense(self):
        assert not _c("z", contrast=0.9, sd=0.001, direction="no_change").confidently_harmful


class TestTheBarIsAbsolute:
    def test_none_of_these_were_worth_doing_is_expressible(self):
        """The sentence a relative ranking can never say. Percentages sum to
        100% however worthless the set."""
        candidates = [_c(f"d{i}", var=0.0001) for i in range(5)]
        result = allocate(candidates, allowance_sec=10_000.0, min_nats_per_sec=0.1)
        assert result.admitted == ()
        assert result.refusals_by_reason() == {"below_information_floor": 5}
        assert result.spent_sec == 0.0

    def test_the_floor_applies_even_with_the_allowance_untouched(self):
        """Being below the bar is not a budget question. A near-infinite
        allowance must not admit an action that teaches nothing.

        Hand-computed: var 0.0001 -> 0.5*ln(1 + 0.0001/0.04) = 0.00124845
        nats; over 5.0s that is 0.00024969 nats/sec, far under the 0.1 floor.
        (The first version of this test used cost=0.001s, which makes the
        SAME action worth 1.248 nats/sec and clears a 1.0 floor comfortably --
        the assertion was wrong, not the allocator.)"""
        c = _c("uninformative", var=0.0001, cost=5.0)
        assert c.nats_per_sec == pytest.approx(0.00024969, abs=1e-8)
        result = allocate([c], allowance_sec=1_000_000.0, min_nats_per_sec=0.1)
        assert result.admitted == ()
        assert result.refused[0][1] == "below_information_floor"

    def test_a_worthwhile_action_clears_it(self):
        result = allocate([_c("good", var=0.25)], allowance_sec=100.0, min_nats_per_sec=0.1)
        assert [c.dispatch_id for c in result.admitted] == ["good"]


class TestSpendingTheAllowance:
    def test_ranked_by_information_per_second_not_raw_information(self):
        """A cheap mildly-informative action beats an expensive very
        informative one. That is the whole point of a denominator."""
        expensive = _c("expensive", var=0.25, cost=60.0)   # 0.9905/60 = 0.0165
        cheap = _c("cheap", var=0.05, cost=2.0)            # 0.4055/2  = 0.2027
        result = allocate([expensive, cheap], allowance_sec=1000.0, min_nats_per_sec=0.0)
        assert [c.dispatch_id for c in result.admitted] == ["cheap", "expensive"]

    def test_the_allowance_actually_runs_out(self):
        candidates = [_c(f"d{i}", var=0.25, cost=4.0) for i in range(10)]
        result = allocate(candidates, allowance_sec=10.0, min_nats_per_sec=0.0)
        assert len(result.admitted) == 2
        assert result.spent_sec == 8.0
        assert result.refusals_by_reason() == {"allowance_exhausted": 8}

    def test_an_untimed_action_is_refused_not_admitted_free(self):
        """An unmeasured cost is the one that could be enormous. Admitting it
        would let the most expensive actions bypass the budget entirely."""
        result = allocate([_c("untimed", cost=None)], allowance_sec=100.0, min_nats_per_sec=0.0)
        assert result.admitted == ()
        assert result.refused[0][1] == "no_cost_estimate"

    def test_zero_cost_is_treated_as_unmeasured_not_as_free(self):
        result = allocate([_c("zero", cost=0.0)], allowance_sec=100.0, min_nats_per_sec=0.0)
        assert result.refused[0][1] == "no_cost_estimate"

    def test_allocation_is_deterministic_for_equal_candidates(self):
        """An allocator that reshuffles ties makes its own logs
        unreproducible."""
        a = [_c("b"), _c("a"), _c("c")]
        first = allocate(a, allowance_sec=10.0, min_nats_per_sec=0.0)
        second = allocate(list(reversed(a)), allowance_sec=10.0, min_nats_per_sec=0.0)
        assert [c.dispatch_id for c in first.admitted] == [c.dispatch_id for c in second.admitted]
        assert [c.dispatch_id for c in first.admitted] == ["a", "b"]

    def test_nats_bought_is_reported(self):
        result = allocate([_c("x", var=0.25)], allowance_sec=100.0, min_nats_per_sec=0.0)
        assert result.admitted_nats == pytest.approx(0.9905007, abs=1e-6)


class TestUnmeasurableIsNotUninformed:
    """The defect that inverted the whole allocator, and the test shape that
    would have caught it.

    Every other test here constructs a Candidate directly with a scalar
    variance, so none of them exercised the reconstruction in
    worker._log_allocator_preview -- which is where the bug lived. Live
    evidence before the fix: across 57 consecutive previews EVERY admitted
    candidate scored exactly 0.9905007 nats (the cold-start default) and every
    measured action was refused below the floor.
    """

    def test_an_action_that_declares_no_signal_scores_zero(self):
        """It is UNMEASURABLE, not maximally uncertain. It can never acquire
        a posterior at all -- outcome_resolution skips it with
        `no_declared_signal` -- so there is no belief for an observation to
        update and running it buys exactly nothing."""
        c = Candidate("undeclared", "inspect", "orion/bus/channels.yaml",
                      posterior_variance=None, cost_sec=4.7)
        assert not c.measurable
        assert c.expected_nats == 0.0
        assert c.nats_per_sec == 0.0

    def test_it_is_refused_with_its_own_reason(self):
        """Not folded into below_information_floor: 'we have learned what this
        does' and 'this can never be scored' need different fixes."""
        c = Candidate("undeclared", "inspect", "t", posterior_variance=None, cost_sec=4.7)
        result = allocate([c], allowance_sec=1000.0, min_nats_per_sec=0.0)
        assert result.admitted == ()
        assert result.refused[0][1] == "unmeasurable"

    def test_it_cannot_outrank_a_measured_action(self):
        """The inversion, stated as an assertion. Before the fix the
        undeclared action scored 0.9905/4.7 = 0.2107 and the measured one
        0.0065/0.71 = 0.0091 -- a 23x advantage for declaring nothing, which
        made omitting expected_effect the dominant strategy."""
        undeclared = Candidate("undeclared", "inspect", "a",
                               posterior_variance=None, cost_sec=4.7)
        measured = Candidate("measured", "maintain", "b",
                             posterior_variance=0.0005, cost_sec=0.71)
        result = allocate([undeclared, measured], allowance_sec=1000.0, min_nats_per_sec=0.0)
        assert [c.dispatch_id for c in result.admitted] == ["measured"]

    def test_a_genuinely_new_but_measurable_action_still_scores_high(self):
        """The fix must not throw away real cold starts -- an action that
        declares a signal and has no history yet is the most informative
        thing available, and that is still true."""
        from orion.autonomy.prediction import DEFAULT_PRIOR_VARIANCE

        c = Candidate("new", "inspect", "t",
                      posterior_variance=DEFAULT_PRIOR_VARIANCE, cost_sec=5.0)
        assert c.measurable
        assert c.expected_nats == pytest.approx(0.9905007, abs=1e-6)


class TestPerBinVarianceNotPooled:
    def test_the_weighted_gain_uses_each_bin_not_a_pooled_mean(self):
        from orion.autonomy.allocator import expected_information_gain_across_bins

        # Two bins, equal volume: variances 0.01 and 0.001.
        # 0.5*ln(1+0.01/0.04)  = 0.5*ln(1.25)  = 0.1115718
        # 0.5*ln(1+0.001/0.04) = 0.5*ln(1.025) = 0.0123457
        # mean = 0.0619587
        assert expected_information_gain_across_bins(
            [(0.01, 100), (0.001, 100)]
        ) == pytest.approx(0.0619587, abs=1e-6)

    def test_volume_weighting_follows_where_the_action_actually_runs(self):
        from orion.autonomy.allocator import expected_information_gain_across_bins

        # 900 observations in the well-known bin, 100 in the uncertain one.
        # 0.9*0.0123457 + 0.1*0.1115718 = 0.0111111 + 0.0111572 = 0.0222683
        assert expected_information_gain_across_bins(
            [(0.001, 900), (0.01, 100)]
        ) == pytest.approx(0.0222683, abs=1e-6)

    def test_empty_cells_are_absent_not_zero(self):
        from orion.autonomy.allocator import expected_information_gain_across_bins

        assert expected_information_gain_across_bins([]) is None
        assert expected_information_gain_across_bins([(0.01, 0)]) is None

    def test_pooling_across_bins_understates_uncertainty(self):
        """Why the pooled figure was wrong, as arithmetic.

        pooled_treated_mean computes sum((n_b/N)^2 * var_b) -- the sampling
        variance of a MEAN across bins, which divides by roughly the bin
        count. So an action read as better-known the more distinct conditions
        it had run under. Live this scored maintain/host:docker_containers at
        0.0091 nats/s (refused) against 0.192 per-bin (top of the whole set).
        """
        from orion.autonomy.allocator import (
            expected_information_gain_across_bins,
            expected_information_gain_nats,
        )

        cells = [(0.01, 100), (0.01, 100), (0.01, 100), (0.01, 100)]
        pooled_variance = sum((100 / 400) ** 2 * 0.01 for _ in cells)  # 0.0025
        assert expected_information_gain_nats(pooled_variance) < (
            expected_information_gain_across_bins(cells) / 3
        )
