"""The control arm, and the estimator that uses it.

Phase 1's value was `after - before`. Actions fire because a pressure is
high and high pressures fall on their own, so that number measures mean
reversion. These tests pin the replacement: a baseline-matched contrast
against an untreated arm, plus the negative results that keep it honest --
a blocked candidate must not advance the action's own belief, an
operator-review block must not become a control, and no coverage must
produce no number.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.autonomy.contrast import (
    BASELINE_BIN_COUNT,
    CONTROL_MOVE_RATE_ALPHA,
    FROZEN_CONTROL_MAX_MOVE_RATE,
    FROZEN_CONTROL_MIN_N,
    MIN_CONTROL_CELL_N,
    ControlCell,
    baseline_bin,
    contrast,
    pooled_treated_mean,
)
from orion.autonomy.prediction import EffectPosterior
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.schemas.action_prediction import ExpectedEffectV1
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

_DIM_CHANNEL = {
    "execution_pressure": "execution_pressure",
    "reasoning_pressure": "reasoning_pressure",
    "reliability_pressure": "reliability_pressure",
    "resource_pressure": "pressure",
}


def _field(tick_id: str, dimensions: dict[str, float]) -> FieldStateV1:
    channels = {_DIM_CHANNEL[d]: v for d, v in dimensions.items()}
    return FieldStateV1(
        generated_at=NOW,
        tick_id=tick_id,
        node_vectors={"node:test": channels},
        node_vector_updated_at={"node:test": {ch: NOW for ch in channels}},
    )


def _effect(signal="resource_pressure", direction="decrease", predicted=0.0, n=0):
    return ExpectedEffectV1(
        signal_id=signal,
        direction=direction,
        predicted_delta=predicted,
        predictor_variance=0.25,
        predictor_n=n,
        cold_start=(n == 0),
    )


def _dispatched(dispatch_id: str, *, kind="maintain", target="host:docker_images", effect=None):
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id,
        source_decision_id=f"decision:{dispatch_id}",
        source_proposal_id=f"proposal:{dispatch_id}",
        dispatch_status="dispatched",
        dispatch_mode="dispatch_read_only",
        dispatch_kind=kind,
        target_id=target,
        target_kind="host",
        risk_score=0.05,
        confidence_score=0.9,
        dispatched_at=NOW,
        result_ref=f"result:{dispatch_id}",
        expected_effect=effect or _effect(),
    )


def _blocked(dispatch_id: str, *, blocked_by, kind="maintain", target="host:docker_build_cache", effect=None):
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id,
        source_decision_id=f"decision:{dispatch_id}",
        source_proposal_id=f"proposal:{dispatch_id}",
        dispatch_status="blocked",
        dispatch_mode="dispatch_read_only",
        dispatch_kind=kind,
        target_id=target,
        target_kind="host",
        risk_score=0.05,
        confidence_score=0.9,
        blocked_by=list(blocked_by),
        expected_effect=effect or _effect(),
    )


def _frame(dispatched=(), blocked=()):
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test",
        source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test",
        source_field_tick_id="tick:test",
        generated_at=NOW,
        execution_dispatch_policy_id="execution_dispatch_policy.v1",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatched_candidates=list(dispatched),
        blocked_candidates=list(blocked),
        dispatch_count=len(dispatched),
        blocked_count=len(blocked),
    )


class TestBaselineBin:
    def test_edges_are_fixed_and_hand_checkable(self):
        assert baseline_bin(0.0) == 0
        assert baseline_bin(0.0999) == 0
        assert baseline_bin(0.1) == 1
        # The live pin: resource_pressure sat at exactly 0.85 for 15h on
        # 2026-08-21 (a saturated vision channel times a 0.85 edge weight).
        assert baseline_bin(0.85) == 8
        assert baseline_bin(0.999) == BASELINE_BIN_COUNT - 1

    def test_out_of_range_clamps_rather_than_crashing_the_feedback_runtime(self):
        assert baseline_bin(1.0) == 9
        assert baseline_bin(1.7) == 9
        assert baseline_bin(-0.3) == 0

    def test_non_finite_is_refused(self):
        with pytest.raises(ValueError):
            baseline_bin(float("nan"))


class TestTheArmsAreRecordedSeparately:
    def test_dispatched_and_capacity_blocked_both_score_in_one_tick(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                dispatched=[_dispatched("d1")],
                blocked=[_blocked("b1", blocked_by=["max_dispatch_candidates:5"])],
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        by_arm = {r.arm: r for r in res.records}
        assert set(by_arm) == {"dispatched", "capacity_blocked"}
        # Acceptance check 1: same window, same numbers, different arm.
        assert by_arm["dispatched"].baseline == by_arm["capacity_blocked"].baseline
        assert (
            by_arm["dispatched"].observed_after
            == by_arm["capacity_blocked"].observed_after
        )
        assert by_arm["dispatched"].baseline_bin == 8

    def test_within_tick_the_two_arms_are_identical_which_is_why_this_arm_is_not_a_control(self):
        """Documents the finding that redirected the design.

        The spec proposed contrasting a dispatched candidate against a
        capacity-blocked one. The field delta is measured frame-wide, so
        both read the same before and the same after: a within-tick contrast
        is identically zero by construction, and a capacity-blocked record
        is contaminated by whichever siblings DID go out. It is recorded,
        never used as the comparison group.
        """
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                dispatched=[_dispatched("d1")],
                blocked=[_blocked("b1", blocked_by=["max_dispatch_candidates:5"])],
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        deltas = {r.observed_delta for r in res.records}
        assert len(deltas) == 1

    def test_operator_review_block_is_not_a_control(self):
        """Acceptance check 2. Blocked for reasons correlated with the
        action's own content is a worse confounder than the one removed."""
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                dispatched=[_dispatched("d1")],
                blocked=[_blocked("b1", blocked_by=["requires_operator_review"])],
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        assert [r.arm for r in res.records] == ["dispatched"]

    def test_blocked_candidate_does_not_advance_the_actions_own_belief(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                blocked=[_blocked("b1", blocked_by=["max_dispatch_candidates:5"])],
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        assert [r.arm for r in res.records] == ["capacity_blocked"]
        assert res.posteriors == {}

    def test_frame_dispatch_count_records_how_contaminated_the_tick_was(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                dispatched=[_dispatched("d1"), _dispatched("d2", target="host:docker_containers")],
                blocked=[_blocked("b1", blocked_by=["max_dispatch_candidates:5"])],
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        assert {r.frame_dispatch_count for r in res.records} == {2}


class TestTheControlArm:
    def test_an_idle_tick_produces_an_untreated_observation(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            now=NOW,
        )
        by_signal = {o.signal_id: o for o in res.control_observations}
        obs = by_signal["resource_pressure"]
        assert obs.baseline_bin == 8
        assert obs.observed_delta == pytest.approx(-0.45)
        assert obs.moved is True
        assert ("resource_pressure", "no_action", 8) in res.control_posteriors
        # deviation_pressure rides along because orion/field/pressure.py
        # injects it unconditionally from a field attribute that defaults to
        # 0.0 -- it is present-and-0.0, not absent, so _present_pressures'
        # absence guard structurally cannot fire on it. That is exactly the
        # instrument the moved_n guard below exists to catch; the resolver
        # records it honestly rather than filtering it here, because a
        # hand-maintained exclusion list is the thing that goes stale.
        assert "deviation_pressure" in by_signal
        assert res.control_posteriors[
            ("deviation_pressure", "no_action", 0)
        ].moved_n == 0
        assert by_signal["deviation_pressure"].moved is False

    def test_a_tick_that_dispatched_anything_is_not_untreated(self):
        """5 of 16 live templates declare NO signal and are 72% of dispatch
        volume. An undeclared action still acts, so 'nothing claimed this
        signal' is not the same as 'nothing touched it'."""
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                dispatched=[
                    _dispatched("d1")
                ]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85, "execution_pressure": 0.2}),
            field_after=_field("a", {"resource_pressure": 0.40, "execution_pressure": 0.2}),
            now=NOW,
        )
        assert res.control_observations == []
        assert res.control_posteriors == {}

    def test_control_observations_never_zero_fill_an_absent_signal(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"execution_pressure": 0.4}),
            now=NOW,
        )
        signals = {o.signal_id for o in res.control_observations}
        # resource_pressure is in `before` and absent from `after`: it must
        # NOT be scored as a delta of -0.85 against a fabricated 0.0, and it
        # must not be scored as 0.0 either. It is simply not measured.
        assert "resource_pressure" not in signals
        assert "execution_pressure" not in signals


class TestContrastArithmetic:
    """Every expected number below is hand-computed in the comment beside it."""

    TREATED = {
        ("maintain", "host:docker_images", "resource_pressure", 8): EffectPosterior(
            mean=-0.10, variance=0.001, n=300
        ),
        ("maintain", "host:docker_images", "resource_pressure", 5): EffectPosterior(
            mean=-0.02, variance=0.002, n=100
        ),
        ("maintain", "host:docker_images", "resource_pressure", 2): EffectPosterior(
            mean=-0.30, variance=0.003, n=100
        ),
    }
    CONTROL = {
        ("resource_pressure", "no_action", 8): ControlCell(
            EffectPosterior(mean=-0.09, variance=0.0005, n=1000),
            moved_n=900,
            move_rate=0.9,
        ),
        ("resource_pressure", "no_action", 5): ControlCell(
            EffectPosterior(mean=-0.03, variance=0.0004, n=2000),
            moved_n=1800,
            move_rate=0.9,
        ),
    }

    def test_matched_contrast_is_the_weighted_difference(self):
        est = contrast(
            self.TREATED, self.CONTROL, "maintain", "host:docker_images", "resource_pressure"
        )
        assert est is not None
        # covered bins {5,8}; covered volume 300+100 = 400
        # w8 = 0.75, w5 = 0.25
        # 0.75*(-0.10 - -0.09) + 0.25*(-0.02 - -0.03)
        #   = 0.75*(-0.01) + 0.25*(+0.01) = -0.0075 + 0.0025 = -0.005
        assert est.value == pytest.approx(-0.005)
        # 0.75^2*(0.001+0.0005) + 0.25^2*(0.002+0.0004)
        #   = 0.5625*0.0015 + 0.0625*0.0024 = 0.00084375 + 0.00015 = 0.00099375
        assert est.variance == pytest.approx(0.00099375)
        assert est.treated_n == 400
        assert est.control_n == 3000

    def test_the_contrast_is_not_the_raw_number_and_the_gap_is_the_confound(self):
        est = contrast(
            self.TREATED, self.CONTROL, "maintain", "host:docker_images", "resource_pressure"
        )
        pooled = pooled_treated_mean(
            self.TREATED, "maintain", "host:docker_images", "resource_pressure"
        )
        # pooled raw = (300*-0.10 + 100*-0.02 + 100*-0.30)/500 = -62/500 = -0.124
        assert pooled.mean == pytest.approx(-0.124)
        assert pooled.n == 500
        # 25x apart, same data. Phase 1 would have reported -0.124.
        assert abs(pooled.mean) > 20 * abs(est.value)

    def test_uncovered_bins_are_reported_not_backfilled(self):
        est = contrast(
            self.TREATED, self.CONTROL, "maintain", "host:docker_images", "resource_pressure"
        )
        # bin 2 has 100 of 500 treated observations and no control cell.
        assert est.uncovered_weight == pytest.approx(0.2)
        assert [b.baseline_bin for b in est.bins] == [5, 8]

    def test_no_control_coverage_returns_none_not_zero(self):
        """Acceptance check 4. A number here is what would be believed."""
        assert (
            contrast(
                self.TREATED,
                {
                    ("resource_pressure", "no_action", 0): ControlCell(
                        EffectPosterior(0.0, 0.01, 50), moved_n=40, move_rate=0.8
                    )
                },
                "maintain",
                "host:docker_images",
                "resource_pressure",
            )
            is None
        )
        assert (
            contrast(self.TREATED, {}, "maintain", "host:docker_images", "resource_pressure")
            is None
        )

    def test_unknown_action_returns_none(self):
        assert (
            contrast(self.TREATED, self.CONTROL, "maintain", "host:nope", "resource_pressure")
            is None
        )

    def test_randomized_holdback_wins_and_is_never_pooled_with_no_action(self):
        control = dict(self.CONTROL)
        control[("resource_pressure", "randomized_holdback", 8)] = ControlCell(
            EffectPosterior(mean=-0.05, variance=0.001, n=50), moved_n=45, move_rate=0.9
        )
        est = contrast(
            self.TREATED, control, "maintain", "host:docker_images", "resource_pressure"
        )
        assert est.control_arm == "randomized_holdback"
        assert est.evidence_class == "experimental"
        # Only bin 8 is covered by the holdback arm, so w8 = 1.0:
        #   -0.10 - (-0.05) = -0.05.  Emphatically NOT the -0.005 the
        #   no_action arm gives -- merging the two would produce a third
        #   number that is neither and would be labelled the better one.
        assert est.value == pytest.approx(-0.05)
        assert est.control_n == 50

    def test_no_action_arm_is_labelled_quasi_experimental(self):
        est = contrast(
            self.TREATED, self.CONTROL, "maintain", "host:docker_images", "resource_pressure"
        )
        assert est.control_arm == "no_action"
        assert est.evidence_class == "quasi_experimental"

    def test_an_action_can_score_zero_or_positive(self):
        """Phase 1 could not express 'this did nothing' or 'this made it
        worse'. An action that cannot lose is not competing."""
        treated = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(-0.09, 0.001, 100)
        }
        est = contrast(treated, self.CONTROL, "maintain", "t", "resource_pressure")
        assert est.value == pytest.approx(0.0)

        worse = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(0.05, 0.001, 100)
        }
        est2 = contrast(worse, self.CONTROL, "maintain", "t", "resource_pressure")
        assert est2.value > 0


class TestPooledTreatedMean:
    def test_none_when_no_history(self):
        assert pooled_treated_mean({}, "maintain", "t", "resource_pressure") is None

    def test_n_zero_cells_do_not_count_as_history(self):
        cells = {("maintain", "t", "resource_pressure", 3): EffectPosterior(0.0, 0.25, 0)}
        assert pooled_treated_mean(cells, "maintain", "t", "resource_pressure") is None

    def test_variance_of_the_weighted_mean_is_propagated(self):
        """Review finding 21: the variance had no test at all."""
        cells = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(-0.10, 0.001, 300),
            ("maintain", "t", "resource_pressure", 5): EffectPosterior(-0.02, 0.002, 100),
        }
        pooled = pooled_treated_mean(cells, "maintain", "t", "resource_pressure")
        # mean = (300*-0.10 + 100*-0.02)/400 = -32/400 = -0.08
        assert pooled.mean == pytest.approx(-0.08)
        assert pooled.n == 400
        # var = (300/400)^2 * 0.001 + (100/400)^2 * 0.002
        #     = 0.5625*0.001 + 0.0625*0.002 = 0.0005625 + 0.000125 = 0.0006875
        assert pooled.variance == pytest.approx(0.0006875)

    def test_other_actions_and_signals_are_not_pooled_in(self):
        cells = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(-0.10, 0.001, 300),
            ("maintain", "OTHER", "resource_pressure", 8): EffectPosterior(9.0, 0.001, 999),
            ("inspect", "t", "resource_pressure", 8): EffectPosterior(9.0, 0.001, 999),
            ("maintain", "t", "execution_pressure", 8): EffectPosterior(9.0, 0.001, 999),
        }
        pooled = pooled_treated_mean(cells, "maintain", "t", "resource_pressure")
        assert pooled.mean == pytest.approx(-0.10)
        assert pooled.n == 300


class TestExpectedEffectAcrossBins:
    """Review finding 21: multi-bin pooling is the actual behaviour change in
    builder.py and nothing exercised it -- the existing declaration tests only
    bumped their single-bin keys."""

    def _candidate(self, signal, direction, target="capability:orchestration"):
        from orion.schemas.proposal_frame import ProposalCandidateV1

        return ProposalCandidateV1(
            proposal_id="p1",
            proposal_kind="inspect",
            title="t",
            description="d",
            target_id=target,
            target_kind="capability",
            priority_score=0.5,
            urgency_score=0.4,
            confidence_score=0.9,
            risk_score=0.05,
            reversibility_score=1.0,
            proposed_effect="increase_observability",
            required_policy_gate="none",
            expected_signal=signal,
            expected_direction=direction,
        )

    def test_prediction_pools_every_bin_weighted_by_volume(self):
        from orion.execution_dispatch.builder import build_expected_effect

        cells = {
            ("inspect", "capability:orchestration", "execution_pressure", 1): EffectPosterior(
                mean=+0.162, variance=0.002, n=285
            ),
            ("inspect", "capability:orchestration", "execution_pressure", 7): EffectPosterior(
                mean=-0.376, variance=0.001, n=1135
            ),
        }
        effect = build_expected_effect(
            self._candidate("execution_pressure", "decrease"), "inspect", cells
        )
        # (285*0.162 + 1135*-0.376) / 1420 = (46.17 - 426.76)/1420 = -0.26802...
        assert effect.predicted_delta == pytest.approx(-0.2680, abs=1e-4)
        assert effect.predictor_n == 1420
        assert effect.cold_start is False

    def test_a_bin_with_no_history_does_not_dilute_the_prediction(self):
        from orion.execution_dispatch.builder import build_expected_effect

        cells = {
            ("inspect", "capability:orchestration", "execution_pressure", 7): EffectPosterior(
                mean=-0.376, variance=0.001, n=1135
            ),
            ("inspect", "capability:orchestration", "execution_pressure", 2): EffectPosterior(
                mean=0.0, variance=0.25, n=0
            ),
        }
        effect = build_expected_effect(
            self._candidate("execution_pressure", "decrease"), "inspect", cells
        )
        assert effect.predicted_delta == pytest.approx(-0.376)
        assert effect.predictor_n == 1135


class TestTheFrozenInstrumentGuard:
    """A control arm that has stopped seeing its signal move is not calm.

    Live case, 2026-08-21: resource_pressure held exactly 0.85 with stddev
    exactly 0.0 across ~12,000 consecutive frames -- a vision channel
    saturated at 1.0 times a 0.85 topology edge weight, rewritten fresh
    every tick so no staleness check could see it. Contrasting against that
    cell hands the treated arm's whole raw delta back as an effect.
    """

    TREATED = {
        ("maintain", "host:docker_images", "resource_pressure", 8): EffectPosterior(
            mean=-0.148, variance=0.001, n=3426
        )
    }

    def test_a_frozen_control_cell_is_refused_as_coverage(self):
        frozen = {
            ("resource_pressure", "no_action", 8): ControlCell(
                EffectPosterior(mean=0.0, variance=1e-5, n=12000),
                moved_n=0,
                move_rate=0.0,
            )
        }
        assert frozen[("resource_pressure", "no_action", 8)].is_frozen
        assert (
            contrast(
                self.TREATED, frozen, "maintain", "host:docker_images", "resource_pressure"
            )
            is None
        )

    def test_a_cell_that_was_healthy_and_then_froze_is_caught(self):
        """The failure the FIRST version of this guard could not see.

        `moved_n` is a monotone lifetime counter, so `moved_n == 0` can only
        ever catch a channel that was born dead. A channel healthy for a
        month and pinned for an hour keeps a large `moved_n` forever. That is
        the actual live scenario -- resource_pressure ran at 2,600 distinct
        values/day for a week and then pinned -- so the lifetime test was
        structurally blind to it. Caught in review, not by the live replay,
        because the replay happened to build its cells from scratch INSIDE
        the pinned window.
        """
        cell = ControlCell(
            EffectPosterior(mean=-0.05, variance=1e-4, n=50_000),
            moved_n=40_000,
            move_rate=0.8,
        )
        assert not cell.is_frozen

        # Pin it. Each observation is a real fold, exactly as the resolver
        # does it -- no hand-set rate.
        for _ in range(2000):
            cell = cell.observe(cell.posterior, moved=False)

        assert cell.moved_n == 40_000  # lifetime counter: unchanged, still huge
        assert cell.is_frozen  # ...and the windowed rate caught it anyway
        assert cell.move_rate < FROZEN_CONTROL_MAX_MOVE_RATE

    def test_the_freeze_is_detected_inside_a_bounded_number_of_ticks(self):
        """A guard that eventually notices is not a guard. Pin the horizon."""
        cell = ControlCell(EffectPosterior(0.0, 1e-4, 50_000), moved_n=40_000, move_rate=1.0)
        ticks = 0
        while not cell.is_frozen and ticks < 20_000:
            cell = cell.observe(cell.posterior, moved=False)
            ticks += 1
        # From a rate of exactly 1.0 (the least favourable start), an alpha
        # of 1/1000 crosses 0.25 at ln(0.25)/ln(1-alpha) = 1386 observations.
        # At the live untreated rate (~27,000/day) that is ~74 minutes.
        assert ticks == 1386
        assert 1.0 * (1 - CONTROL_MOVE_RATE_ALPHA) ** ticks < FROZEN_CONTROL_MAX_MOVE_RATE

    def test_a_recovering_channel_stops_being_frozen(self):
        """Hysteresis check: the guard must not latch."""
        cell = ControlCell(EffectPosterior(0.0, 1e-4, 50_000), moved_n=1, move_rate=0.0)
        assert cell.is_frozen
        for _ in range(2000):
            cell = cell.observe(cell.posterior, moved=True)
        assert not cell.is_frozen

    def test_without_the_guard_it_would_have_returned_the_raw_delta(self):
        """The number the guard prevents, stated explicitly."""
        alive = {
            ("resource_pressure", "no_action", 8): ControlCell(
                EffectPosterior(mean=0.0, variance=1e-5, n=12000),
                moved_n=11000,
                move_rate=0.9,
            )
        }
        est = contrast(
            self.TREATED, alive, "maintain", "host:docker_images", "resource_pressure"
        )
        assert est.value == pytest.approx(-0.148)

    def test_a_young_cell_has_not_earned_a_verdict(self):
        young = ControlCell(
            EffectPosterior(0.0, 0.01, FROZEN_CONTROL_MIN_N - 1), moved_n=0, move_rate=0.0
        )
        assert not young.is_frozen
        old = ControlCell(
            EffectPosterior(0.0, 0.01, FROZEN_CONTROL_MIN_N), moved_n=0, move_rate=0.0
        )
        assert old.is_frozen

    def test_a_new_cell_is_presumed_alive_not_dead(self):
        assert ControlCell(EffectPosterior.cold()).move_rate == 1.0
        assert not ControlCell(EffectPosterior.cold()).is_frozen

    def test_the_resolver_advances_both_counters(self):
        prior = {
            ("resource_pressure", "no_action", 8): ControlCell(
                EffectPosterior(-0.01, 0.001, 500), moved_n=7, move_rate=0.5
            )
        }
        res = resolve_action_outcomes(
            dispatch_frame=_frame(),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.85}),
            control_priors=prior,
            now=NOW,
        )
        cell = res.control_posteriors[("resource_pressure", "no_action", 8)]
        assert cell.posterior.n == 501
        assert cell.moved_n == 7  # frozen reading: n advances, moved_n does not
        assert cell.move_rate == pytest.approx(0.5 * (1 - CONTROL_MOVE_RATE_ALPHA))

        res2 = resolve_action_outcomes(
            dispatch_frame=_frame(),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.85}),
            field_after=_field("a", {"resource_pressure": 0.40}),
            control_priors=prior,
            now=NOW,
        )
        cell2 = res2.control_posteriors[("resource_pressure", "no_action", 8)]
        assert cell2.moved_n == 8
        assert cell2.move_rate == pytest.approx(
            0.5 + CONTROL_MOVE_RATE_ALPHA * (1.0 - 0.5)
        )


class TestThinControlCoverage:
    """One untreated tick must not be allowed to anchor a bin."""

    TREATED = {
        ("maintain", "t", "resource_pressure", 4): EffectPosterior(-0.2, 0.001, 500),
        ("maintain", "t", "resource_pressure", 5): EffectPosterior(-0.1, 0.001, 500),
    }

    def test_a_thin_bin_is_refused_and_reported(self):
        control = {
            ("resource_pressure", "no_action", 4): ControlCell(
                EffectPosterior(-0.19, 0.001, MIN_CONTROL_CELL_N - 1),
                moved_n=20,
                move_rate=0.9,
            ),
            ("resource_pressure", "no_action", 5): ControlCell(
                EffectPosterior(-0.09, 0.001, 5000), moved_n=4000, move_rate=0.9
            ),
        }
        est = contrast(self.TREATED, control, "maintain", "t", "resource_pressure")
        assert [b.baseline_bin for b in est.bins] == [5]
        assert est.thin_bins == (4,)
        assert est.frozen_bins == ()
        assert est.uncovered_weight == pytest.approx(0.5)

    def test_thin_and_frozen_are_reported_separately(self):
        """'never observed untreated', 'observed 4 times' and 'the instrument
        was pinned' are three different facts and must not collapse."""
        control = {
            ("resource_pressure", "no_action", 4): ControlCell(
                EffectPosterior(0.0, 1e-5, 9000), moved_n=3, move_rate=0.0
            ),
            ("resource_pressure", "no_action", 5): ControlCell(
                EffectPosterior(-0.09, 0.001, 5000), moved_n=4000, move_rate=0.9
            ),
        }
        est = contrast(self.TREATED, control, "maintain", "t", "resource_pressure")
        assert est.frozen_bins == (4,)
        assert est.thin_bins == ()


# =========================================================================
# Regression tests for the review findings on PR #1802, found after merge.
# =========================================================================


def test_report_loads_move_rate_so_the_frozen_guard_can_actually_fire() -> None:
    """HIGH: the report SELECTed moved_n but not move_rate, killing is_frozen.

    `ControlCell.move_rate` defaults to 1.0 ("presumed alive"), and `is_frozen`
    reads ONLY move_rate. Building cells without it made every cell in the
    human-facing report read healthy: `contrast()` accepted pinned control
    cells as coverage, `est.frozen_bins` was always empty, and the
    "FROZEN CONTROL CELLS -- refused as coverage" block could never print.

    Live at the time of the fix, three cells satisfied the freeze condition and
    none were reported -- including `reliability_pressure`, which IS a
    PredictableSignal, so an action claiming it in bin 0 would have had its raw
    delta returned wearing the contrast's name and confidence.

    Asserts against the real script source rather than a copy of the query, so
    dropping the column again re-breaks this test.
    """
    import pathlib
    import re

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts" / "analysis" / "report_action_value.py"
    ).read_text()

    m = re.search(r"CONTROL_CELLS_QUERY\s*=\s*\"\"\"(.*?)\"\"\"", src, re.DOTALL)
    assert m, "CONTROL_CELLS_QUERY not found"
    assert "move_rate" in m.group(1), (
        "CONTROL_CELLS_QUERY must select move_rate -- ControlCell.is_frozen "
        "reads only move_rate and defaults it to 1.0, so omitting it silently "
        "reports every frozen cell as alive"
    )
    assert re.search(r"move_rate\s*=\s*float\(r\[.move_rate.\]\)", src), (
        "the loaded move_rate must actually be passed to ControlCell(...)"
    )


def test_is_frozen_reads_move_rate_not_moved_n() -> None:
    """The property the report has to feed. Hand-picked from live values.

    reliability_pressure bin 0 as measured 2026-08-21: n=2221, moved_n=87,
    move_rate=0.1204. A lifetime counter of 87 looks healthy; the EWMA says
    frozen. That divergence is the whole reason move_rate exists.
    """
    live = ControlCell(
        posterior=EffectPosterior(mean=0.0, variance=0.01, n=2221),
        moved_n=87,
        move_rate=0.1204,
    )
    assert live.is_frozen, "a pinned channel with a healthy-looking moved_n must read frozen"

    # The default that made the bug invisible.
    defaulted = ControlCell(
        posterior=EffectPosterior(mean=0.0, variance=0.01, n=2221),
        moved_n=87,
    )
    assert not defaulted.is_frozen, (
        "sanity: an unspecified move_rate defaults to 1.0 and reads alive -- "
        "this is exactly what the report was doing to every row"
    )
