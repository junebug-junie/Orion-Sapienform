"""Resolution rules for the action-outcome ledger.

The tests that matter most here are the NEGATIVE ones: a resolver that
happily scores an unmeasured signal as 0.0 would manufacture a perfect,
permanent confirmation of every `no_change` claim -- the exact
"schema-valid payload with meaningless content" failure the repo's own
contract bans, and it would look like success on every dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.autonomy.prediction import EffectPosterior
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.schemas.action_prediction import ExpectedEffectV1
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


# Dimension -> the raw field CHANNEL that actually produces it
# (orion.field.pressure.CHANNEL_DIMENSION_MAP). Not identity: resource_pressure
# is merged from a channel named plain `pressure`, and a fixture that wrote a
# `resource_pressure` channel would produce a field with NO resource_pressure
# dimension at all -- which reads as "signal absent" and silently turns any
# test using it into a test of the skip path.
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


def _candidate(
    dispatch_id: str,
    *,
    kind: str = "inspect",
    target_id: str = "capability:orchestration",
    effect: ExpectedEffectV1 | None = None,
) -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id,
        source_decision_id=f"decision:{dispatch_id}",
        source_proposal_id=f"proposal:{dispatch_id}",
        dispatch_status="dispatched",
        dispatch_mode="dispatch_read_only",
        dispatch_kind=kind,
        target_id=target_id,
        target_kind="capability",
        risk_score=0.05,
        confidence_score=0.9,
        # Required by ExecutionDispatchCandidateV1's own validator: a
        # 'dispatched' status must carry evidence a send was attempted.
        dispatched_at=NOW,
        result_ref=f"result:{dispatch_id}",
        expected_effect=effect,
    )


def _frame(candidates: list[ExecutionDispatchCandidateV1]) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test",
        source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test",
        source_field_tick_id="tick:test",
        generated_at=NOW,
        execution_dispatch_policy_id="execution_dispatch_policy.v1",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatched_candidates=candidates,
        dispatch_count=len(candidates),
    )


def _effect(signal="execution_pressure", direction="no_change", predicted=0.0, n=0):
    return ExpectedEffectV1(
        signal_id=signal,
        direction=direction,
        predicted_delta=predicted,
        predictor_variance=0.25,
        predictor_n=n,
        cold_start=(n == 0),
    )


class TestScoringHappensAtAll:
    def test_scores_a_declared_signal_against_the_real_field_delta(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect())]),
            feedback_frame_id="feedback.frame:test",
            field_before=_field("before", {"execution_pressure": 0.8}),
            field_after=_field("after", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert len(res.records) == 1
        r = res.records[0]
        assert r.baseline == pytest.approx(0.8)
        assert r.observed_after == pytest.approx(0.3)
        assert r.observed_delta == pytest.approx(-0.5)
        # predicted 0.0, observed -0.5 -> residual is the raw difference
        assert r.prediction_error == pytest.approx(-0.5)
        assert r.surprise_nats > 0.0
        assert r.posterior_n == 1
        assert not res.skipped

    def test_posterior_is_returned_keyed_by_kind_target_signal(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [_candidate("d1", kind="maintain", target_id="host:docker_images",
                            effect=_effect("resource_pressure", "decrease"))]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.9}),
            field_after=_field("a", {"resource_pressure": 0.4}),
            now=NOW,
        )
        assert set(res.posteriors) == {
            ("maintain", "host:docker_images", "resource_pressure")
        }

    def test_prior_is_carried_in_and_advances(self):
        prior = EffectPosterior(mean=-0.4, variance=0.01, n=25)
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect(predicted=-0.4, n=25))]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.4}),
            priors={("inspect", "capability:orchestration", "execution_pressure"): prior},
            now=NOW,
        )
        r = res.records[0]
        assert r.posterior_n == 26
        # observation (-0.4) exactly confirms the prior mean -> tiny surprise
        assert r.prediction_error == pytest.approx(0.0, abs=1e-12)
        assert r.surprise_nats < 0.05


class TestRefusesToInventData:
    def test_absent_signal_is_skipped_not_scored_as_zero(self):
        """THE guard. resource_pressure is not in either field snapshot.

        Scoring it would read baseline=0.0, after=0.0, delta=0.0 and confirm
        a `no_change` claim that was never measured.
        """
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [_candidate("d1", effect=_effect("resource_pressure", "decrease"))]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert res.records == []
        assert res.skipped == {"d1": "signal_absent_from_field:resource_pressure"}

    def test_signal_absent_only_from_the_after_tick_is_still_skipped(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect())]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"reasoning_pressure": 0.2}),
            now=NOW,
        )
        assert res.records == []
        assert "d1" in res.skipped

    def test_missing_field_window_is_skipped(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect())]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=None,
            now=NOW,
        )
        assert res.records == []
        assert res.skipped == {"d1": "missing_field_window"}

    def test_undeclared_action_is_recorded_as_an_absence(self):
        """5 of 16 live templates declare nothing, and they account for ~62%
        of real dispatches. That must show up as a counted skip reason, not
        vanish."""
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=None)]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert res.records == []
        assert res.skipped == {"d1": "no_declared_signal"}

    def test_empty_dispatch_frame_produces_nothing_without_error(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert res.records == [] and res.posteriors == {} and res.skipped == {}


class TestAttributionHonesty:
    def test_co_predictors_counts_the_other_claimants_on_the_same_signal(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [
                    _candidate("d1", target_id="capability:orchestration", effect=_effect()),
                    _candidate("d2", target_id="capability:transport", effect=_effect()),
                    _candidate("d3", target_id="node:atlas", effect=_effect()),
                ]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert len(res.records) == 3
        assert {r.co_predictors for r in res.records} == {2}

    def test_sole_claimant_has_zero_co_predictors(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect())]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert res.records[0].co_predictors == 0

    def test_undeclared_candidates_do_not_inflate_co_predictors(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [_candidate("d1", effect=_effect()), _candidate("d2", effect=None)]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        assert [r.co_predictors for r in res.records] == [0]


class TestScoresTheClaimThatWasActuallyMade:
    def test_predicted_delta_comes_from_the_candidate_not_the_running_prior(self):
        """Two candidates share one posterior key in one frame.

        The second must still be scored against the prediction stamped on it
        before it ran, not against the posterior the first one just moved --
        otherwise the ledger records a prediction that was never made.
        """
        effect = _effect(predicted=-0.10, n=5)
        prior = EffectPosterior(mean=-0.10, variance=0.02, n=5)
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [_candidate("d1", effect=effect), _candidate("d2", effect=effect)]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            priors={("inspect", "capability:orchestration", "execution_pressure"): prior},
            now=NOW,
        )
        assert [r.predicted_delta for r in res.records] == [-0.10, -0.10]
        # ...but the posterior genuinely advanced twice, not once.
        assert [r.posterior_n for r in res.records] == [6, 7]

    def test_direction_is_carried_onto_the_record(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [_candidate("d1", kind="maintain",
                            effect=_effect("resource_pressure", "decrease"))]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"resource_pressure": 0.8}),
            field_after=_field("a", {"resource_pressure": 0.3}),
            now=NOW,
        )
        assert res.records[0].direction == "decrease"

    def test_latency_is_none_when_unreported_never_zero(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect())]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            latency_by_dispatch_id={},
            now=NOW,
        )
        assert res.records[0].latency_ms is None
