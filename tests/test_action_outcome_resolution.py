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
        # Cell key now carries the baseline bin: before=0.9 -> bin 9.
        assert set(res.posteriors) == {
            ("maintain", "host:docker_images", "resource_pressure", 9)
        }

    def test_prior_is_carried_in_and_advances(self):
        prior = EffectPosterior(mean=-0.4, variance=0.01, n=25)
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect(predicted=-0.4, n=25))]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.4}),
            priors={("inspect", "capability:orchestration", "execution_pressure", 8): prior},
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
            priors={("inspect", "capability:orchestration", "execution_pressure", 8): prior},
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


class TestDirectionIsActuallyScored:
    """Review finding 3: `direction` had a schema, a producer and a
    persister, and no consumer -- so an action that declared `decrease` and
    delivered `+0.4` scored identically to one that declared `increase`.
    These pin the consumer down."""

    def _resolve(self, direction, before, after):
        return resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect(direction=direction))]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": before}),
            field_after=_field("a", {"execution_pressure": after}),
            now=NOW,
        )

    def test_decrease_claim_upheld_when_it_decreases(self):
        assert self._resolve("decrease", 0.8, 0.3).records[0].claim_upheld is True

    def test_decrease_claim_broken_when_it_increases(self):
        assert self._resolve("decrease", 0.3, 0.8).records[0].claim_upheld is False

    def test_increase_claim_upheld_when_it_increases(self):
        assert self._resolve("increase", 0.3, 0.8).records[0].claim_upheld is True

    def test_increase_claim_broken_when_it_decreases(self):
        assert self._resolve("increase", 0.8, 0.3).records[0].claim_upheld is False

    def test_no_change_claim_upheld_when_nothing_moves(self):
        assert self._resolve("no_change", 0.5, 0.5).records[0].claim_upheld is True

    def test_no_change_claim_broken_when_something_moves(self):
        assert self._resolve("no_change", 0.5, 0.9).records[0].claim_upheld is False

    def test_directional_claim_inside_the_dead_band_is_undecidable_not_a_pass(self):
        """A `decrease` claim with a 1e-9 wobble must NOT read as upheld.
        Returning True there is how a dead channel confirms every claim."""
        r = self._resolve("decrease", 0.5, 0.5 - 1e-9).records[0]
        assert r.claim_upheld is None

    def test_no_change_claim_is_decidable_in_both_directions(self):
        """no_change never returns None -- the dead band IS its claim."""
        assert self._resolve("no_change", 0.5, 0.5 - 1e-9).records[0].claim_upheld is True

    def test_opposite_directions_on_the_same_delta_do_not_score_alike(self):
        up = self._resolve("increase", 0.3, 0.8).records[0]
        down = self._resolve("decrease", 0.3, 0.8).records[0]
        # Same observation, same nats -- surprise is direction-agnostic by
        # construction -- but the CLAIM verdicts must differ, which is the
        # whole point of the field.
        assert up.surprise_nats == pytest.approx(down.surprise_nats)
        assert up.claim_upheld is True and down.claim_upheld is False


class TestPredictionErrorMatchesTheStoredClaim:
    """Review finding 6: prediction_error was measured against the belief at
    SCORING time, while predicted_delta on the same row is the claim made at
    DISPATCH time. A reader recomputing the error from the row got a
    different number, sometimes with the opposite sign."""

    def test_error_is_recomputable_from_the_row(self):
        prior = EffectPosterior(mean=-0.05, variance=0.02, n=30)
        res = resolve_action_outcomes(
            dispatch_frame=_frame([_candidate("d1", effect=_effect(predicted=-0.30, n=30))]),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.6}),
            priors={("inspect", "capability:orchestration", "execution_pressure", 8): prior},
            now=NOW,
        )
        r = res.records[0]
        assert r.observed_delta == pytest.approx(-0.2)
        assert r.predicted_delta == pytest.approx(-0.3)
        # The exact case that used to disagree: residual-vs-prior would be
        # -0.15 here, opposite in sign to the recomputable +0.10.
        assert r.prediction_error == pytest.approx(0.1)
        assert r.prediction_error == pytest.approx(r.observed_delta - r.predicted_delta)

    def test_holds_for_every_record_in_a_multi_candidate_frame(self):
        res = resolve_action_outcomes(
            dispatch_frame=_frame(
                [
                    _candidate("d1", effect=_effect(predicted=-0.1, n=5)),
                    _candidate("d2", effect=_effect(predicted=-0.1, n=5)),
                ]
            ),
            feedback_frame_id="f",
            field_before=_field("b", {"execution_pressure": 0.8}),
            field_after=_field("a", {"execution_pressure": 0.3}),
            now=NOW,
        )
        for r in res.records:
            assert r.prediction_error == pytest.approx(r.observed_delta - r.predicted_delta)


def test_control_cells_only_written_for_predictable_signals() -> None:
    """LOW: the control arm wrote cells for signals no action can ever claim.

    The loop iterated every key in `_present_pressures(...)` -- all of
    `field_pressures()` -- rather than the closed `PredictableSignal`
    vocabulary. Live consequence: `substrate_signal_control_cells` accumulated
    continuity_pressure / introspection_pressure / social_pressure rows that
    `contrast()` can never look up, because no `ExpectedEffectV1` can name
    them. It also made `load_control_posteriors`'s documented bound of
    `len(PredictableSignal) * arms * 10` false.
    """
    from orion.feedback.outcome_resolution import _PREDICTABLE_SIGNALS
    from orion.schemas.action_prediction import PredictableSignal

    assert _PREDICTABLE_SIGNALS == frozenset(PredictableSignal.__args__)
    for leaked in ("continuity_pressure", "introspection_pressure", "social_pressure"):
        assert leaked not in _PREDICTABLE_SIGNALS, (
            f"{leaked} is not claimable by any action and must not get a control cell"
        )
    assert "reliability_pressure" in _PREDICTABLE_SIGNALS


def test_control_cell_upsert_lets_a_null_frame_id_through() -> None:
    """MEDIUM: `IS DISTINCT FROM` is FALSE when both sides are NULL.

    The docstring claimed "a NULL stored token is DISTINCT FROM anything, so
    the write proceeds". Postgres disagrees: `SELECT NULL::text IS DISTINCT
    FROM NULL::text` -> `f`. So a caller using the `control_frame_id=None`
    default got its first INSERT and then had every later update refused
    forever -- the control arm silently frozen at one observation, no error.

    Asserts on the real SQL text: the guard must have an explicit NULL branch.
    """
    import pathlib
    import re

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "services" / "orion-feedback-runtime" / "app" / "store.py"
    ).read_text()

    i = src.index("INSERT INTO substrate_signal_control_cells")
    guard = src[i:i + 2000]
    assert "EXCLUDED.last_dispatch_frame_id IS NULL" in guard, (
        "the dedup guard needs an explicit `EXCLUDED.last_dispatch_frame_id IS "
        "NULL OR ...` branch; a bare IS DISTINCT FROM wedges NULL-frame callers "
        "after their first write"
    )
    assert not re.search(
        r"AND\s+substrate_signal_control_cells\.last_dispatch_frame_id\s*\n\s*"
        r"IS DISTINCT FROM EXCLUDED\.last_dispatch_frame_id",
        guard,
    ), "the unguarded IS DISTINCT FROM form is back"
