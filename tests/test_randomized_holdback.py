"""The only arm that licenses the word "causal".

The `no_action` control arm is quasi-experimental: ticks where nothing was
proposed are systematically calmer ticks, and baseline binning absorbs most of
that selection and provably not all of it. A randomized holdback withholds
ticks that WOULD have acted, so assignment is independent of the signal.

The design decision these tests exist to protect is PER-TICK vs PER-CANDIDATE.
The field delta is frame-wide. Withholding one candidate while its siblings run
gives a control observation contaminated by those siblings -- which is exactly
what made the capacity-blocked arm unusable. Repeating that mistake here would
produce a worse-than-useless arm wearing the word "randomized".
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.autonomy.contrast import (
    CONTROL_ARM_PRECEDENCE,
    HOLDBACK_BLOCK_REASON,
    ControlCell,
    contrast,
)
from orion.autonomy.prediction import EffectPosterior
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
_DIM_CHANNEL = {"resource_pressure": "pressure", "execution_pressure": "execution_pressure"}


def _field(tick, dims):
    channels = {_DIM_CHANNEL[d]: v for d, v in dims.items()}
    return FieldStateV1(
        generated_at=NOW, tick_id=tick, node_vectors={"node:test": channels},
        node_vector_updated_at={"node:test": {c: NOW for c in channels}},
    )


def _withheld(dispatch_id):
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id,
        source_decision_id=f"d:{dispatch_id}",
        source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="blocked",
        dispatch_mode="dispatch_read_only",
        dispatch_kind="maintain",
        target_id="host:docker_images",
        target_kind="host",
        risk_score=0.05,
        confidence_score=0.9,
        blocked_by=[HOLDBACK_BLOCK_REASON],
        reasons=["approved_maintenance_dispatch_v1", HOLDBACK_BLOCK_REASON],
    )


def _frame(blocked=()):
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:holdback",
        source_policy_frame_id="policy.frame:t",
        source_proposal_frame_id="proposal.frame:t",
        source_field_tick_id="tick:t",
        generated_at=NOW,
        execution_dispatch_policy_id="execution_dispatch_policy.v1",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=False,
        blocked_candidates=list(blocked),
        blocked_count=len(blocked),
    )


def _resolve(frame):
    return resolve_action_outcomes(
        dispatch_frame=frame,
        feedback_frame_id="f",
        field_before=_field("b", {"resource_pressure": 0.85}),
        field_after=_field("a", {"resource_pressure": 0.40}),
        now=NOW,
    )


class TestAWithheldTickIsAnUntreatedObservation:
    def test_it_scores_as_the_holdback_arm_not_no_action(self):
        res = _resolve(_frame([_withheld("w1")]))
        arms = {o.arm for o in res.control_observations}
        assert arms == {"randomized_holdback"}
        assert ("resource_pressure", "randomized_holdback", 8) in res.control_posteriors

    def test_an_ordinary_idle_tick_is_still_no_action(self):
        res = _resolve(_frame())
        assert {o.arm for o in res.control_observations} == {"no_action"}

    def test_the_two_arms_do_not_share_a_cell(self):
        """They must never be pooled -- one is causal and one is not, and a
        merged number would be neither while being described as the better."""
        held = _resolve(_frame([_withheld("w1")]))
        idle = _resolve(_frame())
        assert set(held.control_posteriors) & set(idle.control_posteriors) == set()

    def test_a_withheld_tick_dispatched_nothing(self):
        """The whole point. If anything ran, the observation is contaminated
        by it and is not a control at all."""
        frame = _frame([_withheld("w1"), _withheld("w2")])
        assert frame.dispatched_candidates == []
        res = _resolve(frame)
        assert all(r.arm != "dispatched" for r in res.records)

    def test_the_arm_is_named_in_the_log_summary(self):
        """A withheld tick and an idle tick produce identical-looking counts
        and mean completely different things."""
        from orion.feedback.outcome_resolution import summarize_control_observations

        held = summarize_control_observations(_resolve(_frame([_withheld("w")])).control_observations)
        idle = summarize_control_observations(_resolve(_frame()).control_observations)
        assert held.startswith("[randomized_holdback]")
        assert idle.startswith("[no_action]")


class TestTheHoldbackArmWinsThePrecedence:
    def test_it_outranks_no_action(self):
        assert CONTROL_ARM_PRECEDENCE[0] == "randomized_holdback"

    def test_contrast_prefers_it_even_with_far_less_data(self):
        """A small causal arm beats a large confounded one. That is the whole
        reason to spend capability on it."""
        treated = {
            ("maintain", "host:docker_images", "resource_pressure", 8):
                EffectPosterior(mean=-0.15, variance=0.001, n=3000)
        }
        control = {
            ("resource_pressure", "no_action", 8):
                ControlCell(EffectPosterior(-0.14, 0.0005, 20000), moved_n=18000, move_rate=0.9),
            ("resource_pressure", "randomized_holdback", 8):
                ControlCell(EffectPosterior(-0.02, 0.01, 40), moved_n=35, move_rate=0.9),
        }
        est = contrast(treated, control, "maintain", "host:docker_images", "resource_pressure")
        assert est.control_arm == "randomized_holdback"
        assert est.evidence_class == "experimental"
        # -0.15 - (-0.02) = -0.13, not the -0.01 the confounded arm would give.
        assert est.value == pytest.approx(-0.13, abs=1e-9)
        assert est.control_n == 40

    def test_without_a_holdback_arm_it_falls_back_and_says_so(self):
        treated = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(-0.15, 0.001, 3000)
        }
        control = {
            ("resource_pressure", "no_action", 8):
                ControlCell(EffectPosterior(-0.14, 0.0005, 20000), moved_n=18000, move_rate=0.9)
        }
        est = contrast(treated, control, "maintain", "t", "resource_pressure")
        assert est.control_arm == "no_action"
        assert est.evidence_class == "quasi_experimental"
