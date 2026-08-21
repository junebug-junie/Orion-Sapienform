"""The declaration contract: what a template is allowed to claim, and what
the dispatch builder does with it.

A half-wired prediction is worse than no prediction -- it looks scored on
every surface while measuring nothing -- so these are load-time failures,
not build-time skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orion.execution_dispatch.builder import build_expected_effect
from orion.autonomy.prediction import EffectPosterior
from orion.proposals.policy import ProposalTemplateV1, load_proposal_policy
from orion.schemas.proposal_frame import ProposalCandidateV1

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "proposals" / "proposal_policy.v1.yaml"


def _template(**kwargs) -> dict:
    base = dict(
        kind="inspect",
        target_kind="capability",
        target_id="capability:orchestration",
        proposed_effect="increase_observability",
        required_policy_gate="none",
    )
    base.update(kwargs)
    return base


class TestTemplateValidation:
    def test_signal_without_direction_is_rejected(self):
        with pytest.raises(ValidationError, match="must be set together"):
            ProposalTemplateV1(**_template(expected_signal="execution_pressure"))

    def test_direction_without_signal_is_rejected(self):
        with pytest.raises(ValidationError, match="must be set together"):
            ProposalTemplateV1(**_template(expected_direction="decrease"))

    def test_unmeasured_signal_is_rejected(self):
        with pytest.raises(ValidationError, match="not a measured signal"):
            ProposalTemplateV1(
                **_template(expected_signal="vibes_pressure", expected_direction="decrease")
            )

    def test_unknown_direction_is_rejected(self):
        with pytest.raises(ValidationError, match="expected_direction"):
            ProposalTemplateV1(
                **_template(expected_signal="execution_pressure", expected_direction="sideways")
            )

    def test_declaring_nothing_is_allowed(self):
        t = ProposalTemplateV1(**_template())
        assert t.expected_signal is None and t.expected_direction is None

    def test_valid_declaration_round_trips(self):
        t = ProposalTemplateV1(
            **_template(expected_signal="resource_pressure", expected_direction="decrease")
        )
        assert (t.expected_signal, t.expected_direction) == ("resource_pressure", "decrease")


class TestLivePolicyFile:
    """Pinned against the real shipped config, not a synthetic fixture."""

    def test_every_mutating_template_declares_a_real_claim(self):
        """A `maintain` action changes the host. If it cannot say what it
        expects to change, nothing can ever tell whether it was worth its
        cost -- and mutating actions are the ones where that matters most."""
        policy = load_proposal_policy(POLICY_PATH)
        undeclared = [
            name
            for name, t in policy.proposal_templates.items()
            if t.kind == "maintain" and t.expected_signal is None
        ]
        assert undeclared == []

    def test_read_only_templates_never_claim_to_move_something(self):
        """Read-only kinds are constrained to no file writes, no external
        side effects. Any direction other than no_change would be a claim
        their own constraint set forbids."""
        policy = load_proposal_policy(POLICY_PATH)
        offenders = [
            (name, t.expected_direction)
            for name, t in policy.proposal_templates.items()
            if t.kind != "maintain"
            and t.expected_direction is not None
            and t.expected_direction != "no_change"
        ]
        assert offenders == []

    def test_declaration_coverage_is_recorded_not_assumed(self):
        """As of 2026-08-21: 11 of 16 templates declare a claim. The other 5
        declare nothing and account for ~62% of real dispatches.

        This asserts the direction of travel, not the exact number: coverage
        may go UP as templates are fixed or deleted, but a change that
        silently drops declarations fails here.
        """
        policy = load_proposal_policy(POLICY_PATH)
        declared = sum(1 for t in policy.proposal_templates.values() if t.expected_signal)
        assert declared >= 11, f"declaration coverage regressed to {declared}"


class TestBuildExpectedEffect:
    def _candidate(self, signal=None, direction=None, target="capability:orchestration"):
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

    def test_undeclared_candidate_yields_no_effect(self):
        assert build_expected_effect(self._candidate(), "inspect", {}) is None

    def test_cold_start_is_flagged_when_no_history_exists(self):
        effect = build_expected_effect(
            self._candidate("execution_pressure", "no_change"), "inspect", {}
        )
        assert effect is not None
        assert effect.cold_start is True
        assert effect.predicted_delta == 0.0
        assert effect.predictor_n == 0

    def test_prediction_uses_measured_history_when_it_exists(self):
        posteriors = {
            ("inspect", "capability:orchestration", "execution_pressure", 7): EffectPosterior(
                mean=-0.23, variance=0.004, n=140
            )
        }
        effect = build_expected_effect(
            self._candidate("execution_pressure", "no_change"), "inspect", posteriors
        )
        assert effect.predicted_delta == pytest.approx(-0.23)
        assert effect.predictor_n == 140
        assert effect.cold_start is False

    def test_posterior_is_keyed_by_resolved_target_not_template(self):
        """`inspect_attended_target` binds a different target per tick.
        Pooling those into one average would make every prediction wrong."""
        posteriors = {
            ("inspect", "node:atlas", "execution_pressure", 5): EffectPosterior(
                mean=-0.5, variance=0.004, n=90
            )
        }
        other = build_expected_effect(
            self._candidate("execution_pressure", "no_change", target="capability:transport"),
            "inspect",
            posteriors,
        )
        assert other.cold_start is True and other.predicted_delta == 0.0

    def test_n_zero_posterior_still_reads_as_cold(self):
        """A stored row with n=0 is not history; it must not claim to be."""
        posteriors = {
            ("inspect", "capability:orchestration", "execution_pressure", 4): EffectPosterior(
                mean=0.0, variance=0.25, n=0
            )
        }
        effect = build_expected_effect(
            self._candidate("execution_pressure", "no_change"), "inspect", posteriors
        )
        assert effect.cold_start is True
