"""ProposalEnvelopeV1 and proposal review contract tests."""

from __future__ import annotations

import pytest

from orion.schemas.context_exec import (
    CONTEXT_EXEC_CREATABLE_REVIEW_STATES,
    PatchProposalV1,
    ProposalEnvelopeV1,
    assert_context_exec_proposal_safe,
    build_patch_proposal_envelope,
)
from orion.schemas.registry import _REGISTRY


def test_proposal_envelope_defaults_require_review_and_disallow_mutation() -> None:
    envelope = ProposalEnvelopeV1(
        proposal_id="prop_test",
        proposal_type="patch_proposal",
        source_mode="patch_proposal",
        title="Test proposal",
        summary="Summary",
        artifact_type="PatchProposalV1",
        artifact={
            "problem": "p",
            "proposed_change_summary": "s",
            "rollback_plan": "r",
            "mutation_allowed": False,
        },
    )
    assert envelope.mutation_allowed is False
    assert envelope.requires_human_approval is True
    assert envelope.review_status in CONTEXT_EXEC_CREATABLE_REVIEW_STATES


def test_proposal_envelope_registered_with_context_exec_schemas() -> None:
    assert _REGISTRY["ProposalEnvelopeV1"] is ProposalEnvelopeV1
    assert _REGISTRY["PatchProposalV1"] is PatchProposalV1
    assert _REGISTRY["BeliefProvenanceReportV1"] is not None
    assert _REGISTRY["TraceAutopsyReportV1"] is not None
    assert _REGISTRY["RepoImpactAnalysisReportV1"] is not None


def test_build_patch_proposal_envelope_wraps_inner_payload() -> None:
    patch = PatchProposalV1(
        problem="weak synthesis",
        proposed_change_summary="Improve heuristics",
        rollback_plan="Revert edits",
    )
    envelope = build_patch_proposal_envelope(patch, source_mode="patch_proposal")
    assert envelope.proposal_type == "patch_proposal"
    assert envelope.artifact_type == "PatchProposalV1"
    assert envelope.artifact["mutation_allowed"] is False
    assert envelope.mutation_allowed is False
    assert envelope.requires_human_approval is True
    assert envelope.review_status in {"draft", "pending_review"}


def test_context_exec_proposal_cannot_self_approve_or_execute() -> None:
    patch = PatchProposalV1(
        problem="p",
        proposed_change_summary="s",
        rollback_plan="r",
    )
    envelope = build_patch_proposal_envelope(patch, source_mode="patch_proposal")
    assert envelope.review_status not in {"approved", "executed"}
    assert envelope.mutation_allowed is False

    bad_envelope = envelope.model_copy(update={"review_status": "approved"})
    with pytest.raises(ValueError, match="review_status"):
        assert_context_exec_proposal_safe(bad_envelope)

    bad_mutation = envelope.model_copy(update={"mutation_allowed": True})
    with pytest.raises(ValueError, match="mutation_allowed"):
        assert_context_exec_proposal_safe(bad_mutation)
