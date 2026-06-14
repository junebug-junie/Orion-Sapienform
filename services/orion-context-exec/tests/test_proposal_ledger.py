"""Proposal ledger, triage, review-gate, and lifecycle contract tests."""

from __future__ import annotations

import pytest

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    build_patch_proposal_envelope,
)
from orion.schemas.proposal_ledger import (
    InMemoryProposalLedgerRepository,
    ProposalExecutionEligibilityV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)
from orion.schemas.proposal_lifecycle import (
    apply_review_decision,
    apply_triage_decision,
    derive_execution_eligibility,
    validate_proposal_transition,
)
from orion.schemas.registry import _REGISTRY


def _minimal_envelope(**overrides: object) -> ProposalEnvelopeV1:
    patch = PatchProposalV1(
        problem="test problem",
        proposed_change_summary="test summary",
        rollback_plan="revert",
    )
    envelope = build_patch_proposal_envelope(patch, source_mode="patch_proposal")
    if overrides:
        envelope = envelope.model_copy(update=overrides)
    return envelope


def _stored_record(**overrides: object) -> ProposalLedgerRecordV1:
    envelope = _minimal_envelope()
    record = ProposalLedgerRecordV1(
        proposal_id=envelope.proposal_id,
        envelope=envelope,
        status="stored",
    )
    if overrides:
        record = record.model_copy(update=overrides)
    return record


def test_proposal_ledger_review_schemas_registered() -> None:
    expected = {
        "ProposalLedgerRecordV1": ProposalLedgerRecordV1,
        "ProposalTriageDecisionV1": ProposalTriageDecisionV1,
        "ProposalReviewDecisionV1": ProposalReviewDecisionV1,
        "ProposalExecutionEligibilityV1": ProposalExecutionEligibilityV1,
        "ProposalEnvelopeV1": ProposalEnvelopeV1,
        "PatchProposalV1": PatchProposalV1,
        "MemoryCorrectionProposalV1": MemoryCorrectionProposalV1,
    }
    for name, cls in expected.items():
        assert _REGISTRY[name] is cls


def test_proposal_ledger_record_defaults_do_not_require_attention() -> None:
    record = _stored_record()
    assert record.status == "stored"
    assert record.attention_required is False
    assert record.triage_action == "store_only"


def test_triage_promote_to_review_sets_pending_review_attention() -> None:
    record = _stored_record()
    decision = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action="promote_to_review",
        rationale="worth human review",
        attention_required=True,
        attention_reason="high impact change",
    )
    updated = apply_triage_decision(record, decision)
    assert updated.status == "pending_review"
    assert updated.attention_required is True
    assert updated.attention_reason == "high impact change"


def test_triage_store_only_does_not_create_human_chore() -> None:
    record = _stored_record()
    decision = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action="store_only",
        rationale="low signal, keep on ledger only",
    )
    updated = apply_triage_decision(record, decision)
    assert updated.status == "stored"
    assert updated.attention_required is False


@pytest.mark.parametrize(
    "action",
    ["store_only", "discard", "expire", "supersede"],
)
def test_triage_non_promote_actions_do_not_require_attention(action: str) -> None:
    record = _stored_record()
    decision = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action=action,
        rationale="test",
    )
    updated = apply_triage_decision(record, decision)
    assert updated.attention_required is False


def test_triage_supersede_sets_superseded_status() -> None:
    record = _stored_record()
    decision = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action="supersede",
        rationale="replaced by newer proposal",
    )
    updated = apply_triage_decision(record, decision)
    assert updated.status == "superseded"
    assert updated.attention_required is False


def test_review_approval_does_not_execute() -> None:
    record = _stored_record()
    promote = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action="promote_to_review",
        rationale="review needed",
    )
    pending = apply_triage_decision(record, promote)

    review = ProposalReviewDecisionV1(
        decision_id="dec_001",
        proposal_id=pending.proposal_id,
        decision="approve",
        reviewer_type="human",
        reviewer_id="operator",
        rationale="looks good",
        approved_actions=["apply_patch"],
    )
    approved = apply_review_decision(pending, review)
    eligibility = derive_execution_eligibility(approved, review)

    assert approved.status == "approved"
    assert eligibility.eligible is True
    assert eligibility.execution_requested is False
    assert approved.status != "executed"


def test_context_exec_actor_cannot_approve_or_execute() -> None:
    for next_status in ("approved", "execution_requested", "executed"):
        result = validate_proposal_transition("stored", next_status, "context-exec")
        assert not result.valid, f"context-exec should not reach {next_status}"

    result_pending = validate_proposal_transition("pending_review", "approved", "context-exec")
    assert not result_pending.valid

    result_exec = validate_proposal_transition("approved", "execution_requested", "context-exec")
    assert not result_exec.valid


def test_invalid_proposal_transitions_rejected() -> None:
    forbidden = [
        ("stored", "executed", "executor"),
        ("rejected", "executed", "executor"),
        ("discarded", "approved", "human"),
        ("expired", "approved", "human"),
    ]
    for current, nxt, actor in forbidden:
        result = validate_proposal_transition(current, nxt, actor)
        assert not result.valid, f"expected forbidden: {current} -> {nxt} by {actor}"


def test_attention_required_invalid_outside_review_states() -> None:
    envelope = _minimal_envelope()
    with pytest.raises(ValueError, match="attention_required"):
        ProposalLedgerRecordV1(
            proposal_id=envelope.proposal_id,
            envelope=envelope,
            status="stored",
            attention_required=True,
        )


def test_eligible_requires_approved_decision_id() -> None:
    with pytest.raises(ValueError, match="approved_decision_id"):
        ProposalExecutionEligibilityV1(
            proposal_id="prop_x",
            eligible=True,
            reason="bad",
        )


def test_in_memory_repository_triage_and_review() -> None:
    repo = InMemoryProposalLedgerRepository()
    record = _stored_record()
    repo.store(record)

    triage = ProposalTriageDecisionV1(
        proposal_id=record.proposal_id,
        action="promote_to_review",
        rationale="needs eyes",
    )
    pending = repo.apply_triage(triage)
    assert pending.status == "pending_review"

    review = ProposalReviewDecisionV1(
        decision_id="dec_repo",
        proposal_id=record.proposal_id,
        decision="approve",
        reviewer_type="human",
        reviewer_id="op",
        rationale="ok",
    )
    approved = repo.apply_review(review)
    assert approved.status == "approved"

    listed = repo.list_by_status("approved")
    assert len(listed) == 1
