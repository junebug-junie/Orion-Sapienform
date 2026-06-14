"""Unit tests for dry-run execution receipt lifecycle helpers."""

from __future__ import annotations

import pytest

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    build_memory_correction_proposal_envelope,
)
from orion.schemas.proposal_ledger import (
    ProposalExecutionEligibilityV1,
    ProposalExecutionReceiptV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
)
from orion.schemas.proposal_lifecycle import (
    apply_review_decision,
    apply_triage_decision,
    build_dry_run_execution_receipt,
    validate_dry_run_execution_eligibility,
)
from orion.schemas.proposal_ledger import ProposalTriageDecisionV1


def _approved_record() -> tuple[ProposalLedgerRecordV1, ProposalReviewDecisionV1]:
    correction = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        proposed_belief="User location is unknown",
        correction_type="mark_uncertain",
        rationale="Insufficient evidence",
        rollback_plan="No mutation proposed; no rollback required.",
        risk="medium",
    )
    envelope = build_memory_correction_proposal_envelope(
        correction,
        source_mode="memory_correction_proposal",
    )
    record = ProposalLedgerRecordV1(
        proposal_id=envelope.proposal_id,
        envelope=envelope,
        status="stored",
        source_mode="memory_correction_proposal",
    )
    pending = apply_triage_decision(
        record,
        ProposalTriageDecisionV1(
            proposal_id=record.proposal_id,
            action="promote_to_review",
            rationale="review needed",
        ),
    )
    review = ProposalReviewDecisionV1(
        decision_id="dec_dry",
        proposal_id=pending.proposal_id,
        decision="approve",
        reviewer_type="human",
        reviewer_id="operator",
        rationale="bounded and reversible",
    )
    approved = apply_review_decision(pending, review)
    return approved, review


def test_validate_dry_run_rejects_execution_requested() -> None:
    record, review = _approved_record()
    eligibility = ProposalExecutionEligibilityV1(
        proposal_id=record.proposal_id,
        eligible=True,
        reason="approved",
        approved_decision_id=review.decision_id,
        execution_requested=True,
    )

    with pytest.raises(ValueError, match="execution_requested=false"):
        validate_dry_run_execution_eligibility(record, eligibility)


def test_build_dry_run_execution_receipt_enforces_invariants() -> None:
    record, review = _approved_record()
    eligibility = ProposalExecutionEligibilityV1(
        proposal_id=record.proposal_id,
        eligible=True,
        reason="approved",
        approved_decision_id=review.decision_id,
    )
    receipt = build_dry_run_execution_receipt(
        record,
        eligibility,
        executor_name="dry-run",
        receipt_id="rec_test",
        created_at="2026-06-14T00:00:00+00:00",
    )

    assert receipt.status == "simulated"
    assert receipt.dry_run is True
    assert receipt.mutation_performed is False
    assert receipt.planned_actions == ["simulate_memory_correction"]


def test_proposal_execution_receipt_schema_rejects_invalid_dry_run() -> None:
    with pytest.raises(ValueError, match="status='simulated'"):
        ProposalExecutionReceiptV1(
            receipt_id="rec_bad",
            proposal_id="prop_bad",
            executor_name="dry-run",
            status="succeeded",
            dry_run=True,
            mutation_performed=False,
            summary="invalid",
        )
