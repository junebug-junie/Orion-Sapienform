"""Schema-level proposal ledger registry tests."""

from __future__ import annotations

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
)
from orion.schemas.proposal_ledger import (
    ProposalExecutionEligibilityV1,
    ProposalExecutionReceiptV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)
from orion.schemas.registry import _REGISTRY, resolve

PROPOSAL_CONTROL_PLANE_SCHEMAS = (
    "PatchProposalV1",
    "ProposalEnvelopeV1",
    "MemoryCorrectionProposalV1",
    "ProposalLedgerRecordV1",
    "ProposalTriageDecisionV1",
    "ProposalReviewDecisionV1",
    "ProposalExecutionEligibilityV1",
    "ProposalExecutionReceiptV1",
)


def test_all_proposal_control_plane_schemas_registered() -> None:
    for schema_id in PROPOSAL_CONTROL_PLANE_SCHEMAS:
        assert schema_id in _REGISTRY
        assert resolve(schema_id) is _REGISTRY[schema_id]


def test_proposal_ledger_schemas_resolve() -> None:
    for schema_id in PROPOSAL_CONTROL_PLANE_SCHEMAS:
        assert resolve(schema_id) is _REGISTRY[schema_id]


def test_memory_correction_proposal_registered_schema_is_full_version() -> None:
    registered = resolve("MemoryCorrectionProposalV1")
    assert registered is MemoryCorrectionProposalV1

    sample = MemoryCorrectionProposalV1(
        current_belief="User is from Denver",
        proposed_belief="User location is unknown",
        correction_type="mark_uncertain",
        rationale="Insufficient evidence for Denver claim",
        supporting_evidence=["user said maybe Colorado once"],
        contradicting_evidence=["no verified Denver source"],
        missing_evidence=["passport or address record"],
        target_memory_domains=["cards", "rdf"],
        affected_ids=["card_123"],
        confidence=0.2,
        risk="medium",
        tests_to_run=["recall regression"],
        rollback_plan="No mutation proposed; no rollback required.",
        open_questions=["Which memory card holds the Denver claim?"],
        mutation_allowed=False,
    )
    validated = registered.model_validate(sample.model_dump(mode="json"))
    assert validated.current_belief == sample.current_belief
    assert validated.correction_type == "mark_uncertain"
    assert validated.target_memory_domains == ["cards", "rdf"]
    assert validated.mutation_allowed is False

    assert _REGISTRY["MemoryCorrectionProposalV1"] is MemoryCorrectionProposalV1
    assert _REGISTRY["MemoryCorrectionProposalV1"] is not PatchProposalV1


def test_proposal_execution_receipt_registered_schema_round_trip() -> None:
    registered = resolve("ProposalExecutionReceiptV1")
    assert registered is ProposalExecutionReceiptV1

    sample = ProposalExecutionReceiptV1(
        receipt_id="rec_sample",
        proposal_id="prop_sample",
        executor_name="dry-run",
        status="simulated",
        dry_run=True,
        mutation_performed=False,
        summary="Dry-run scaffold receipt",
        planned_actions=["simulate_memory_correction"],
        created_at="2026-06-14T00:00:00+00:00",
    )
    validated = registered.model_validate(sample.model_dump(mode="json"))
    assert validated.receipt_id == sample.receipt_id
    assert validated.dry_run is True
    assert validated.mutation_performed is False
