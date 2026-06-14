"""Pure proposal lifecycle transition validation and state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from orion.schemas.proposal_ledger import (
    ProposalExecutionEligibilityV1,
    ProposalExecutionReceiptV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
    ProposalStatus,
    ProposalTriageDecisionV1,
)

ProposalActor = Literal["context-exec", "cortex_policy", "human", "executor", "system"]

# Allowed transitions: (current, next) -> set of actors permitted
_ALLOWED_TRANSITIONS: dict[tuple[ProposalStatus, ProposalStatus], frozenset[str]] = {
    ("draft", "stored"): frozenset({"context-exec", "system"}),
    ("stored", "pending_review"): frozenset({"cortex_policy", "system"}),
    ("stored", "blocked"): frozenset({"cortex_policy", "system"}),
    ("stored", "discarded"): frozenset({"cortex_policy", "system"}),
    ("stored", "expired"): frozenset({"cortex_policy", "system"}),
    ("stored", "superseded"): frozenset({"cortex_policy", "system"}),
    ("blocked", "pending_review"): frozenset({"cortex_policy", "system"}),
    ("blocked", "discarded"): frozenset({"cortex_policy", "system"}),
    ("blocked", "superseded"): frozenset({"cortex_policy", "system"}),
    ("pending_review", "approved"): frozenset({"human", "cortex_policy"}),
    ("pending_review", "rejected"): frozenset({"human", "cortex_policy"}),
    ("pending_review", "request_changes"): frozenset({"human", "cortex_policy"}),
    ("pending_review", "superseded"): frozenset({"cortex_policy", "system"}),
    ("request_changes", "pending_review"): frozenset({"human", "cortex_policy", "system"}),
    ("approved", "execution_requested"): frozenset({"executor", "system"}),
    ("execution_requested", "executed"): frozenset({"executor", "system"}),
    ("execution_requested", "failed"): frozenset({"executor", "system"}),
    ("approved", "superseded"): frozenset({"cortex_policy", "system"}),
}

# Explicitly forbidden actor+transition pairs (context-exec cannot approve/execute)
_CONTEXT_EXEC_FORBIDDEN_NEXT: frozenset[ProposalStatus] = frozenset(
    {"approved", "execution_requested", "executed"}
)


@dataclass(frozen=True)
class ProposalTransitionResult:
    valid: bool
    reason: str | None = None


def validate_proposal_transition(
    current_status: ProposalStatus,
    next_status: ProposalStatus,
    actor: ProposalActor,
) -> ProposalTransitionResult:
    """Validate a proposal status transition for the given actor."""
    if current_status == next_status:
        return ProposalTransitionResult(valid=True)

    if actor == "context-exec" and next_status in _CONTEXT_EXEC_FORBIDDEN_NEXT:
        return ProposalTransitionResult(
            valid=False,
            reason=f"context-exec cannot transition to {next_status!r}",
        )

    key = (current_status, next_status)
    allowed_actors = _ALLOWED_TRANSITIONS.get(key)
    if allowed_actors is None:
        return ProposalTransitionResult(
            valid=False,
            reason=f"transition {current_status!r} -> {next_status!r} is not allowed",
        )

    if actor not in allowed_actors:
        return ProposalTransitionResult(
            valid=False,
            reason=(
                f"actor {actor!r} may not perform transition "
                f"{current_status!r} -> {next_status!r}"
            ),
        )

    return ProposalTransitionResult(valid=True)


def _status_for_triage_action(action: str) -> ProposalStatus:
    mapping: dict[str, ProposalStatus] = {
        "store_only": "stored",
        "promote_to_review": "pending_review",
        "block_for_evidence": "blocked",
        "discard": "discarded",
        "supersede": "superseded",
        "expire": "expired",
    }
    return mapping[action]


def _triage_actor(decision: ProposalTriageDecisionV1) -> ProposalActor:
    """Map triage reviewer_type to lifecycle actor (human triage uses cortex_policy)."""
    if decision.reviewer_type == "system":
        return "system"
    return "cortex_policy"


def apply_triage_decision(
    record: ProposalLedgerRecordV1,
    decision: ProposalTriageDecisionV1,
) -> ProposalLedgerRecordV1:
    """Apply triage decision to a ledger record (pure, no persistence)."""
    next_status = _status_for_triage_action(decision.action)
    actor = _triage_actor(decision)

    result = validate_proposal_transition(record.status, next_status, actor)
    if not result.valid:
        raise ValueError(result.reason or "invalid triage transition")

    attention_required = decision.attention_required
    attention_reason = decision.attention_reason
    if decision.action == "promote_to_review":
        attention_required = True
        attention_reason = attention_reason or decision.rationale
    elif decision.action == "store_only":
        attention_required = False
        attention_reason = None
    elif decision.action == "block_for_evidence":
        attention_required = bool(decision.attention_required)
    else:
        attention_required = False
        attention_reason = None

    return ProposalLedgerRecordV1.model_validate(
        record.model_copy(
            update={
                "status": next_status,
                "triage_action": decision.action,
                "attention_required": attention_required,
                "attention_reason": attention_reason,
            }
        ).model_dump()
    )


def _review_actor(decision: ProposalReviewDecisionV1) -> ProposalActor:
    if decision.reviewer_type == "human":
        return "human"
    if decision.reviewer_type == "system":
        return "system"
    return "cortex_policy"


def apply_review_decision(
    record: ProposalLedgerRecordV1,
    decision: ProposalReviewDecisionV1,
) -> ProposalLedgerRecordV1:
    """Apply review decision to a ledger record (pure, no persistence)."""
    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "request_changes": "request_changes",
    }
    next_status: ProposalStatus = status_map[decision.decision]
    actor = _review_actor(decision)

    result = validate_proposal_transition(record.status, next_status, actor)
    if not result.valid:
        raise ValueError(result.reason or "invalid review transition")

    return ProposalLedgerRecordV1.model_validate(
        record.model_copy(
            update={
                "status": next_status,
                "attention_required": False,
                "attention_reason": None,
            }
        ).model_dump()
    )


def derive_execution_eligibility(
    record: ProposalLedgerRecordV1,
    review_decision: ProposalReviewDecisionV1 | None = None,
) -> ProposalExecutionEligibilityV1:
    """Derive execution eligibility from ledger state and optional review decision."""
    if record.status == "approved" and review_decision and review_decision.decision == "approve":
        return ProposalExecutionEligibilityV1(
            proposal_id=record.proposal_id,
            eligible=True,
            reason="approved by review decision",
            approved_decision_id=review_decision.decision_id,
            allowed_actions=list(review_decision.approved_actions),
            constraints=dict(review_decision.constraints),
            executor_required=True,
            execution_requested=False,
        )

    return ProposalExecutionEligibilityV1(
        proposal_id=record.proposal_id,
        eligible=False,
        reason=f"proposal status is {record.status!r}, not approved",
        executor_required=True,
        execution_requested=False,
    )


def validate_dry_run_execution_eligibility(
    record: ProposalLedgerRecordV1,
    eligibility: ProposalExecutionEligibilityV1,
) -> None:
    """Raise ValueError when a proposal is not eligible for dry-run execution."""
    if record.status != "approved":
        raise ValueError(
            f"dry-run execution requires status='approved', got {record.status!r}"
        )
    if not eligibility.eligible:
        raise ValueError(
            f"dry-run execution requires eligible=true: {eligibility.reason}"
        )
    if eligibility.execution_requested:
        raise ValueError(
            "dry-run execution requires execution_requested=false"
        )


def _planned_actions_for_record(record: ProposalLedgerRecordV1) -> list[str]:
    artifact_type = record.envelope.artifact_type
    if artifact_type == "PatchProposalV1":
        return ["simulate_patch_apply"]
    if artifact_type == "MemoryCorrectionProposalV1":
        return ["simulate_memory_correction"]
    return ["simulate_proposal_execution"]


def build_dry_run_execution_receipt(
    record: ProposalLedgerRecordV1,
    eligibility: ProposalExecutionEligibilityV1,
    *,
    executor_name: str,
    receipt_id: str,
    created_at: str,
    execution_request_id: str | None = None,
    trace_id: str | None = None,
) -> ProposalExecutionReceiptV1:
    """Build a simulated execution receipt without mutating ledger, memory, or repo."""
    validate_dry_run_execution_eligibility(record, eligibility)

    planned_actions = _planned_actions_for_record(record)
    if eligibility.allowed_actions:
        planned_actions = list(eligibility.allowed_actions)

    return ProposalExecutionReceiptV1(
        receipt_id=receipt_id,
        proposal_id=record.proposal_id,
        execution_request_id=execution_request_id,
        executor_name=executor_name,
        status="simulated",
        dry_run=True,
        mutation_performed=False,
        summary=(
            f"Dry-run simulated execution for {record.envelope.proposal_type} "
            f"via {executor_name}; no mutation performed"
        ),
        planned_actions=planned_actions,
        created_at=created_at,
        trace_id=trace_id,
    )
