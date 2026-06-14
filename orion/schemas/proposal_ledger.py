"""Generic proposal ledger and review-gate state contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.schemas.context_exec import ProposalEnvelopeV1, ProposalRiskLevel

ProposalStatus = Literal[
    "draft",
    "stored",
    "blocked",
    "discarded",
    "pending_review",
    "request_changes",
    "approved",
    "rejected",
    "superseded",
    "execution_requested",
    "executed",
    "failed",
    "expired",
]

ProposalTriageAction = Literal[
    "store_only",
    "promote_to_review",
    "block_for_evidence",
    "discard",
    "supersede",
    "expire",
]

ProposalReviewDecisionKind = Literal[
    "approve",
    "reject",
    "request_changes",
]

ProposalReviewerType = Literal["human", "cortex_policy", "system"]


class ProposalLedgerRecordV1(BaseModel):
    """Durable ledger row wrapping a proposal envelope and lifecycle state."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    envelope: ProposalEnvelopeV1

    status: ProposalStatus = "stored"

    triage_action: ProposalTriageAction = "store_only"

    attention_required: bool = False
    attention_reason: str | None = None

    source_run_id: str | None = None
    source_trace_id: str | None = None
    source_mode: str | None = None

    created_at: str | None = None
    updated_at: str | None = None
    expires_at: str | None = None

    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None

    ledger_version: str = "v1"

    @model_validator(mode="after")
    def _attention_only_for_reviewable_states(self) -> ProposalLedgerRecordV1:
        if self.attention_required and self.status not in {"pending_review", "blocked"}:
            raise ValueError(
                "attention_required=true is only valid for pending_review or blocked states"
            )
        return self


class ProposalTriageDecisionV1(BaseModel):
    """Policy triage outcome — prevents Hub from becoming an approval sweatshop."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    action: ProposalTriageAction

    rationale: str
    risk: ProposalRiskLevel = "unknown"
    attention_required: bool = False
    attention_reason: str | None = None

    reviewer_type: ProposalReviewerType = "cortex_policy"
    created_at: str | None = None


class ProposalReviewDecisionV1(BaseModel):
    """Human or policy review outcome — does not execute anything."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    proposal_id: str

    decision: ProposalReviewDecisionKind

    reviewer_type: ProposalReviewerType
    reviewer_id: str
    rationale: str

    constraints: dict[str, Any] = Field(default_factory=dict)
    approved_actions: list[str] = Field(default_factory=list)

    created_at: str | None = None


class ProposalExecutionEligibilityV1(BaseModel):
    """Describes whether a future executor may consume an approved proposal."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    eligible: bool = False
    reason: str

    approved_decision_id: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    executor_required: bool = True
    execution_requested: bool = False

    @model_validator(mode="after")
    def _eligible_requires_approval(self) -> ProposalExecutionEligibilityV1:
        if self.eligible and not self.approved_decision_id:
            raise ValueError("eligible=true requires an approve decision (approved_decision_id)")
        return self


class ProposalLedgerRepository(Protocol):
    """Contract for durable proposal ledger persistence (implementation deferred)."""

    def store(self, record: ProposalLedgerRecordV1) -> ProposalLedgerRecordV1: ...

    def get(self, proposal_id: str) -> ProposalLedgerRecordV1 | None: ...

    def list_by_status(self, status: ProposalStatus) -> list[ProposalLedgerRecordV1]: ...

    def apply_triage(self, decision: ProposalTriageDecisionV1) -> ProposalLedgerRecordV1: ...

    def apply_review(self, decision: ProposalReviewDecisionV1) -> ProposalLedgerRecordV1: ...


class InMemoryProposalLedgerRepository:
    """In-memory ledger for tests and local harnesses."""

    def __init__(self) -> None:
        self._records: dict[str, ProposalLedgerRecordV1] = {}

    def store(self, record: ProposalLedgerRecordV1) -> ProposalLedgerRecordV1:
        self._records[record.proposal_id] = record
        return record

    def get(self, proposal_id: str) -> ProposalLedgerRecordV1 | None:
        return self._records.get(proposal_id)

    def list_by_status(self, status: ProposalStatus) -> list[ProposalLedgerRecordV1]:
        return [r for r in self._records.values() if r.status == status]

    def apply_triage(self, decision: ProposalTriageDecisionV1) -> ProposalLedgerRecordV1:
        from orion.schemas.proposal_lifecycle import apply_triage_decision

        record = self._records.get(decision.proposal_id)
        if record is None:
            raise KeyError(f"proposal not found: {decision.proposal_id}")
        updated = apply_triage_decision(record, decision)
        self._records[updated.proposal_id] = updated
        return updated

    def apply_review(self, decision: ProposalReviewDecisionV1) -> ProposalLedgerRecordV1:
        from orion.schemas.proposal_lifecycle import apply_review_decision

        record = self._records.get(decision.proposal_id)
        if record is None:
            raise KeyError(f"proposal not found: {decision.proposal_id}")
        updated = apply_review_decision(record, decision)
        self._records[updated.proposal_id] = updated
        return updated

    def list_all(self) -> list[ProposalLedgerRecordV1]:
        return list(self._records.values())


class JsonFileProposalLedgerRepository:
    """JSON file-backed ledger for operator CLI and local testing."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._inner = InMemoryProposalLedgerRepository()
        self._review_decisions: dict[str, list[ProposalReviewDecisionV1]] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for record_data in raw.get("records", []):
            record = ProposalLedgerRecordV1.model_validate(record_data)
            self._inner.store(record)
        for proposal_id, decisions in raw.get("review_decisions", {}).items():
            self._review_decisions[proposal_id] = [
                ProposalReviewDecisionV1.model_validate(item) for item in decisions
            ]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [
                record.model_dump(mode="json") for record in self._inner.list_all()
            ],
            "review_decisions": {
                proposal_id: [decision.model_dump(mode="json") for decision in decisions]
                for proposal_id, decisions in self._review_decisions.items()
            },
        }
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def store(self, record: ProposalLedgerRecordV1) -> ProposalLedgerRecordV1:
        stored = self._inner.store(record)
        self.save()
        return stored

    def get(self, proposal_id: str) -> ProposalLedgerRecordV1 | None:
        return self._inner.get(proposal_id)

    def list_by_status(self, status: ProposalStatus) -> list[ProposalLedgerRecordV1]:
        return self._inner.list_by_status(status)

    def list_all(self) -> list[ProposalLedgerRecordV1]:
        return self._inner.list_all()

    def apply_triage(self, decision: ProposalTriageDecisionV1) -> ProposalLedgerRecordV1:
        updated = self._inner.apply_triage(decision)
        self.save()
        return updated

    def apply_review(self, decision: ProposalReviewDecisionV1) -> ProposalLedgerRecordV1:
        updated = self._inner.apply_review(decision)
        history = self._review_decisions.setdefault(decision.proposal_id, [])
        history.append(decision)
        self.save()
        return updated

    def review_history(self, proposal_id: str) -> list[ProposalReviewDecisionV1]:
        return list(self._review_decisions.get(proposal_id, []))

    def latest_review_decision(self, proposal_id: str) -> ProposalReviewDecisionV1 | None:
        history = self._review_decisions.get(proposal_id, [])
        return history[-1] if history else None
