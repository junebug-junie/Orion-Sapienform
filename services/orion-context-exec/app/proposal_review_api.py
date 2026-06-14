"""Proposal ledger review API — Hub-facing control-plane seam (no execution)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.context_exec import (
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
)
from orion.schemas.proposal_ledger import (
    JsonFileProposalLedgerRepository,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionKind,
    ProposalReviewDecisionV1,
    ProposalReviewerType,
    ProposalStatus,
    ProposalTriageAction,
    ProposalTriageDecisionV1,
)
from orion.schemas.proposal_lifecycle import derive_execution_eligibility

from .settings import settings

logger = logging.getLogger("orion-context-exec.proposal_review_api")
router = APIRouter(tags=["proposal-review"])


class ProposalReviewApiError(Exception):
    """Controlled proposal review API failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_store_path() -> Path:
    raw = (settings.proposal_ledger_store_path or "").strip()
    if not raw:
        raise ProposalReviewApiError(
            "PROPOSAL_LEDGER_STORE_PATH is required (explicit JSON ledger path; no repo default)"
        )
    return Path(raw)


def _parse_reviewer(reviewer_type: ProposalReviewerType, reviewer_id: str) -> tuple[str, str]:
    if reviewer_type == "human" and reviewer_id == "context-exec":
        raise ProposalReviewApiError("context-exec cannot act as proposal reviewer")
    if reviewer_id == "context-exec":
        raise ProposalReviewApiError("context-exec cannot act as proposal reviewer")
    return reviewer_type, reviewer_id


def _open_repo(store_path: Path) -> JsonFileProposalLedgerRepository:
    try:
        return JsonFileProposalLedgerRepository(store_path)
    except json.JSONDecodeError as exc:
        raise ProposalReviewApiError(f"malformed proposal ledger store: {exc}") from exc
    except ValueError as exc:
        raise ProposalReviewApiError(f"invalid proposal ledger store: {exc}") from exc


def _record_row(record: ProposalLedgerRecordV1) -> dict[str, Any]:
    return {
        "proposal_id": record.proposal_id,
        "proposal_type": record.envelope.proposal_type,
        "status": record.status,
        "risk": record.envelope.risk,
        "attention_required": record.attention_required,
        "title": record.envelope.title,
    }


def _inner_artifact_summary(envelope: ProposalEnvelopeV1) -> dict[str, Any]:
    artifact = envelope.artifact
    if envelope.artifact_type == "PatchProposalV1":
        patch = PatchProposalV1.model_validate(artifact)
        return {
            "artifact_type": envelope.artifact_type,
            "problem": patch.problem,
            "proposed_change_summary": patch.proposed_change_summary,
            "files_to_change": patch.files_to_change,
            "rollback_plan": patch.rollback_plan,
        }
    if envelope.artifact_type == "MemoryCorrectionProposalV1":
        correction = MemoryCorrectionProposalV1.model_validate(artifact)
        return {
            "artifact_type": envelope.artifact_type,
            "current_belief": correction.current_belief,
            "proposed_belief": correction.proposed_belief,
            "correction_type": correction.correction_type,
            "rationale": correction.rationale,
            "target_memory_domains": correction.target_memory_domains,
            "rollback_plan": correction.rollback_plan,
        }
    return {
        "artifact_type": envelope.artifact_type,
        "artifact_keys": sorted(artifact.keys()) if isinstance(artifact, dict) else [],
    }


def _api_error(exc: ProposalReviewApiError) -> HTTPException:
    message = str(exc)
    if "required" in message.lower() or "malformed" in message.lower() or "invalid" in message.lower():
        return HTTPException(status_code=503, detail=message)
    if "context-exec" in message.lower():
        return HTTPException(status_code=403, detail=message)
    return HTTPException(status_code=400, detail=message)


class ProposalTriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ProposalTriageAction
    rationale: str
    reviewer_type: ProposalReviewerType = "cortex_policy"


class ProposalReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ProposalReviewDecisionKind
    rationale: str
    reviewer_type: ProposalReviewerType = "human"
    reviewer_id: str = "operator"
    approved_actions: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


@router.get("/proposals")
async def list_proposals(
    status: ProposalStatus | None = Query(default=None),
) -> dict[str, Any]:
    try:
        repo = _open_repo(_require_store_path())
    except ProposalReviewApiError as exc:
        raise _api_error(exc) from exc

    records = repo.list_by_status(status) if status else repo.list_all()
    return {"proposals": [_record_row(record) for record in records], "count": len(records)}


@router.get("/proposals/{proposal_id}")
async def get_proposal(proposal_id: str) -> dict[str, Any]:
    try:
        repo = _open_repo(_require_store_path())
    except ProposalReviewApiError as exc:
        raise _api_error(exc) from exc

    record = repo.get(proposal_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"proposal not found: {proposal_id}")

    review_history = [item.model_dump(mode="json") for item in repo.review_history(proposal_id)]
    latest_review = repo.latest_review_decision(proposal_id)
    eligibility = derive_execution_eligibility(record, latest_review)

    return {
        "proposal_id": record.proposal_id,
        "status": record.status,
        "attention_required": record.attention_required,
        "attention_reason": record.attention_reason,
        "envelope": record.envelope.model_dump(mode="json"),
        "inner_artifact_summary": _inner_artifact_summary(record.envelope),
        "evidence": record.envelope.evidence,
        "risk": record.envelope.risk,
        "review_history": review_history,
        "execution_eligibility": eligibility.model_dump(mode="json"),
    }


@router.post("/proposals/{proposal_id}/triage")
async def triage_proposal(proposal_id: str, body: ProposalTriageRequest) -> dict[str, Any]:
    try:
        repo = _open_repo(_require_store_path())
    except ProposalReviewApiError as exc:
        raise _api_error(exc) from exc

    decision = ProposalTriageDecisionV1(
        proposal_id=proposal_id,
        action=body.action,
        rationale=body.rationale,
        reviewer_type=body.reviewer_type,
        created_at=_utc_now(),
    )
    try:
        updated = repo.apply_triage(decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "proposal_id": updated.proposal_id,
        "status": updated.status,
        "attention_required": updated.attention_required,
        "attention_reason": updated.attention_reason,
        "triage_action": updated.triage_action,
    }


@router.post("/proposals/{proposal_id}/review")
async def review_proposal(proposal_id: str, body: ProposalReviewRequest) -> dict[str, Any]:
    try:
        _parse_reviewer(body.reviewer_type, body.reviewer_id)
        repo = _open_repo(_require_store_path())
    except ProposalReviewApiError as exc:
        raise _api_error(exc) from exc

    decision = ProposalReviewDecisionV1(
        decision_id=f"dec_{uuid.uuid4().hex[:12]}",
        proposal_id=proposal_id,
        decision=body.decision,
        reviewer_type=body.reviewer_type,
        reviewer_id=body.reviewer_id,
        rationale=body.rationale,
        approved_actions=list(body.approved_actions),
        constraints=dict(body.constraints),
        created_at=_utc_now(),
    )
    try:
        updated = repo.apply_review(decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    eligibility = derive_execution_eligibility(updated, decision)
    return {
        "proposal_id": updated.proposal_id,
        "status": updated.status,
        "review_decision": decision.model_dump(mode="json"),
        "execution_eligibility": eligibility.model_dump(mode="json"),
    }


@router.get("/proposals/{proposal_id}/eligibility")
async def proposal_eligibility(proposal_id: str) -> dict[str, Any]:
    try:
        repo = _open_repo(_require_store_path())
    except ProposalReviewApiError as exc:
        raise _api_error(exc) from exc

    record = repo.get(proposal_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"proposal not found: {proposal_id}")

    latest_review = repo.latest_review_decision(proposal_id)
    eligibility = derive_execution_eligibility(record, latest_review)
    return eligibility.model_dump(mode="json")


def proposal_review_health_block() -> dict[str, Any]:
    store = (settings.proposal_ledger_store_path or "").strip()
    configured = bool(store)
    store_ok = False
    store_error: str | None = None
    if configured:
        try:
            _open_repo(Path(store))
            store_ok = True
        except ProposalReviewApiError as exc:
            store_error = str(exc)
    return {
        "enabled": settings.proposal_review_api_enabled,
        "store_configured": configured,
        "store_path": store or None,
        "store_ok": store_ok,
        "store_error": store_error,
    }
