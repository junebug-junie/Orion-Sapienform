"""Hub routes proxying the context-exec proposal review API."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from scripts import proposal_review_client as client

logger = logging.getLogger("orion-hub.proposal-review")

router = APIRouter(prefix="/api/proposal-review", tags=["proposal-review"])

ProposalListFilter = Literal[
    "pending_review",
    "blocked",
    "stored",
    "approved",
    "rejected",
    "request_changes",
]

ProposalReviewDecision = Literal["approve", "reject", "request_changes"]


class ProposalReviewActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ProposalReviewDecision
    rationale: str
    reviewer_type: Literal["human"] = "human"
    reviewer_id: str = "hub-operator"
    approved_actions: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


def _disabled_payload(*, state: str = "disabled") -> dict[str, Any]:
    return {
        "enabled": False,
        "available": False,
        "state": state,
        "proposals": [],
        "count": 0,
    }


def _unavailable_payload(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "available": False,
        "state": "unavailable",
        "message": "Proposal review API unavailable.",
        "proposals": [],
        "count": 0,
    }


@router.get("/health")
async def proposal_review_health() -> dict[str, Any]:
    if not client.is_enabled():
        return {**_disabled_payload(), "upstream": None}

    try:
        upstream = await client.fetch_health()
    except client.ProposalReviewUnavailable:
        return _unavailable_payload()
    except client.ProposalReviewClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    block = upstream.get("proposal_review_api") if isinstance(upstream, dict) else None
    available = bool(isinstance(block, dict) and block.get("ok"))
    return {
        "enabled": True,
        "available": available,
        "state": "ok" if available else "unavailable",
        "upstream": upstream,
    }


@router.get("/pending")
async def proposal_review_pending(
    status: ProposalListFilter = Query(default="pending_review"),
) -> dict[str, Any]:
    if not client.is_enabled():
        return _disabled_payload()

    try:
        payload = await client.list_proposals(status=status)
    except client.ProposalReviewUnavailable:
        return _unavailable_payload()
    except client.ProposalReviewClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    proposals = payload.get("proposals") if isinstance(payload, dict) else []
    if not isinstance(proposals, list):
        proposals = []
    return {
        "enabled": True,
        "available": True,
        "state": "ok",
        "status_filter": status,
        "proposals": proposals,
        "count": len(proposals),
    }


@router.get("/proposals/{proposal_id}")
async def proposal_review_detail(proposal_id: str) -> dict[str, Any]:
    if not client.is_enabled():
        raise HTTPException(status_code=503, detail="proposal_review_disabled")

    try:
        return await client.get_proposal(proposal_id)
    except client.ProposalReviewUnavailable as exc:
        raise HTTPException(status_code=503, detail="proposal_review_unavailable") from exc
    except client.ProposalReviewClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/proposals/{proposal_id}/eligibility")
async def proposal_review_eligibility(proposal_id: str) -> dict[str, Any]:
    if not client.is_enabled():
        raise HTTPException(status_code=503, detail="proposal_review_disabled")

    try:
        return await client.get_eligibility(proposal_id)
    except client.ProposalReviewUnavailable as exc:
        raise HTTPException(status_code=503, detail="proposal_review_unavailable") from exc
    except client.ProposalReviewClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/proposals/{proposal_id}/review")
async def proposal_review_action(proposal_id: str, body: ProposalReviewActionRequest) -> dict[str, Any]:
    if not client.is_enabled():
        raise HTTPException(status_code=503, detail="proposal_review_disabled")

    rationale = body.rationale.strip()
    if not rationale:
        raise HTTPException(status_code=422, detail="rationale is required")

    payload = body.model_dump(mode="json")
    payload["rationale"] = rationale

    try:
        return await client.post_review(proposal_id, payload)
    except client.ProposalReviewUnavailable as exc:
        raise HTTPException(status_code=503, detail="proposal_review_unavailable") from exc
    except client.ProposalReviewClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
