"""Typed frontier landing governance contracts (Phase 7)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.core.schemas.cognitive_substrate import SubstrateRiskTierV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1

FrontierLandingDecisionKindV1 = Literal[
    "reject",
    "proposed_only",
    "provisional",
    "materialize_now",
    "hitl_required",
    "blocked_due_to_zone",
    "blocked_due_to_schema",
    "blocked_due_to_risk",
    "blocked_due_to_conflict",
]

FrontierBlockedReasonV1 = Literal["zone", "schema", "risk", "conflict", "policy"]


class FrontierLandingRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    landing_request_id: str = Field(default_factory=lambda: f"frontier-land-req-{uuid4()}")
    bundle_id: str
    request_id: str
    correlation_id: Optional[str] = None
    target_zone: FrontierTargetZoneV1
    triggering_source: Optional[str] = None
    landing_context: Dict[str, Any] = Field(default_factory=dict)
    current_substrate_node_refs: List[str] = Field(default_factory=list, max_length=128)
    current_substrate_edge_refs: List[str] = Field(default_factory=list, max_length=256)
    graph_cognition_brief_refs: List[str] = Field(default_factory=list, max_length=32)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FrontierDeltaLandingDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_item_id: str
    decision: FrontierLandingDecisionKindV1
    target_zone: FrontierTargetZoneV1
    suggested_promotion_state: Literal["proposed", "provisional", "canonical", "rejected"] = "proposed"
    hitl_required: bool = False
    blocked_reason: Optional[FrontierBlockedReasonV1] = None
    materialize_now: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    risk_tier: SubstrateRiskTierV1
    notes: List[str] = Field(default_factory=list, max_length=32)
    linked_candidate_refs: List[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "FrontierDeltaLandingDecisionV1":
        blocked = self.decision.startswith("blocked_due_to_")
        if blocked and self.blocked_reason is None:
            raise ValueError("blocked_reason required for blocked decisions")
        if not blocked and self.blocked_reason is not None:
            raise ValueError("blocked_reason must be null for non-blocked decisions")
        if self.materialize_now and self.decision != "materialize_now":
            raise ValueError("materialize_now can only be true when decision=materialize_now")
        if self.decision == "hitl_required" and not self.hitl_required:
            raise ValueError("hitl_required must be true when decision=hitl_required")
        return self


class FrontierLandingResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    landing_result_id: str = Field(default_factory=lambda: f"frontier-land-res-{uuid4()}")
    bundle_id: str
    request_id: str
    target_zone: FrontierTargetZoneV1
    decisions: List[FrontierDeltaLandingDecisionV1]
    outcome_counts: Dict[str, int]
    hitl_summary: Dict[str, int]
    materialization_summary: Dict[str, int]
    blocked_summary: Dict[str, int]
    confidence: float = Field(ge=0.0, le=1.0)
    degraded: bool = False
    notes: List[str] = Field(default_factory=list, max_length=64)
