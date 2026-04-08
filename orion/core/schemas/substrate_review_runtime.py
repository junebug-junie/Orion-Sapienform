"""Runtime execution contracts for bounded graph-review cycles (Phase 11)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1

GraphReviewRuntimeSurfaceV1 = Literal["operator_review", "chat_reflective_lane"]
GraphReviewRuntimeOutcomeV1 = Literal["executed", "noop", "suppressed", "terminated", "operator_only", "failed"]


class GraphReviewRuntimeRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"graph-review-runtime-req-{uuid4()}")
    invocation_surface: GraphReviewRuntimeSurfaceV1
    correlation_id: Optional[str] = None
    explicit_queue_item_id: Optional[str] = None
    anchor_scope: Optional[SubstrateAnchorScopeV1] = None
    subject_ref: Optional[str] = None
    max_items_to_consider: int = Field(default=20, ge=1, le=100)
    execute_frontier_followup_allowed: bool = False
    operator_override_strict_zone: bool = False


class GraphReviewRuntimeResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    correlation_id: Optional[str] = None
    selected_queue_item_id: Optional[str] = None
    outcome: GraphReviewRuntimeOutcomeV1
    consolidation_result_ref: Optional[str] = None
    queue_update_summary: Dict[str, object] = Field(default_factory=dict)
    cycle_budget_summary: Dict[str, object] = Field(default_factory=dict)
    frontier_followup_invoked: bool = False
    audit_summary: Dict[str, object] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list, max_length=64)
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
