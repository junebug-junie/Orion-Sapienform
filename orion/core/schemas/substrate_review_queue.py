"""Bounded review scheduling and queue contracts (Phase 10)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1

GraphReviewScheduleOutcomeV1 = Literal["enqueue_now", "schedule_later", "suppress", "terminate", "operator_only"]


class GraphReviewCyclePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cycles_world: int = Field(default=6, ge=1, le=24)
    max_cycles_concept: int = Field(default=5, ge=1, le=24)
    max_cycles_autonomy: int = Field(default=4, ge=1, le=24)
    max_cycles_self_relationship: int = Field(default=2, ge=1, le=12)
    urgent_revisit_seconds: int = Field(default=1800, ge=60, le=86400)
    normal_revisit_seconds: int = Field(default=7200, ge=300, le=172800)
    slow_revisit_seconds: int = Field(default=21600, ge=600, le=604800)
    queue_max_items: int = Field(default=200, ge=10, le=2000)
    suppress_after_low_value_cycles: int = Field(default=2, ge=1, le=10)


class GraphReviewCycleBudgetV1(BaseModel):
    """Explicit cycle budget state for inspectable revisit termination controls."""

    model_config = ConfigDict(extra="forbid")

    cycle_count: int = Field(default=0, ge=0)
    max_cycles: int = Field(ge=1)
    remaining_cycles: int = Field(ge=0)
    no_change_cycles: int = Field(default=0, ge=0)
    suppress_after_low_value_cycles: int = Field(default=2, ge=1, le=10)


class GraphReviewQueueItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_item_id: str = Field(default_factory=lambda: f"graph-review-item-{uuid4()}")
    focal_node_refs: List[str] = Field(default_factory=list, max_length=64)
    focal_edge_refs: List[str] = Field(default_factory=list, max_length=128)
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    target_zone: FrontierTargetZoneV1
    originating_decision_id: str
    originating_request_id: str
    reason_for_revisit: str
    priority: int = Field(ge=0, le=100)
    next_review_at: datetime
    cycle_budget: GraphReviewCycleBudgetV1
    suppression_state: bool = False
    termination_state: bool = False
    last_review_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: List[str] = Field(default_factory=list, max_length=32)


class GraphReviewScheduleDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: f"graph-review-sched-{uuid4()}")
    target_refs: List[str] = Field(default_factory=list, max_length=64)
    outcome: GraphReviewScheduleOutcomeV1
    next_review_at: Optional[datetime] = None
    cadence_reason: str
    cycle_budget_reason: str
    priority: int = Field(ge=0, le=100)
    notes: List[str] = Field(default_factory=list, max_length=32)


class GraphReviewQueueSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_items: List[GraphReviewQueueItemV1] = Field(default_factory=list)
    counts_by_zone: Dict[str, int] = Field(default_factory=dict)
    counts_by_state: Dict[str, int] = Field(default_factory=dict)
    top_priorities: List[int] = Field(default_factory=list)
    truncated: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
