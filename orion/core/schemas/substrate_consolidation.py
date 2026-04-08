"""Bounded reflective graph consolidation contracts (Phase 9)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1

GraphConsolidationOutcomeV1 = Literal[
    "reinforce",
    "keep_provisional",
    "requeue_review",
    "damp",
    "retire",
    "maintain_priority",
    "noop",
    "operator_only",
]


class GraphConsolidationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"graph-cons-req-{uuid4()}")
    correlation_id: Optional[str] = None
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    focal_node_refs: List[str] = Field(default_factory=list, max_length=64)
    focal_edge_refs: List[str] = Field(default_factory=list, max_length=128)
    reason_for_review: str = Field(min_length=3)
    triggering_signal_refs: List[str] = Field(default_factory=list, max_length=32)
    prior_cycle_refs: List[str] = Field(default_factory=list, max_length=32)
    time_window_seconds: int = Field(default=86400, ge=60, le=2_592_000)
    target_zone: FrontierTargetZoneV1
    bounded_context_refs: List[str] = Field(default_factory=list, max_length=64)


class GraphStateDeltaDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_persistence_ratio: float = Field(ge=0.0, le=1.0)
    edge_persistence_ratio: float = Field(ge=0.0, le=1.0)
    activation_delta: float
    pressure_delta: float
    contradiction_delta: int
    evidence_gap_delta: int
    isolated_frontier_delta: int


class GraphConsolidationDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: f"graph-cons-decision-{uuid4()}")
    target_refs: List[str] = Field(default_factory=list, max_length=64)
    outcome: GraphConsolidationOutcomeV1
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    zone: FrontierTargetZoneV1
    priority: int = Field(default=0, ge=0, le=100)
    notes: List[str] = Field(default_factory=list, max_length=32)
    evidence_summary: str = ""


class GraphConsolidationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    decisions: List[GraphConsolidationDecisionV1] = Field(default_factory=list)
    outcome_counts: Dict[str, int] = Field(default_factory=dict)
    regions_reviewed: List[str] = Field(default_factory=list)
    unresolved_regions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    degraded: bool = False
    notes: List[str] = Field(default_factory=list, max_length=64)
    state_delta_digest: Optional[GraphStateDeltaDigestV1] = None


class GraphReviewCycleRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_id: str = Field(default_factory=lambda: f"graph-review-cycle-{uuid4()}")
    request_id: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    focal_node_refs: List[str] = Field(default_factory=list)
    focal_edge_refs: List[str] = Field(default_factory=list)
    mean_activation: float = Field(ge=0.0, le=1.0)
    mean_pressure: float = Field(ge=0.0, le=1.0)
    contradiction_count: int = Field(ge=0)
    evidence_gap_count: int = Field(ge=0)
    isolated_frontier_count: int = Field(ge=0)
    outcome_counts: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
