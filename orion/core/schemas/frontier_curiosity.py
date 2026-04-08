"""Curiosity/gap-driven frontier invocation contracts (Phase 8)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1
from orion.core.schemas.frontier_expansion import FrontierExpansionRequestV1, FrontierTargetZoneV1, FrontierTaskTypeV1

FrontierInvocationSignalTypeV1 = Literal[
    "ontology_sparse_region",
    "concept_instability",
    "contradiction_hotspot",
    "evidence_gap_cluster",
    "unresolved_pressure_region",
    "explicit_operator_request",
    "curiosity_candidate",
]

FrontierInvocationOutcomeV1 = Literal["invoke", "defer", "noop", "blocked", "operator_only"]


class FrontierInvocationSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(default_factory=lambda: f"frontier-signal-{uuid4()}")
    signal_type: FrontierInvocationSignalTypeV1
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    target_zone: FrontierTargetZoneV1
    task_type_candidate: FrontierTaskTypeV1
    focal_node_refs: List[str] = Field(default_factory=list, max_length=32)
    focal_edge_refs: List[str] = Field(default_factory=list, max_length=64)
    signal_strength: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list, max_length=16)


class FrontierInvocationDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: f"frontier-decision-{uuid4()}")
    outcome: FrontierInvocationOutcomeV1
    chosen_task_type: Optional[FrontierTaskTypeV1] = None
    target_zone: Optional[FrontierTargetZoneV1] = None
    chosen_focal_node_refs: List[str] = Field(default_factory=list, max_length=32)
    chosen_focal_edge_refs: List[str] = Field(default_factory=list, max_length=64)
    bounded_context_reason: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0)
    block_reason: Optional[str] = None
    notes: List[str] = Field(default_factory=list, max_length=32)


class FrontierInvocationPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: f"frontier-plan-{uuid4()}")
    request_payload_summary: str
    selected_node_refs: List[str] = Field(default_factory=list, max_length=32)
    selected_edge_refs: List[str] = Field(default_factory=list, max_length=64)
    selected_evidence_refs: List[str] = Field(default_factory=list, max_length=128)
    selected_graph_cognition_refs: List[str] = Field(default_factory=list, max_length=16)
    task_type: FrontierTaskTypeV1
    target_zone: FrontierTargetZoneV1
    expected_safety_posture: str
    request: FrontierExpansionRequestV1


class FrontierInvocationRunResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signals: List[FrontierInvocationSignalV1] = Field(default_factory=list)
    decision: FrontierInvocationDecisionV1
    plan: Optional[FrontierInvocationPlanV1] = None
    notes: List[str] = Field(default_factory=list)
