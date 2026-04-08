"""Frontier expansion contracts for typed substrate graph-delta generation (Phase 6)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateAnchorScopeV1,
    SubstrateEdgeV1,
)

FrontierTaskTypeV1 = Literal[
    "ontology_expand",
    "concept_expand",
    "relation_discovery",
    "contradiction_discovery",
    "evidence_gap_scan",
    "taxonomy_proposal",
    "world_fact_hypothesis",
    "autonomy_hypothesis",
    "self_or_relationship_hypothesis",
]

FrontierTargetZoneV1 = Literal[
    "world_ontology",
    "concept_graph",
    "autonomy_graph",
    "self_relationship_graph",
]

FrontierDeltaItemKindV1 = Literal[
    "node_add",
    "edge_add",
    "node_refine",
    "edge_refine",
    "contradiction_flag",
    "evidence_gap",
    "taxonomy_branch",
    "relation_hypothesis",
    "fact_hypothesis",
]

LandingPostureV1 = Literal[
    "fast_track_proposal",
    "moderate_proposal",
    "conservative_proposal",
    "strict_proposal_only",
]


class FrontierSourceProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_authority: Literal["frontier_model", "teacher_model"] = "frontier_model"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FrontierGraphRegionRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focal_node_ids: List[str] = Field(default_factory=list, max_length=64)
    focal_edge_ids: List[str] = Field(default_factory=list, max_length=128)
    max_hops: int = Field(default=2, ge=0, le=4)


class FrontierContextRefsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_ids: List[str] = Field(default_factory=list, max_length=64)
    entity_refs: List[str] = Field(default_factory=list, max_length=64)
    contradiction_refs: List[str] = Field(default_factory=list, max_length=64)
    evidence_refs: List[str] = Field(default_factory=list, max_length=128)


class FrontierExpansionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"frontier-req-{uuid4()}")
    correlation_id: Optional[str] = None
    task_type: FrontierTaskTypeV1
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    target_zone: FrontierTargetZoneV1
    topic: str = Field(min_length=3)
    expansion_goal: str = Field(min_length=3)
    context_refs: FrontierContextRefsV1 = Field(default_factory=FrontierContextRefsV1)
    graph_region: FrontierGraphRegionRefV1 = Field(default_factory=FrontierGraphRegionRefV1)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FrontierDeltaItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default_factory=lambda: f"frontier-item-{uuid4()}")
    item_kind: FrontierDeltaItemKindV1
    candidate_node: Optional[BaseSubstrateNodeV1] = None
    candidate_edge: Optional[SubstrateEdgeV1] = None
    contradiction_summary: Optional[str] = None
    evidence_gap_question: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "FrontierDeltaItemV1":
        if self.item_kind in {"node_add", "node_refine", "taxonomy_branch", "fact_hypothesis"} and self.candidate_node is None:
            raise ValueError("candidate_node required for node-oriented frontier delta kinds")
        if self.item_kind in {"edge_add", "edge_refine", "relation_hypothesis"} and self.candidate_edge is None:
            raise ValueError("candidate_edge required for edge-oriented frontier delta kinds")
        if self.item_kind == "contradiction_flag" and not self.contradiction_summary:
            raise ValueError("contradiction_summary required for contradiction_flag")
        if self.item_kind == "evidence_gap" and not self.evidence_gap_question:
            raise ValueError("evidence_gap_question required for evidence_gap")
        return self


class FrontierExpansionResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    response_id: str = Field(default_factory=lambda: f"frontier-res-{uuid4()}")
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    task_type: FrontierTaskTypeV1
    target_zone: FrontierTargetZoneV1
    delta_items: List[FrontierDeltaItemV1] = Field(min_length=1, max_length=256)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale_notes: List[str] = Field(default_factory=list, max_length=32)
    evidence_gap_flags: List[str] = Field(default_factory=list, max_length=64)


class FrontierGraphDeltaBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str = Field(default_factory=lambda: f"frontier-bundle-{uuid4()}")
    request_id: str
    response_id: str
    target_zone: FrontierTargetZoneV1
    task_type: FrontierTaskTypeV1
    suggested_landing_posture: LandingPostureV1
    candidate_nodes: List[BaseSubstrateNodeV1] = Field(default_factory=list, max_length=256)
    candidate_edges: List[SubstrateEdgeV1] = Field(default_factory=list, max_length=512)
    contradiction_candidates: List[str] = Field(default_factory=list, max_length=128)
    evidence_gap_candidates: List[str] = Field(default_factory=list, max_length=128)
    source_provenance: FrontierSourceProvenanceV1
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list, max_length=64)
