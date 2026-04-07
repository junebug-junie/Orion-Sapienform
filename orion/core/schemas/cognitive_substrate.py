"""Unified cognitive substrate contracts (Phase 1 foundation)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SubstrateNodeKindV1 = Literal[
    "entity",
    "concept",
    "event",
    "evidence",
    "contradiction",
    "tension",
    "drive",
    "goal",
    "state_snapshot",
    "hypothesis",
    "ontology_branch",
]
SubstrateEdgePredicateV1 = Literal[
    "supports",
    "contradicts",
    "refines",
    "causes",
    "associated_with",
    "observed_in",
    "activates",
    "suppresses",
    "seeks",
    "blocks",
    "satisfies",
    "part_of",
    "subtype_of",
    "instance_of",
    "co_occurs_with",
]
SubstratePromotionStateV1 = Literal["proposed", "provisional", "canonical", "deprecated", "rejected"]
SubstrateAnchorScopeV1 = Literal["orion", "juniper", "relationship", "world", "session"]
SubstrateRiskTierV1 = Literal["low", "medium", "high"]
SubstrateAuthorityV1 = Literal[
    "sensed",
    "user_asserted",
    "local_inferred",
    "mentor_inferred",
    "human_verified",
]


class NodeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=3)
    node_kind: SubstrateNodeKindV1


class EdgeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(min_length=3)


class SubjectRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_ref: str = Field(min_length=3)


class EvidenceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=3)


class SubstrateActivationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    decay_half_life_seconds: Optional[int] = Field(default=None, ge=1)
    decay_floor: float = Field(default=0.0, ge=0.0, le=1.0)


class SubstrateTemporalWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    @field_validator("observed_at", "valid_from", "valid_to")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def _validate_window(self) -> "SubstrateTemporalWindowV1":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")
        return self


class SubstrateProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    authority: SubstrateAuthorityV1
    source_kind: str
    source_channel: str
    producer: str
    model_name: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)


class SubstrateSignalBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    activation: SubstrateActivationV1 = Field(default_factory=SubstrateActivationV1)


class BaseSubstrateNodeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(default_factory=lambda: f"sub-node-{uuid4()}")
    node_kind: SubstrateNodeKindV1
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    promotion_state: SubstratePromotionStateV1 = "proposed"
    risk_tier: SubstrateRiskTierV1 = "low"
    temporal: SubstrateTemporalWindowV1
    signals: SubstrateSignalBundleV1 = Field(default_factory=SubstrateSignalBundleV1)
    provenance: SubstrateProvenanceV1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["entity"] = "entity"
    entity_type: str = Field(default="unknown", min_length=1)
    label: str = Field(min_length=1)
    aliases: List[str] = Field(default_factory=list)


class ConceptNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["concept"] = "concept"
    label: str = Field(min_length=1)
    definition: Optional[str] = None
    taxonomy_path: List[str] = Field(default_factory=list)


class EventNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["event"] = "event"
    event_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class EvidenceNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["evidence"] = "evidence"
    evidence_type: str = Field(min_length=1)
    content_ref: str = Field(min_length=1)


class ContradictionNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["contradiction"] = "contradiction"
    contradiction_type: str = Field(default="logical_conflict", min_length=1)
    summary: str = Field(min_length=1)
    involved_node_ids: List[str] = Field(min_length=2)


class TensionNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["tension"] = "tension"
    tension_kind: str = Field(min_length=1)
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)


class DriveNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["drive"] = "drive"
    drive_kind: str = Field(min_length=1)
    target_state: Optional[str] = None


class GoalNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["goal"] = "goal"
    goal_text: str = Field(min_length=1)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class StateSnapshotNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["state_snapshot"] = "state_snapshot"
    dimensions: Dict[str, float] = Field(default_factory=dict)
    snapshot_source: str = Field(default="unknown")


class HypothesisNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["hypothesis"] = "hypothesis"
    hypothesis_text: str = Field(min_length=1)
    test_status: Literal["unverified", "supported", "contradicted"] = "unverified"


class OntologyBranchNodeV1(BaseSubstrateNodeV1):
    node_kind: Literal["ontology_branch"] = "ontology_branch"
    branch_key: str = Field(min_length=1)
    branch_label: str = Field(min_length=1)


SubstrateNodeV1 = Annotated[
    Union[
        EntityNodeV1,
        ConceptNodeV1,
        EventNodeV1,
        EvidenceNodeV1,
        ContradictionNodeV1,
        TensionNodeV1,
        DriveNodeV1,
        GoalNodeV1,
        StateSnapshotNodeV1,
        HypothesisNodeV1,
        OntologyBranchNodeV1,
    ],
    Field(discriminator="node_kind"),
]


class SubstrateEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(default_factory=lambda: f"sub-edge-{uuid4()}")
    source: NodeRefV1
    target: NodeRefV1
    predicate: SubstrateEdgePredicateV1
    temporal: SubstrateTemporalWindowV1
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: SubstrateProvenanceV1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubstrateGraphRecordV1(BaseModel):
    """Batch envelope for canonical substrate writes/reads."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(default_factory=lambda: f"sub-graph-{uuid4()}")
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    nodes: List[SubstrateNodeV1] = Field(default_factory=list)
    edges: List[SubstrateEdgeV1] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


EntityNodeV1.model_rebuild()
ConceptNodeV1.model_rebuild()
EventNodeV1.model_rebuild()
EvidenceNodeV1.model_rebuild()
ContradictionNodeV1.model_rebuild()
TensionNodeV1.model_rebuild()
DriveNodeV1.model_rebuild()
GoalNodeV1.model_rebuild()
StateSnapshotNodeV1.model_rebuild()
HypothesisNodeV1.model_rebuild()
OntologyBranchNodeV1.model_rebuild()
SubstrateGraphRecordV1.model_rebuild()
