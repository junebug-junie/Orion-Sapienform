"""Canonical epistemic reasoning schemas (Phase 1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AnchorScope = Literal["orion", "juniper", "relationship", "world", "session"]
ReasoningStatus = Literal["proposed", "provisional", "canonical", "deprecated", "rejected"]
ReasoningAuthority = Literal[
    "sensed",
    "user_asserted",
    "local_inferred",
    "mentor_inferred",
    "human_verified",
]
RiskTier = Literal["low", "medium", "high"]
ArtifactType = Literal[
    "observation",
    "claim",
    "concept",
    "relation",
    "contradiction",
    "self_model_facet",
    "relationship_facet",
    "world_model_facet",
    "autonomy_drive_audit",
    "goal_proposal",
    "verb_evaluation",
    "mentor_proposal",
    "promotion_decision",
    "spark_state_snapshot",
]
ReasoningEdgePredicate = Literal[
    "supports",
    "contradicts",
    "refines",
    "supersedes",
    "clusters_with",
    "grounds",
    "proposes",
    "triggered_by",
]

RelationType = Literal[
    "supports",
    "causes",
    "refines",
    "conflicts_with",
    "reframes",
    "suggests_goal",
    "related_to",
]
ContradictionType = Literal[
    "logical_conflict",
    "evidence_conflict",
    "temporal_conflict",
    "scope_conflict",
    "identity_conflict",
]
ContradictionSeverity = Literal["low", "medium", "high", "critical"]
ContradictionResolution = Literal["open", "under_review", "resolved", "deferred"]
MentorTaskType = Literal[
    "ontology_cleanup",
    "contradiction_review",
    "concept_refinement",
    "autonomy_review",
    "goal_critique",
    "verb_eval_review",
    "missing_evidence_scan",
]
PromotionAction = Literal["accepted", "deferred", "downgraded", "rejected", "escalated_hitl"]
VerbOutcome = Literal["helpful", "neutral", "harmful", "unknown"]


class ReasoningProvenanceV1(BaseModel):
    """Reusable provenance lineage for reasoning artifacts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evidence_refs: List[str] = Field(default_factory=list)
    source_channel: str
    source_kind: str
    producer: str
    model: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    message_id: Optional[str] = None


class ReasoningEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    predicate: ReasoningEdgePredicate
    target_id: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class ReasoningArtifactBaseV1(BaseModel):
    """Shared typed envelope for canonical reasoning artifacts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    artifact_id: str = Field(default_factory=lambda: f"artifact-{uuid4()}")
    artifact_type: ArtifactType
    anchor_scope: AnchorScope
    subject_ref: Optional[str] = Field(
        default=None,
        description="Dynamic entity/domain reference (e.g. person:alex, project:orion_sapienform).",
    )
    status: ReasoningStatus = "proposed"
    authority: ReasoningAuthority
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_tier: RiskTier = "low"
    observed_at: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    provenance: ReasoningProvenanceV1
    edges: List[ReasoningEdgeV1] = Field(default_factory=list)

    @field_validator("observed_at", "valid_from", "valid_to")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def _validate_window(self) -> "ReasoningArtifactBaseV1":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")
        return self


class ClaimV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["claim"] = "claim"

    claim_text: str
    claim_kind: str = "assertion"
    qualifiers: Dict[str, Any] = Field(default_factory=dict)


class ConceptV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["concept"] = "concept"

    concept_id: str
    label: str = Field(min_length=1)
    aliases: List[str] = Field(default_factory=list)
    concept_type: str = "unknown"
    cluster_refs: List[str] = Field(default_factory=list)
    source_family: str = "concept_induction"
    source_artifact_ref: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["relation"] = "relation"

    source_ref: str
    target_ref: str
    relation_type: RelationType
    directed: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContradictionV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["contradiction"] = "contradiction"

    contradiction_type: ContradictionType
    severity: ContradictionSeverity
    resolution_status: ContradictionResolution = "open"
    involved_artifact_ids: List[str] = Field(min_length=2)
    summary: str


class MentorProposalV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["mentor_proposal"] = "mentor_proposal"
    status: Literal["proposed"] = "proposed"
    authority: Literal["mentor_inferred"] = "mentor_inferred"

    mentor_provider: str
    mentor_model: str
    task_type: MentorTaskType
    proposal_type: str
    rationale: str
    suggested_payload: Dict[str, Any] = Field(default_factory=dict)


class PromotionDecisionV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["promotion_decision"] = "promotion_decision"
    status: Literal["canonical"] = "canonical"

    proposal_artifact_id: str
    action: PromotionAction
    rationale: str
    decided_by: str
    escalated_to_hitl: bool = False


class VerbEvaluationV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["verb_evaluation"] = "verb_evaluation"

    verb_name: str
    execution_depth: int = Field(ge=0, le=3)
    outcome: VerbOutcome = "unknown"
    utility_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    notes: Optional[str] = None


class ReasoningSparkStateSnapshotV1(ReasoningArtifactBaseV1):
    artifact_type: Literal["spark_state_snapshot"] = "spark_state_snapshot"

    dimensions: Dict[str, float] = Field(default_factory=dict)
    tensions: List[str] = Field(default_factory=list)
    attention_targets: List[str] = Field(default_factory=list)
    trend_window_s: Optional[int] = Field(default=None, ge=0)


ClaimV1.model_rebuild()
ConceptV1.model_rebuild()
RelationV1.model_rebuild()
ContradictionV1.model_rebuild()
MentorProposalV1.model_rebuild()
PromotionDecisionV1.model_rebuild()
VerbEvaluationV1.model_rebuild()
ReasoningSparkStateSnapshotV1.model_rebuild()

# Compatibility alias for phase plans that reference SparkStateSnapshotV1 in the reasoning family.
SparkStateSnapshotV1 = ReasoningSparkStateSnapshotV1
