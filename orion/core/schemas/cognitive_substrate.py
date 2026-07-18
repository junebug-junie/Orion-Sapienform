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


# Default half-life for concept-node activation decay, used by ConceptNodeV1's
# auto-seed validator below when a producer doesn't set one explicitly. Mirrors
# orion.memory.crystallization's own precedent for the separate crystallization
# system (orion/memory/crystallization/schemas.py: decay_half_life_days = 30.0).
DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS = 30 * 24 * 60 * 60


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
    tier_rank: Optional[int] = Field(default=None, description="Trust tier rank (1=operator_static … 4=snapshot_ephemeral). Lower = higher authority. Omit for pre-tier nodes.")


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

    @model_validator(mode="after")
    def _seed_activation_if_unset(self) -> "ConceptNodeV1":
        """Auto-seed activation/half-life when a producer left `signals.activation`
        at the pure schema default (activation=0.0, decay_half_life_seconds=None).

        Every real ConceptNodeV1 producer historically left this sub-object
        untouched, which made Hub's live decay scheduler
        (services/orion-hub/scripts/api_routes.py::decay_concept_activations())
        a permanent no-op: decay_activation() treats a falsy half-life as "clamp
        to floor, don't decay." Enforcing the seed here, at the schema boundary,
        means every current and future producer gets working decay for free
        instead of each adapter needing to remember to call a helper by hand
        (16+ live construction sites were found still missing it when this was
        first patched at the adapter level alone -- see
        orion/substrate/adapters/_common.py::make_activation() for the
        opt-in helper adapters can still use to seed a non-default value
        explicitly, e.g. from a real confidence/salience prior).

        Only fires when BOTH fields are still at their literal schema
        defaults, so a producer that explicitly sets activation (even to
        0.0 with a real half-life, or vice versa) is never overridden.
        """
        activation = self.signals.activation
        if activation.activation == 0.0 and activation.decay_half_life_seconds is None:
            seeded = activation.model_copy(
                update={
                    "activation": self.signals.salience,
                    "decay_half_life_seconds": DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS,
                }
            )
            self.signals = self.signals.model_copy(update={"activation": seeded})
        return self


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
