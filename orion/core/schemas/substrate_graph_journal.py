"""The shared append-only substrate graph journal: proposal -> decision -> materialization.

Generalizes #2497's proposed ``ReadingGraph{Proposal,Decision,Materialization}V1``
(docs/plans/substrate/2026-10-06-reading-property-graph-design.md, "Persistence and
ownership") to ``SubstrateGraph*V1`` with a ``proposal_kind``, so reading and memory
keep one audit trail instead of two. Spec:
docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md, section 1.4.

These are Postgres row contracts (table ``substrate_graph_journal``), not bus
payloads. #2497 also proposed ``orion:*:graph:*`` bus channels; none ship here
because nothing subscribes to them yet.

Producer -> consumer:
- Proposal: whoever proposes a claim (memory referents, PR B; the reading pipeline,
  later) -> ``AssertionProjector`` reads it to build the Assertion node and edges.
- Decision: the acceptance step (a named policy such as ``source_cooccurrence_v1``,
  or a reviewer) -> ``AssertionProjector`` applies it as the assertion's new state
  and revision.
- Materialization: ``AssertionProjector`` -> ``SubstrateGraphJournal.pending_decisions``
  (a decision with no applied materialization is still pending) and the failed-row
  de-duplication in the projector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orion.core.schemas.cognitive_substrate import (
    SubstrateAnchorScopeV1,
    SubstrateAuthorityV1,
    SubstrateEdgePredicateV1,
    SubstratePromotionStateV1,
)

# One value per kind something consumes. relationship_assertion: AssertionProjector.
# (memory Stage 2 PR B adds referent_identity together with its consumer.)
SubstrateGraphProposalKindV1 = Literal["relationship_assertion"]
SemanticEndpointKindV1 = Literal["concept", "entity"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _JournalEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=3)
    proposal_kind: SubstrateGraphProposalKindV1
    # The node the event is about: for relationship_assertion, the Assertion node id.
    target_id: str = Field(min_length=3)
    actor: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=_utcnow)

    @field_validator("recorded_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class SubstrateGraphProposalV1(_JournalEventV1):
    """A claim that two semantic nodes are related, with the evidence behind it."""

    event_kind: Literal["proposal"] = "proposal"
    subject_node_id: str = Field(min_length=3)
    subject_kind: SemanticEndpointKindV1
    object_node_id: str = Field(min_length=3)
    object_kind: SemanticEndpointKindV1
    predicate: SubstrateEdgePredicateV1
    # Immutable resolved subject|predicate|object|context; one Assertion per key.
    statement_key: str = Field(min_length=3)
    statement_text: str = Field(min_length=1)
    anchor_scope: SubstrateAnchorScopeV1
    subject_ref: Optional[str] = None
    authority: SubstrateAuthorityV1
    # Evidence node ids that support the claim; each becomes an Evidence -supports-> Assertion edge.
    supporting_evidence_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> "SubstrateGraphProposalV1":
        if self.subject_node_id == self.object_node_id:
            raise ValueError("an assertion relates two different nodes")
        if self.predicate in {"assertion_subject", "assertion_object"}:
            raise ValueError("assertion structure predicates are not claims")
        return self

    @property
    def event_id(self) -> str:
        return self.proposal_id


class SubstrateGraphDecisionV1(_JournalEventV1):
    """Acceptance, rejection or retraction of a proposal, under a named policy."""

    event_kind: Literal["decision"] = "decision"
    decision_id: str = Field(min_length=3)
    # Optimistic concurrency: the assertion revision this decision was made against.
    expected_prior_revision: int = Field(ge=0)
    resulting_state: SubstratePromotionStateV1
    # The rule or reviewer that decided, e.g. "source_cooccurrence_v1", "operator_review".
    policy: str = Field(min_length=1)
    authority: SubstrateAuthorityV1
    rationale: str = ""
    evidence_refs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _state(self) -> "SubstrateGraphDecisionV1":
        if self.resulting_state == "proposed":
            raise ValueError("a decision moves a proposal out of 'proposed'")
        return self

    @property
    def resulting_revision(self) -> int:
        return self.expected_prior_revision + 1

    @property
    def event_id(self) -> str:
        return self.decision_id


class SubstrateGraphMaterializationV1(_JournalEventV1):
    """What the projector actually wrote for a decision, with the canonical ids it got back."""

    event_kind: Literal["materialization"] = "materialization"
    materialization_id: str = Field(min_length=3)
    decision_id: str = Field(min_length=3)
    revision: int = Field(ge=1)
    outcome: Literal["applied", "failed"]
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None

    @model_validator(mode="after")
    def _reason(self) -> "SubstrateGraphMaterializationV1":
        if (self.outcome == "failed") != bool(self.failure_reason):
            raise ValueError("failure_reason is required exactly when outcome='failed'")
        return self

    @property
    def event_id(self) -> str:
        return self.materialization_id
