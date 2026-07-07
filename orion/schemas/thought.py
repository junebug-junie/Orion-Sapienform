from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


class CoalitionSnapshotV1(BaseModel):
    """Typed subset of AttentionBroadcastProjectionV1 at draft emit time."""

    schema_version: Literal["coalition.snapshot.v1"] = "coalition.snapshot.v1"
    attended_node_ids: list[str]
    selected_open_loop_id: str | None
    open_loop_ids: list[str]
    generated_at: datetime
    broadcast_stale: bool = False


class StanceHarnessSliceV1(BaseModel):
    schema_version: Literal["stance.harness.slice.v1"] = "stance.harness.slice.v1"
    task_mode: str
    conversation_frame: str
    interaction_regime: str | None = None
    response_priorities: list[str] = Field(default_factory=list)
    response_hazards: list[str] = Field(default_factory=list)
    answer_strategy: str
    companion_closing_move: str | None = None


class HubAssociationBundleV1(BaseModel):
    schema_version: Literal["hub.association.bundle.v1"] = "hub.association.bundle.v1"
    correlation_id: str
    broadcast: AttentionBroadcastProjectionV1 | None
    broadcast_stale: bool
    execution_trajectory_slice: dict[str, Any] | None = None
    repair_bundle: TurnAppraisalBundleV1 | None = None
    read_source: Literal["felt_state_reader", "hub_sql_fallback"]


class GroundingCapsuleV1(BaseModel):
    """Bounded self-context for the unified turn: identity + relationship + policy + PCR digests."""

    schema_version: Literal["grounding.capsule.v1"] = "grounding.capsule.v1"
    identity_summary: list[str] = Field(default_factory=list)
    relationship_summary: list[str] = Field(default_factory=list)
    response_policy_summary: list[str] = Field(default_factory=list)
    continuity_digest: str | None = None
    belief_digest: str | None = None
    memory_digest: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ThoughtEventV1(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    event_id: str
    correlation_id: str
    session_id: str | None
    created_at: datetime
    profile: Literal["stance_react"] = "stance_react"

    imperative: str = Field(max_length=300)
    tone: str = Field(max_length=200)
    strain_refs: list[str]

    # Fail-closed defer when empty — enforced in stance_quality / policy_refusal, not at parse.
    evidence_refs: list[str] = Field(default_factory=list)
    repair_pressure_level: float | None = None
    trust_rupture_score: float | None = None

    disposition: Literal["proceed", "defer", "refuse"] = "proceed"
    disposition_reasons: list[str] = Field(default_factory=list)
    boundary_register: bool = False

    stance_harness_slice: StanceHarnessSliceV1
    grounding_capsule: GroundingCapsuleV1 | None = None

    llm_profile: str = "brain"
    producer: str = "stance_react_v1"
    model_id: str | None = None


class StanceReactRequestV1(BaseModel):
    schema_version: Literal["stance.react.request.v1"] = "stance.react.request.v1"
    correlation_id: str
    session_id: str | None
    user_message: str
    association: HubAssociationBundleV1
    repair_bundle: TurnAppraisalBundleV1 | None
    stance_inputs: dict[str, Any]
    llm_profile: str = "brain"


def __getattr__(name: str) -> object:
    if name == "GrammarReceiptV1":
        from orion.schemas.harness_finalize import GrammarReceiptV1

        return GrammarReceiptV1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
