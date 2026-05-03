"""Orion Mind wire contracts (Pydantic only)."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MindTriggerV1 = Literal["user_turn", "scheduled", "operator", "replay"]
TrustLevelV1 = Literal["low", "med", "high"]
HypothesisTagV1 = Literal["assumption", "hypothesis", "simulation_result"]
MergePolicyV1 = Literal["deterministic_merge"]
BindingV1 = Literal["advisory", "mandatory"]


class MindRunPolicyV1(BaseModel):
    n_loops_max: int = Field(default=1, ge=1, le=32)
    wall_time_ms_max: int = Field(default=120_000, ge=1, le=3_600_000)
    llm_enabled_per_loop: list[bool] = Field(default_factory=list)
    router_profile_id: str = Field(default="default")


class MindRunRequestV1(BaseModel):
    schema_version: Literal["mind.run.v1"] = "mind.run.v1"
    correlation_id: str
    session_id: str | None = None
    trace_id: str | None = None
    trigger: MindTriggerV1 = "user_turn"
    snapshot_inputs: dict[str, Any] = Field(default_factory=dict)
    policy: MindRunPolicyV1 = Field(default_factory=MindRunPolicyV1)
    upstream_artifacts: dict[str, Any] | None = None


class MindSnapshotFacetV1(BaseModel):
    trust: TrustLevelV1 = "med"
    source: str = "orch"
    compact_json: dict[str, Any] = Field(default_factory=dict)
    bytes_approx: int = Field(ge=0, default=0)


class MindUniverseSnapshotV1(BaseModel):
    schema_version: Literal["mind.universe.v1"] = "mind.universe.v1"
    facets: dict[str, MindSnapshotFacetV1] = Field(default_factory=dict)
    total_bytes_approx: int = Field(ge=0, default=0)


class MindHypothesisV1(BaseModel):
    tag: HypothesisTagV1 = "hypothesis"
    text: str = ""


class MindProvenanceV1(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = "deterministic"
    temperature: float = 0.0
    input_hash: str = ""


class MindStancePatchV1(BaseModel):
    loop_index: int = Field(ge=0)
    structured: dict[str, Any] = Field(default_factory=dict)
    narrative_notes: str | None = None
    noop_reason: str | None = None
    hypotheses: list[MindHypothesisV1] = Field(default_factory=list)
    provenance: MindProvenanceV1 = Field(default_factory=MindProvenanceV1)


class MindStanceTrajectoryV1(BaseModel):
    patches: list[MindStancePatchV1] = Field(default_factory=list)
    merged_stance_brief: dict[str, Any] = Field(default_factory=dict)
    merge_policy: MergePolicyV1 = "deterministic_merge"


class MindControlDecisionV1(BaseModel):
    route_kind: str = "no_chat"
    allowed_verbs: list[str] = Field(default_factory=list)
    recall_profile_override: str | None = None
    mode_suggestion: Literal["brain", "agent", "workflow_only", "no_chat"] = "brain"
    mode_binding: BindingV1 = "advisory"
    budgets: dict[str, Any] = Field(default_factory=dict)
    refusals: list[dict[str, Any]] = Field(default_factory=list)


class MindHandoffBriefV1(BaseModel):
    summary_one_paragraph: str | None = None
    machine_contract: dict[str, Any] = Field(default_factory=dict)
    mandatory_keys: list[str] = Field(default_factory=list)
    advisory_keys: list[str] = Field(default_factory=list)
    stance_payload: dict[str, Any] = Field(default_factory=dict)


class MindRunResultV1(BaseModel):
    mind_run_id: UUID
    ok: bool
    error_code: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    snapshot_hash: str = ""
    trajectory: MindStanceTrajectoryV1 = Field(default_factory=MindStanceTrajectoryV1)
    decision: MindControlDecisionV1 = Field(default_factory=MindControlDecisionV1)
    brief: MindHandoffBriefV1 = Field(default_factory=MindHandoffBriefV1)
    timing_ms_by_phase: dict[str, float] = Field(default_factory=dict)
