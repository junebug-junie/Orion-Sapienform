from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.drives import TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1

AutonomyStateQuality = Literal[
    "healthy",
    "degraded_drives_timeout",
    "degraded_drives_error",
    "degraded_identity_timeout",
    "degraded_goals_timeout",
    "degraded_partial",
    "empty",
    "unavailable",
]
AutonomyStanceMode = Literal["normal", "proposal_only", "fallback_contextual", "unavailable"]


class AutonomyGoalHeadlineV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_id: str
    goal_statement: str
    drive_origin: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_until: datetime | None = None
    proposal_signature: str
    proposal_status: str = "proposed"
    semantic_source: str | None = None
    planned_task_id: str | None = None
    completed_at: datetime | None = None


class AutonomyActiveGoalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    drive_origin: str
    headline: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_id: str
    proposal_status: str | None = None
    planned_task_id: str | None = None
    completed_at: datetime | None = None


class AutonomyStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    subject: str
    model_layer: str
    entity_id: str
    latest_identity_snapshot_id: str | None = None
    latest_drive_audit_id: str | None = None
    latest_goal_ids: list[str] = Field(default_factory=list)
    identity_summary: str | None = None
    anchor_strategy: str | None = None
    dominant_drive: str | None = None
    active_drives: list[str] = Field(default_factory=list)
    drive_pressures: dict[str, float] = Field(default_factory=dict)
    tension_kinds: list[str] = Field(default_factory=list)
    goal_headlines: list[AutonomyGoalHeadlineV1] = Field(default_factory=list)
    source: str
    generated_at: datetime | None = None


class DriveCompetitionSummaryV1(BaseModel):
    """When tension.drive_competition.v1 is active: which drives disagree and by how much."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    top_drive: str
    runner_drive: str
    spread: float = Field(ge=0.0, le=1.0)
    pressure_top: float = Field(ge=0.0, le=1.0)
    pressure_runner: float = Field(ge=0.0, le=1.0)


class AutonomySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stance_hint: str
    dominant_drive: str | None = None
    top_drives: list[str] = Field(default_factory=list)
    active_tensions: list[str] = Field(default_factory=list)
    proposal_headlines: list[str] = Field(default_factory=list)
    response_hazards: list[str] = Field(default_factory=list)
    raw_state_present: bool = False
    drive_competition: DriveCompetitionSummaryV1 | None = None
    state_quality: AutonomyStateQuality = "empty"
    stance_mode: AutonomyStanceMode = "unavailable"
    degraded_reason: str | None = None
    facet_health: dict[str, str] = Field(default_factory=dict)
    context_note: str | None = None
    selected_subject: str | None = None
    active_goals: list[AutonomyActiveGoalV1] = Field(default_factory=list)
    goals_present: bool = False


class AutonomyEvidenceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evidence_id: str
    source: str
    kind: str
    summary: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observed_at: datetime | None = None


class AttentionItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    item_id: str
    summary: str
    source: str
    salience: float = Field(ge=0.0, le=1.0)
    drive_links: list[str] = Field(default_factory=list)
    tension_links: list[str] = Field(default_factory=list)
    unresolved: bool = True
    evidence_refs: list[str] = Field(default_factory=list)


class CandidateImpulseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    impulse_id: str
    kind: str
    summary: str
    drive_origin: str | None = None
    expected_effect: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


class InhibitedImpulseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    impulse_id: str
    kind: str
    summary: str
    inhibition_reason: str
    risk: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class ActionOutcomeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_id: str
    kind: str
    summary: str
    success: bool | None = None
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None


class ActionOutcomeEmitV1(BaseModel):
    """Bus payload carrying an action outcome for durable persistence via sql-writer.

    Flat shape (subject + outcome fields) so sql-writer's generic row mapper can
    project directly to `action_outcomes` columns.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    subject: str
    action_id: str
    kind: str
    summary: str
    success: bool | None = None
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None

    @classmethod
    def from_outcome(cls, *, subject: str, outcome: ActionOutcomeRefV1) -> "ActionOutcomeEmitV1":
        return cls(
            subject=subject,
            action_id=outcome.action_id,
            kind=outcome.kind,
            summary=outcome.summary,
            success=outcome.success,
            surprise=outcome.surprise,
            observed_at=outcome.observed_at,
        )

    def to_outcome(self) -> ActionOutcomeRefV1:
        return ActionOutcomeRefV1(
            action_id=self.action_id,
            kind=self.kind,
            summary=self.summary,
            success=self.success,
            surprise=self.surprise,
            observed_at=self.observed_at,
        )


class MetabolismResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    drive_deltas: dict[str, float] = Field(default_factory=dict)
    tensions: list[TensionEventV1] = Field(default_factory=list)
    curiosity_signals: list[FrontierInvocationSignalV1] = Field(default_factory=list)


@dataclass(frozen=True)
class SubstrateEpisodeIntentV1:
    goal_artifact_id: str
    drive_origin: str
    spawned_correlation_id: str
    subject: str


class SubstrateActResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fetch_attempted: bool = False
    journal_attempted: bool = False
    fetch_outcome_id: str | None = None
    journal_entry_id: str | None = None
    # Full outcome carried up to the worker so it can emit action.outcome.emit.v1
    # onto the bus for durable, cross-service persistence.
    fetch_outcome: ActionOutcomeRefV1 | None = None


class AutonomyStateDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    subject: str
    changed_fields: list[str] = Field(default_factory=list)
    drive_deltas: dict[str, float] = Field(default_factory=dict)
    new_tensions: list[str] = Field(default_factory=list)
    resolved_tensions: list[str] = Field(default_factory=list)
    new_attention_items: list[str] = Field(default_factory=list)
    new_impulses: list[str] = Field(default_factory=list)
    new_inhibitions: list[str] = Field(default_factory=list)
    confidence_delta: float = 0.0
    notes: list[str] = Field(default_factory=list)


class AutonomyStateV2(AutonomyStateV1):
    """Graph or reducer-produced autonomy snapshot with evidence, attention, and appraisal fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = "autonomy.state.v2"
    evidence_refs: list[AutonomyEvidenceRefV1] = Field(default_factory=list)
    freshness: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    unknowns: list[str] = Field(default_factory=list)
    attention_items: list[AttentionItemV1] = Field(default_factory=list)
    candidate_impulses: list[CandidateImpulseV1] = Field(default_factory=list)
    inhibited_impulses: list[InhibitedImpulseV1] = Field(default_factory=list)
    last_action_outcomes: list[ActionOutcomeRefV1] = Field(default_factory=list)
    previous_state_ref: str | None = None


def upgrade_autonomy_state_v1_to_v2(v1: AutonomyStateV1) -> AutonomyStateV2:
    """Lift a persisted V1 graph row into V2 with synthetic evidence and conservative defaults."""
    evidence_refs: list[AutonomyEvidenceRefV1] = []
    if v1.latest_identity_snapshot_id:
        evidence_refs.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"identity_snapshot:{v1.latest_identity_snapshot_id}",
                source="graph",
                kind="identity_snapshot",
                summary=v1.identity_summary,
                confidence=0.55,
                observed_at=v1.generated_at,
            )
        )
    if v1.latest_drive_audit_id:
        evidence_refs.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"drive_audit:{v1.latest_drive_audit_id}",
                source="graph",
                kind="drive_audit",
                summary=None,
                confidence=0.55,
                observed_at=v1.generated_at,
            )
        )
    for gid in v1.latest_goal_ids:
        evidence_refs.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"goal_ref:{gid}",
                source="graph",
                kind="goal_ref",
                summary=None,
                confidence=0.5,
                observed_at=v1.generated_at,
            )
        )

    unknowns: list[str] = ["no_action_outcome_history", "evidence_from_graph_only"]
    if v1.latest_identity_snapshot_id is None:
        unknowns.append("no_identity_snapshot")
    if v1.latest_drive_audit_id is None:
        unknowns.append("no_drive_audit")

    attention_items: list[AttentionItemV1] = []
    dom = (v1.dominant_drive or "").strip()
    if dom or v1.tension_kinds:
        seed_kind = "attention_seed"
        item_id = hashlib.sha256(f"{v1.subject}:{seed_kind}:{v1.dominant_drive or ''}".encode()).hexdigest()[:16]
        parts: list[str] = []
        if dom:
            parts.append(f"dominant_drive={dom}")
        if v1.tension_kinds:
            parts.append("tensions=" + ",".join(v1.tension_kinds[:6]))
        summary = "; ".join(parts) if parts else "attention"
        attention_items.append(
            AttentionItemV1(
                item_id=item_id,
                summary=summary,
                source="graph_upgrade",
                salience=0.75,
                drive_links=[dom] if dom else [],
                tension_links=list(v1.tension_kinds)[:6],
            )
        )

    core = v1.model_dump()
    core.update(
        {
            "schema_version": "autonomy.state.v2",
            "evidence_refs": [e.model_dump() for e in evidence_refs],
            "freshness": {},
            "confidence": 0.55,
            "unknowns": unknowns,
            "attention_items": [a.model_dump() for a in attention_items],
            "candidate_impulses": [],
            "inhibited_impulses": [],
            "last_action_outcomes": [],
            "previous_state_ref": None,
        }
    )
    return AutonomyStateV2.model_validate(core)


CapabilityDecisionOutcome = Literal["allowed", "denied", "requires_promote"]


class CapabilityPolicyRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capability_id: str
    side_effect_class: Literal["readonly", "write", "external"]
    auto_execute: bool = False
    requires_goal_status: str = "none"
    required_drive_origins: list[str] = Field(default_factory=list)
    required_signal_kinds: list[str] = Field(default_factory=list)
    budget_per_cycle: int = 0


class CapabilityPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str = "v1"
    rules: list[CapabilityPolicyRuleV1] = Field(default_factory=list)


class CapabilityDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capability_id: str
    outcome: CapabilityDecisionOutcome
    reason_code: str
    auto_execute: bool = False
    notes: list[str] = Field(default_factory=list)
