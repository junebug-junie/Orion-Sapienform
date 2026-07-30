from __future__ import annotations

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

    # dominant_drive, active_drives, drive_pressures, tension_kinds, and
    # latest_drive_audit_id were removed 2026-07-30 (chore/delete-orion-drives
    # Wave 2a) -- full removal, not a present-but-always-empty stub, per
    # explicit direction: DriveEngine/DriveAuditV1's publisher were already
    # deleted in Wave 1, so nothing computes these anymore. goal_headlines
    # (and its per-goal drive_origin field) survives: GoalProposalV1 real
    # historical/live readers still exist.
    subject: str
    model_layer: str
    entity_id: str
    latest_identity_snapshot_id: str | None = None
    latest_goal_ids: list[str] = Field(default_factory=list)
    identity_summary: str | None = None
    anchor_strategy: str | None = None
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
    # Optional typed pressure fields. Audit-only refs may omit these.
    # Confidence values remain kind-literal constants in v1 (uncalibrated).
    signal_kind: str | None = None
    dimension: str | None = None
    value: float | None = Field(default=None, ge=0.0, le=1.0)


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


class FetchedArticleRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    url: str
    title: str = ""
    description: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)


class ActionOutcomeRefV1(BaseModel):
    """`surprise` is a mix of one real signal and several fake ones -- check the emitter
    before trusting a row. Confirmed 2026-07-24 by tracing every call site that existed at
    the time (`orion/autonomy/episode_fetch.py`, `orion/autonomy/policy_act.py`,
    `orion/autonomy/curiosity_reuse.py`): all three are a binary success/fail proxy
    (`0.0 if success else 1.0`, or a hardcoded `1.0` for "found vs not found"), never a
    continuous measure of anything -- do not reuse this field as an Active-
    Inference/epistemic-value term for those emitters' rows (see
    `docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md` for the design this
    was ruled out of there).

    As of 2026-07-28, `services/orion-execution-dispatch-runtime` is a genuine exception:
    its rows carry a real, continuous value from `bus_synaptic_prediction_error()`
    (`orion/substrate/prediction_error.py`, live-fixed for a calm-floor bias in PR #1391),
    not a success/fail proxy. That emitter is identifiable by `kind` values
    `inspect`/`summarize`/`observe`/`noop` (`ExecutionDispatchCandidateV1.dispatch_kind`) --
    everything else emitting onto this same route is still the fake proxy described above.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_id: str
    kind: str
    summary: str
    success: bool | None = None
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None
    query: str | None = None
    articles: list[FetchedArticleRefV1] = Field(default_factory=list)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)


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
    # P4: recall-first check, tried before fetch. Mirrors fetch_attempted/
    # fetch_outcome exactly so a successful recall reaches the same bus-emit
    # -> sql-writer -> action_outcomes durability path a fetch success does --
    # without this, a recall success is recorded only via the local
    # append_action_outcome file-store fallback inside policy_act.py, never
    # the durable SQL path load_action_outcomes reads from in production.
    recall_attempted: bool = False
    recall_outcome: ActionOutcomeRefV1 | None = None


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
    # drive_audit evidence ref removed 2026-07-30 (chore/delete-orion-drives
    # Wave 2a): v1.latest_drive_audit_id no longer exists on AutonomyStateV1.
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
    # "no_drive_audit" unknown removed 2026-07-30 (chore/delete-orion-drives
    # Wave 2a): drive_audit is no longer a producible artifact at all, so
    # flagging its absence as an "unknown" would be misleading -- it isn't
    # unknown, it's gone.

    # attention_items seeded from dominant_drive/tension_kinds removed
    # 2026-07-30 (chore/delete-orion-drives Wave 2a): those fields no longer
    # exist on AutonomyStateV1. No replacement attention seed is derived here
    # -- real attention_items now come solely from wherever else populates
    # AutonomyStateV2.attention_items downstream of this upgrade.
    attention_items: list[AttentionItemV1] = []

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
    required_signal_kinds: list[str] = Field(default_factory=list)
    budget_per_cycle: int = 0
    # Real, additive gate on CapabilityEvaluationContext.domain_surprise_score -- see
    # docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md. Unset (None) on
    # every rule initially: ship soft/advisory first (score surfaces in
    # CapabilityDecisionV1.notes regardless of this field), don't hard-gate until the
    # signal's own outcome-correlation is validated against real traffic.
    required_domain_surprise_below: float | None = None


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
