from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


AttentionTargetTypeV1 = Literal[
    "person",
    "place",
    "activity",
    "plan",
    "relation",
    "belief",
    "object",
    "concept",
    "anomaly",
    "memory_gap",
    "future_event",
    "other",
]
CuriosityActionTypeV1 = Literal["ask", "reflect", "remember", "defer", "watch", "suppress", "none"]
CuriositySuppressionReasonV1 = Literal[
    "generic_reciprocity",
    "already_known",
    "too_many_questions",
    "user_needs_direct_answer",
    "low_value_question",
    "vague_broad_question",
    "no_conversational_bandwidth",
    "unsafe_or_sensitive",
    "stale_thread",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SalienceFeaturesV1(BaseModel):
    """Evidence-derived feature vector scored by the salience combiner.

    Replaces the hand-tuned constant ladder. `habituation` is a penalty term
    (higher = more habituated = lower salience) applied subtractively by the
    combiner. All features are bounded [0,1].
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.salience.features.v1"] = "attention.salience.features.v1"
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_breadth: float = Field(default=0.0, ge=0.0, le=1.0)
    recurrence: float = Field(default=0.0, ge=0.0, le=1.0)
    recency: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_vs_known: float = Field(default=0.0, ge=0.0, le=1.0)
    dwell: float = Field(default=0.0, ge=0.0, le=1.0)
    habituation: float = Field(default=0.0, ge=0.0, le=1.0)


class OpenLoopV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    target_type: AttentionTargetTypeV1 = "other"
    description: str
    source_text: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    why_it_matters: str = ""
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    relational_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    predictive_value: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_value: float = Field(default=0.0, ge=0.0, le=1.0)
    autonomy_value: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_charge: float = Field(default=0.0, ge=0.0, le=1.0)
    already_known: bool = False
    askability: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    # Salience v2 (additive, back-compatible). The 7 legacy score fields above
    # remain populated for one deprecation release; new code reads these.
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    salience_features: dict[str, Any] = Field(default_factory=dict)
    # Voluntary attention (additive, back-compatible). top_down_bias is the
    # goal-derived bias applied to this loop; combined_salience = salience +
    # gain·applied_bias. Both default 0.0 -> pure bottom-up when the feature is off.
    top_down_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    combined_salience: float = Field(default=0.0, ge=0.0, le=1.0)


class VoluntaryOverrideV1(BaseModel):
    """Recorded when top-down goal bias makes a lower-bottom-up loop win — an
    inspectable act of volitional attention (Desimone & Duncan biased competition)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    goal_artifact_id: str | None = None
    goal_drive_origin: str | None = None
    chosen_loop_id: str
    beat_loop_id: str
    chosen_bottom_up: float = Field(ge=0.0, le=1.0)
    beat_bottom_up: float = Field(ge=0.0, le=1.0)
    applied_bias: float = Field(ge=0.0, le=1.0)
    effort_spent: float = Field(ge=0.0)


class CuriosityCandidateActionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_type: CuriosityActionTypeV1
    open_loop_id: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    question_text: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class CuriositySuppressionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reason: CuriositySuppressionReasonV1
    target_ref: str | None = None
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AttentionSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    signal_id: str
    source: str
    target_text: str
    target_type_hint: str = "other"
    signal_kind: str
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

class AttentionFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.frame.v1"] = "attention.frame.v1"
    generated_at: datetime = Field(default_factory=_utc_now)
    turn_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    open_loops: list[OpenLoopV1] = Field(default_factory=list)
    live_unknowns: list[str] = Field(default_factory=list)
    candidate_actions: list[CuriosityCandidateActionV1] = Field(default_factory=list)
    selected_action: CuriosityCandidateActionV1 | None = None
    suppressions: list[CuriositySuppressionV1] = Field(default_factory=list)
    deferred_items: list[str] = Field(default_factory=list)
    # Voluntary attention (additive). Set when top-down goal bias flipped the
    # winner; None when selection was pure bottom-up (default -> current behavior).
    voluntary_override: VoluntaryOverrideV1 | None = None
    effort_budget_used: float = Field(default=0.0, ge=0.0)
    debug: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class AttentionBroadcastProjectionV1(BaseModel):
    """Current substrate-wide attention (rung 3): the selected coalition of the
    latest workspace competition, re-broadcast as a single queryable projection.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.broadcast.projection.v1"] = "attention.broadcast.projection.v1"
    projection_id: str = "substrate.attention.broadcast.v1"
    generated_at: datetime = Field(default_factory=_utc_now)
    frame: AttentionFrameV1
    selected_action_type: str = "none"
    selected_open_loop_id: str | None = None
    selected_description: str | None = None
    attended_node_ids: list[str] = Field(default_factory=list)
    dwell_ticks: int = 0
    coalition_stability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    coalition_history: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
