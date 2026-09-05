from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    """The two real, evidence-derived features Borda rank-aggregated into
    coalition-strength salience (see `orion.substrate.attention.salience`).

    Trimmed 2026-07-31 (kill means kill, CLAUDE.md Sec 0A) from a 7-field
    vector: `recurrence`, `recency`, `novelty_vs_known`, `dwell`, and
    `habituation` had no real theory anchor (recency's 6h half-life and
    dwell/habituation's blend weights were picked, not measured;
    novelty_vs_known collapsed to a flat 0.15 for any known loop) and were
    killed with nothing put back rather than kept as always-zero fields
    (that would be exactly the "empty-shell"/fake-precision pattern this
    trim exists to avoid). `habituation` was, as far as this investigation
    found, the only automatic repeat-suppression mechanism in this scoring
    path -- its removal is a real, disclosed capability gap, not silently
    absorbed. See `orion/sentience_striving_program/README.md`'s
    2026-07-31 entry for the full rationale.

    `evidence_strength` (the strongest single detector's own real
    activation) and `evidence_breadth` (how many independent
    detectors/evidence_refs corroborate this loop) are both real,
    already-shipped signals -- not hand-picked -- and map onto Global
    Workspace Theory / Society-of-Mind coalition formation (Baars 1988,
    Dehaene 2014).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.salience.features.v1"] = "attention.salience.features.v1"
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_breadth: float = Field(default=0.0, ge=0.0, le=1.0)


class OpenLoopV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    target_type: AttentionTargetTypeV1 = "other"
    description: str
    source_text: str | None = None
    # LOAD-BEARING for goal-directed attention since 2026-09-05: relevance()
    # (orion/substrate/attention/top_down.py) matches GoalContext.target_id
    # against this list, so a substrate loop must carry its originating
    # `node:substrate.*` id here or no goal can ever bias it. Populated from
    # AttentionSignalV1.evidence_refs in scoring.py::build_open_loops, which
    # substrate_pressure_signals seeds with the node id. Changing the shape,
    # order, or prefix of these ids silently reverts voluntary override to
    # impossible -- pinned by test_relevance_joins_a_real_substrate_node.
    source_refs: list[str] = Field(default_factory=list)
    why_it_matters: str = ""
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    relational_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    predictive_value: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_value: float = Field(default=0.0, ge=0.0, le=1.0)
    autonomy_value: float = Field(default=0.0, ge=0.0, le=1.0)
    already_known: bool = False
    askability: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    # Salience v2 (additive, back-compatible). The legacy score fields above
    # remain populated for one deprecation release; new code reads these.
    #
    # emotional_charge REMOVED 2026-08-25 (kill means kill, CLAUDE.md Sec
    # 0A) -- it was a bare regex over 11 emotion-adjacent words
    # (_EMOTION_RE in orion/substrate/attention/scoring.py, added
    # 2026-05-16) computed on every chat turn, but its only reader
    # (score_loop()'s old hand-tuned formula, `+ loop.emotional_charge *
    # 0.07`) was deleted 2026-07-31 in the same salience-v2 rewrite that
    # trimmed SalienceFeaturesV1's other untethered fields (see that
    # class's own docstring) -- that PR missed this sibling field on
    # OpenLoopV1. Confirmed dead by full-repo grep before removal: no
    # scoring, persistence (attention_salience_trace never stored it), or
    # UI/debug consumer ever read it again after 2026-07-31. Found while
    # investigating a suspected competing architecture for reading
    # Juniper's affect -- this predates and was never reconciled with
    # either JuniperAffectiveStateV1 (orion-cocreation-signals) or
    # JuniperMultimodalAffectV1 (AffectGPT, PR #1865/#1871).
    #
    # novelty/continuity_relevance/relational_relevance appear similarly
    # unread by the same grep (code review, 2026-08-25, corrected an
    # earlier draft of this note that wrongly claimed `novelty` was still
    # live -- it is not; `loop.novelty`/`OpenLoopV1.novelty` has no reader
    # anywhere, only a comment describing the deleted pre-v2 formula that
    # used to reference it). predictive_value/autonomy_value ARE confirmed
    # still live: orion/substrate/attention/policy.py reads
    # loop.autonomy_value/loop.predictive_value directly in its ask-gating
    # condition.
    #
    # concept_value JOINED THE UNREAD SET 2026-09-05. It was live only
    # because top_down.py's relevance() read it -- and that was the defect:
    # relevance() took a goal, never read it, and returned this field, which
    # scoring.py had floored to a constant 0.55 for every substrate loop. So
    # top-down bias was identical across candidates and voluntary override was
    # mathematically impossible. relevance() now joins the goal's target to
    # `source_refs` (see top_down.py::relevance) and nothing reads
    # concept_value any more -- the producer at scoring.py still writes its
    # real value. Retiring the field was NOT in this patch's scope; recorded
    # here as a disclosed follow-up so the next cleanup does not re-derive
    # this from scratch or trust the stale claim above.
    # novelty/continuity_relevance/relational_relevance were NOT part of
    # this patch's scope/approval -- flagged as a disclosed follow-up, not
    # removed here.
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    salience_features: dict[str, Any] = Field(default_factory=dict)
    # Voluntary attention (additive, back-compatible). top_down_bias is the
    # goal-derived bias applied to this loop; combined_salience = salience +
    # gain·applied_bias. Both default 0.0 -> pure bottom-up when the feature is off.
    top_down_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    combined_salience: float = Field(default=0.0, ge=0.0, le=1.0)

    # Fields removed from this model over time -- stripped from the raw
    # payload BEFORE strict extra="forbid" validation runs, so a historical
    # row still carrying one of these keys can still be replayed/read
    # instead of raising. Review finding, 2026-08-25: OpenLoopV1 nests
    # inside AttentionBroadcastProjectionV1 (orion/substrate/attention_
    # broadcast.py), which is persisted as JSONB in both
    # substrate_attention_broadcast_projection (live singleton) and
    # substrate_attention_broadcast_log (168h append-only history) --
    # without this, removing emotional_charge broke model_validate() on
    # every pre-removal stored row: scripts/analysis/measure_ast_hot_
    # reducer.py's replay silently counted every one as a skip (not
    # surfaced as a schema break), and orion/hub/association.py /
    # services/orion-thought/app/broadcast_reader.py would silently
    # degrade to "no coalition"/stale during any window where a
    # not-yet-redeployed producer writes a row shaped the old way. Add to
    # this set (never remove an entry, even after the field it names is
    # long gone from live traffic -- 168h of already-written history keeps
    # needing it) the next time a field is removed from this model.
    # ClassVar, not a field: an underscore-prefixed annotated attribute on a
    # pydantic v2 BaseModel becomes a PrivateAttr by default, which is only
    # reliably accessible on an INSTANCE (after __init__) -- but this needs
    # to be readable from `cls` inside a mode="before" validator, which
    # runs before any instance exists. ClassVar opts out of pydantic's
    # field/private-attr machinery entirely, so it stays a plain class
    # attribute.
    _REMOVED_LEGACY_FIELDS: ClassVar[frozenset[str]] = frozenset({"emotional_charge"})

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(k in data for k in cls._REMOVED_LEGACY_FIELDS):
            return {k: v for k, v in data.items() if k not in cls._REMOVED_LEGACY_FIELDS}
        return data


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

# Why no `voluntary_override` was recorded. Each value except the last maps 1:1
# to a real exit in `orion/substrate/attention_broadcast.py::
# _apply_voluntary_attention`; the last is set by the reducer, which is noted on
# the value itself. This is a record of which branch ran, not a taxonomy
# invented on top of one. Added 2026-09-04 because the stored self-model said an
# override had not happened and nothing said *why*: the `bottom_up_salience`
# branch had won 19,408 of 19,408 ticks over seven days and the cause was
# unrecoverable, only guessable (CLAUDE.md Sec 0A -- "an aggregate that cannot
# name a cause will hide one").
#
# Three of the producer exits are indistinguishable from the frame alone -- all
# leave `effort_budget_used` at 0.0 with no per-loop bias set -- which is
# exactly why the reason has to be recorded at the site that knows it rather
# than derived downstream.
#
# **Where this actually lives, and why not a typed frame field.** It rides in
# `AttentionFrameV1.debug` under `VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY`, and is
# typed only on `AttentionSelfModelV1`, where it is stored and read. A typed
# field on the frame was the first implementation and was reverted in review:
# `AttentionFrameV1` sets `extra="forbid"` and crosses the bus nested inside
# `HubAssociationBundleV1.broadcast` -> `StanceReactRequestV1` on
# `orion:thought:request` (orion-hub -> orion-thought), so any consumer not
# redeployed in the same window would reject the whole payload. One of those
# windows fails at `logger.debug` (`orion/hub/association.py:130`), dropping
# the attention broadcast on every turn essentially invisibly. `debug` is an
# open dict that already exists on the frame, so old consumers accept it
# unchanged and the deploy-order hazard disappears entirely. The typed contract
# still exists -- on the schema that actually stores and serves this value.
VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY = "voluntary_override_absent_reason"


VoluntaryOverrideAbsentReasonV1 = Literal[
    # ORION_ATTENTION_TOPDOWN_ENABLED is off; the combiner never ran.
    "top_down_disabled",
    # `get_active_goal()` returned None -- nothing to bias toward.
    "no_active_goal",
    # The frame carried no open loops; there was no competition to bias.
    "no_open_loops",
    # The combiner ran and top-down bias did NOT change the winner
    # (top_down.py Rule 6). Bottom-up would have chosen the same loop.
    "bias_did_not_flip_winner",
    # Split out of bias_did_not_flip_winner 2026-09-05. That one string was
    # covering three outcomes that mean different things:
    #   goal_matched_no_loop        -- the goal is about something that is not
    #                                  competing at all; every bias is 0.0.
    #                                  Says nothing about Orion's agency, only
    #                                  about goal/attention overlap (measured
    #                                  live at ~22% of competitive ticks).
    #   goal_target_already_winning -- the goal's target was ALREADY the
    #                                  bottom-up winner. The goal got what it
    #                                  wanted; there was nothing to override.
    #   bias_did_not_flip_winner    -- bias landed on a loop that then LOST.
    #                                  The only one of the three that is a real
    #                                  competitive defeat.
    # Measured on the first 44 post-fix ticks: 33 of 43 non-firing ticks were
    # actually goal_matched_no_loop, so reading the old aggregate as "goals
    # keep losing" was wrong for three quarters of it.
    "goal_matched_no_loop",
    "goal_target_already_winning",
    # Top-down DID flip the winner, but that loop had no candidate action to
    # re-point to, so the frame refused to claim an override it cannot enact.
    "winner_had_no_action",
    # `_apply_voluntary_attention` raised and was swallowed by its own
    # never-raises guard. Distinct from every value above: this is a defect
    # signal, not a normal outcome.
    "combiner_error",
    # Reducer-only. Never set by the producer: the frame is the thing being
    # read. Set by `reduce_attention_self_model()` when the broadcast lane was
    # absent or stale, so no producer-side reason exists to report. (Not "or
    # carried no frame" -- `AttentionBroadcastProjectionV1.frame` is a required
    # field and cannot be None.) Kept in this one Literal so a consumer has a
    # single field to switch on rather than two half-answers.
    "broadcast_lane_unreadable",
]


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
