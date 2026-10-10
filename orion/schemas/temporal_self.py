"""Temporal Self: Orion's own day as a chronology of arcs (PR #2369 rev 4, patch 2).

Stored models only. Nothing here is a bus payload in v1, so these live in ``_REGISTRY``
(``resolve()``) and not in ``SCHEMA_REGISTRY``. The reducer that builds them is
``orion/temporal_self/`` (pure, no I/O, no LLM).

Every enum value below has a producer in ``orion/temporal_self/``. Values the spec listed
whose source is dead or missing on 2026-10-10 were dropped rather than kept as names with
no producer (see docs/superpowers/pr-reports/2026-10-10-temporal-self-chronology-reducer-pr.md):

* ``town_exchange`` / arc kind ``town``: zero ``aitown_chat_history_log`` rows since 09-17.
* ``presence_transition`` / arc kind ``company``: ``vision_presence_transition`` (seam S1)
  does not exist.
* ``unresolved_percept`` (1 row ever), ``attention_worthy_sighting`` (walkway not live).
* ``prior_revision`` / ``peer_ask``: FalkorDB reads, deferred to the patch-3 driver.
* ``situation_revision``: Redis keeps only the latest revision, no history to replay.
* ``immune_finding`` / ``arousal_transition`` / ``drive_reading``: regulation lane, not here.
* ``closed_reason='source_stale'``: its only producer was the company lane.
* ``ArcSelfModelSummaryV1``: acceptance check 9 requires reproducing the calibration
  script's 66.0% on the same rows first; those rows aged out (168 h retention). Not bound.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

TEMPORAL_SELF_REDUCER_VERSION = "temporal_self.reducer.v1"

TemporalSourceKind = Literal[
    # process boundaries
    "chat_turn",
    "curiosity_run",
    "reverie_chain",
    "visual_run",
    "dream_cycle",
    "field_dominance_run",
    "attention_loop_raised",
    "attention_loop_verdict",
    # expectations and self-change
    "dream_hypothesis",
    "expectation_verdict",
    "action_outcome",
    # constraints
    "visual_deferral",
    "gpu_wait",
    # context (subject-less, bound by time or by reference)
    "metacog_observation",
    "consolidation_window_close",
    "attention_row",
    "vision_percept",
    "memory_episode",
]

ArcKind = Literal[
    "attention",
    "concern",
    "interoception",
    "conversation",
    "curiosity",
    "reverie",
    "imagery",
    "sleep",
]

ArcStatus = Literal["open", "suspended", "closed"]
ClosedReason = Literal["process_ended", "verdict", "day_boundary", "return_window_expired"]
PrivacyClass = Literal["orion_internal", "juniper_chat"]

LABEL_MAX = 300
EVIDENCE_CAP = 256
CONTEXT_CAP = 64
PERCEPT_CAP = 16
ARCS_IN_FRAME_CAP = 64


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalSelfEventV1(_Forbid):
    """One sparse fact from one source row. Never prose: ``label`` is the row's own label."""

    schema_version: Literal["temporal_self.event.v1"] = "temporal_self.event.v1"
    event_id: str  # deterministic: f"{source_kind}:{source_ref}"
    day_id: str  # local calendar date of occurred_at
    occurred_at: AwareDatetime  # UTC; the cast for each source is documented in sources.py
    # Process events are emitted only once complete, so their interval is known on arrival.
    ended_at: AwareDatetime | None = None
    source_kind: TemporalSourceKind
    source_table: str
    source_ref: str  # the row's own primary key, verbatim
    correlation_id: str | None = None
    subject_ref: str | None = None  # one canonical ref; None for context events
    related_refs: list[str] = Field(default_factory=list)
    label: str = Field(default="", max_length=LABEL_MAX)
    # An OUTWARD boundary only: juniper_chat is readable by every Orion consumer, never sent
    # to contractor peers or published artifacts.
    privacy_class: PrivacyClass = "orion_internal"
    verdict: str | None = None  # the source's own word, never normalised
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "TemporalSelfEventV1":
        if self.event_id != f"{self.source_kind}:{self.source_ref}":
            raise ValueError("event_id must be f'{source_kind}:{source_ref}'")
        if self.ended_at is not None and self.ended_at < self.occurred_at:
            raise ValueError("ended_at precedes occurred_at")
        return self


class ArcAttentionSummaryV1(_Forbid):
    """Folded from substrate_attention_schema rows inside the arc. Counts, never per row."""

    rows_by_lane: dict[str, int] = Field(default_factory=dict)
    reasons_by_lane: dict[str, list[str]] = Field(default_factory=dict)  # unnormalised words


class ArcBodySummaryV1(_Forbid):
    """Existing body sensors at arc altitude (orion/temporal_self/body.py). No feeling claim.

    Two spec fields failed the metric gate's live check on 10-09 and were not built:
    ``peak_pressure_max`` (disk_capacity pins the cluster peak at 0.813 on 78% of rows and
    power saturates it at 1.0, so it can never read calm) and ``cooling_switch_changes``
    (``home_cooling_sample.switch_on`` has been true on every sample since 09-26).
    """

    cluster_sample_count: int = 0
    chassis_watts_mean: float | None = None
    cabinet_sample_count: int = 0
    cabinet_temp_c_min: float | None = None
    cabinet_temp_c_max: float | None = None
    ambient_spike_count: int = 0
    thermal_refusals: int = 0


class ArcSegmentV1(_Forbid):
    """One uninterrupted stretch of an arc. Dwell is the sum of segment spans."""

    began_at: AwareDatetime
    ended_at: AwareDatetime


class TemporalSelfArcV1(_Forbid):
    schema_version: Literal["temporal_self.arc.v1"] = "temporal_self.arc.v1"
    arc_id: str  # sha256(day_id|kind|subject_ref|first_evidence_ref)[:16]
    day_id: str
    kind: ArcKind
    subject_ref: str
    subject_label: str = Field(default="", max_length=LABEL_MAX)
    began_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    last_seen_at: AwareDatetime
    status: ArcStatus
    closed_reason: ClosedReason | None = None
    attention_returns: int = 0  # resumes after a suspension, same day
    cumulative_dwell_sec: float = 0.0
    segments: list[ArcSegmentV1] = Field(default_factory=list)
    interruptions: list[str] = Field(default_factory=list)  # arc_ids that suspended this one
    carried_from_previous_day: bool = False
    carried_from_arc_id: str | None = None
    related_refs: list[str] = Field(default_factory=list)  # e.g. offered prior ids
    evidence_refs: list[str] = Field(default_factory=list)  # refs that SHARE subject_ref
    evidence_overflow: int = 0
    context_event_ids: list[str] = Field(default_factory=list)  # subject-less, bound by time/ref
    context_overflow: int = 0
    expectation_event_ids: list[str] = Field(default_factory=list)
    constraint_event_ids: list[str] = Field(default_factory=list)
    body: ArcBodySummaryV1 | None = None
    attention: ArcAttentionSummaryV1 | None = None
    percept_entities: list[str] = Field(default_factory=list)  # distinct, capped
    warnings: list[str] = Field(default_factory=list)
    reducer_version: str = TEMPORAL_SELF_REDUCER_VERSION

    @model_validator(mode="after")
    def _check(self) -> "TemporalSelfArcV1":
        if len(self.evidence_refs) > EVIDENCE_CAP:
            raise ValueError("evidence_refs over cap")
        if len(self.context_event_ids) > CONTEXT_CAP:
            raise ValueError("context_event_ids over cap")
        if self.status == "closed" and (self.closed_reason is None or self.ended_at is None):
            raise ValueError("a closed arc needs closed_reason and ended_at")
        if self.status != "closed" and self.closed_reason is not None:
            raise ValueError("only a closed arc carries closed_reason")
        if self.ended_at is not None and self.ended_at < self.began_at:
            raise ValueError("ended_at precedes began_at")
        return self


class ArcSummaryV1(_Forbid):
    arc_id: str
    kind: ArcKind
    subject_ref: str
    subject_label: str = ""
    began_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    status: ArcStatus
    attention_returns: int = 0
    cumulative_dwell_sec: float = 0.0


class OpenThreadV1(_Forbid):
    """A concern loop raised in conversation and not yet given a verdict."""

    subject_ref: str
    subject_label: str = ""
    first_seen_today: AwareDatetime
    last_returned: AwareDatetime
    returns_today: int = 0
    carried_from_previous_day: bool = False


class ExpectationRefV1(_Forbid):
    event_id: str
    source_kind: str
    committed_at: AwareDatetime
    resolved_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None  # dream hypotheses expire unoffered after 72 h
    verdict: str | None = None  # the source's own word


class TemporalSelfFrameV1(_Forbid):
    schema_version: Literal["temporal_self.frame.v1"] = "temporal_self.frame.v1"
    frame_id: str
    day_id: str
    as_of: AwareDatetime
    day_phase: str  # TimeContextV1.day_phase vocabulary
    active_arc: ArcSummaryV1 | None = None
    previous_arc: ArcSummaryV1 | None = None
    active_by_kind: dict[str, ArcSummaryV1] = Field(default_factory=dict)
    arcs_today: list[ArcSummaryV1] = Field(default_factory=list)
    arcs_today_total: int = 0
    open_threads: list[OpenThreadV1] = Field(default_factory=list)
    expectations_pending: list[ExpectationRefV1] = Field(default_factory=list)
    expectations_resolved_today: list[ExpectationRefV1] = Field(default_factory=list)
    self_change_event_ids: list[str] = Field(default_factory=list)
    self_change_overflow: int = 0
    constraint_event_ids: list[str] = Field(default_factory=list)
    constraint_overflow: int = 0
    expectations_resolved_total: int = 0  # before the list cap
    sleep_arc_ids: list[str] = Field(default_factory=list)
    # Subject-less context (metacog observations, memory episodes, consolidation closes) that
    # fell inside no arc: rule 7's "or in the day when none is open". Sorted, capped.
    unbound_context_event_ids: list[str] = Field(default_factory=list)
    unbound_context_overflow: int = 0
    entered_day_with: list[str] = Field(default_factory=list)
    source_cursors: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    reducer_version: str = TEMPORAL_SELF_REDUCER_VERSION

    @model_validator(mode="after")
    def _check(self) -> "TemporalSelfFrameV1":
        if len(self.arcs_today) > ARCS_IN_FRAME_CAP:
            raise ValueError("arcs_today over cap")
        return self


class TemporalSelfDayV1(_Forbid):
    schema_version: Literal["temporal_self.day.v1"] = "temporal_self.day.v1"
    day_id: str
    closed_at: AwareDatetime
    frame: TemporalSelfFrameV1  # the final frame of that day
    arcs: list[TemporalSelfArcV1] = Field(default_factory=list)  # full, self-contained


class TemporalSelfContextItemV1(_Forbid):
    """A buffered subject-less event, kept briefly so arcs that complete later can bind it."""

    event_id: str
    at: AwareDatetime
    source_kind: TemporalSourceKind
    correlation_id: str | None = None
    process: str | None = None  # attention_row lane
    reason: str | None = None  # attention_row reason word
    entities: list[str] = Field(default_factory=list)  # vision_percept
    constraint: bool = False
    bound_arc_ids: list[str] = Field(default_factory=list)  # dedupe: bound at most once per arc


class TemporalSelfStateV1(_Forbid):
    """Reducer working state. Fully serialisable so a mid-day checkpoint replays identically."""

    schema_version: Literal["temporal_self.state.v1"] = "temporal_self.state.v1"
    day_id: str | None = None
    watermark: AwareDatetime | None = None
    arcs: dict[str, TemporalSelfArcV1] = Field(default_factory=dict)  # today's (and carried) arcs
    active: dict[str, str] = Field(default_factory=dict)  # lane kind -> active arc_id
    # Attention-lane driver state (broadcast ticks are never stored as events).
    tick_prev_at: AwareDatetime | None = None
    tick_prev_ref: str | None = None
    tick_candidate_ref: str | None = None
    tick_candidate_is_none: bool = False
    tick_candidate_count: int = 0
    tick_candidate_first_at: AwareDatetime | None = None
    tick_candidate_log_ids: list[str] = Field(default_factory=list)
    tick_count_today: int = 0
    tick_winner_count_today: int = 0
    tick_source_gaps_today: int = 0
    tick_winner_refs_today: list[str] = Field(default_factory=list)  # distinct, capped
    # Cross-day continuation: subject -> (arc_id, last_seen) for arcs closed at midnight.
    carry: dict[str, dict[str, str]] = Field(default_factory=dict)
    entered_day_with: list[str] = Field(default_factory=list)
    # Reference bindings that are not arc fields: arc_id -> correlation ids (bounded).
    arc_correlations: dict[str, list[str]] = Field(default_factory=dict)
    context_buffer: list[TemporalSelfContextItemV1] = Field(default_factory=list)
    day_context_event_ids: list[str] = Field(default_factory=list)  # bound to no arc
    day_context_overflow: int = 0
    expectations: dict[str, ExpectationRefV1] = Field(default_factory=dict)
    # By-reference attachments that arrived before their process arc (a dream hypothesis is
    # written seconds before its cycle row's ended_at): f"{kind}|{subject_ref}" -> event ids.
    awaiting_arc: dict[str, list[str]] = Field(default_factory=dict)
    self_change_event_ids: list[str] = Field(default_factory=list)
    self_change_overflow: int = 0
    constraint_event_ids: list[str] = Field(default_factory=list)
    constraint_overflow: int = 0
    # Idempotence: items sorting at or before this key were already folded.
    last_key: list[str] = Field(default_factory=list)
    cursors: dict[str, str] = Field(default_factory=dict)
    pending_closed_days: list[TemporalSelfDayV1] = Field(default_factory=list)
    reducer_version: str = TEMPORAL_SELF_REDUCER_VERSION
