"""Temporal Self: Orion's own day as a chronology of arcs (PR #2369 rev 4, patch 2).

Stored models only. Nothing here is a bus payload in v1, so these live in ``_REGISTRY``
(``resolve()``) and not in ``SCHEMA_REGISTRY``. The reducer that builds them is
``orion/temporal_self/`` (pure, no I/O, no LLM).

Every enum value below has a producer in ``orion/temporal_self/``. Values the spec listed
whose source is dead or missing on 2026-10-10 were dropped rather than kept as names with
no producer (see docs/superpowers/pr-reports/2026-10-10-temporal-self-chronology-reducer-pr.md):

* ``town_exchange`` / arc kind ``town``: zero ``aitown_chat_history_log`` rows since 09-17.
* ``unresolved_percept`` (1 row ever), ``attention_worthy_sighting`` (walkway not live).
* ``prior_revision`` / ``peer_ask``: FalkorDB reads, deferred to the patch-3 driver.
* ``situation_revision``: Redis keeps only the latest revision, no history to replay.
* ``immune_finding`` / ``drive_reading``: regulation lane, not here.
* ``ArcSelfModelSummaryV1``: acceptance check 9 requires reproducing the calibration
  script's 66.0% on the same rows first; those rows aged out (168 h retention). Not bound.

Seam S1 (2026-10-14) added ``presence_transition`` / arc kind ``company`` / closed reason
``source_stale`` once ``vision_presence_transition`` existed. Consumer-first rollout: these
models forbid unknown values, so every reader of frames/arcs (orion-thought, orion-hub) must
deploy before the producer (orion-durable-runs) folds a company arc.

Strain loop part 1 (2026-10-20, docs/superpowers/specs/2026-10-20-strain-loop-design.md) added
``arousal_transition`` / arc kind ``strain`` and ``episode_began_at``: the regulation step's own
level-change and cause rows (already in ``temporal_self_event``) become one arc per strain
episode. Same consumer-first rollout: orion-thought and orion-hub before orion-durable-runs.

Orion plays Zelda (2026-10-22, docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md
section F) added ``retro_play`` / arc kind ``play``: one finished Retro session
(``retro_play_sessions``, written by orion-hub) is one process arc, labelled with its deterministic
summary. Same consumer-first rollout: orion-thought and orion-hub before orion-durable-runs.

``causes_now`` (2026-10-23, on the arc and its summary) follows the same order: every stored frame
carries it (``null`` on non-strain arcs), so an orion-thought or orion-hub still on older code
rejects the whole frame and the stance cue goes dark until they are rebuilt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

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
    # the room camera's presence state machine (seam S1)
    "presence_transition",
    # the regulation step's strain level changes and cause rows (strain loop part 1)
    "arousal_transition",
    # one finished Retro (game) session, orion-hub's retro_play_sessions (Orion plays Zelda, F)
    "retro_play",
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
    "company",
    "strain",
    "play",
]

ArcStatus = Literal["open", "suspended", "closed"]
# source_stale: the instrument went dark (a company arc closed by ``unknown`` or a producer restart).
ClosedReason = Literal["process_ended", "verdict", "day_boundary", "return_window_expired", "source_stale"]
PrivacyClass = Literal["orion_internal", "juniper_chat"]

LABEL_MAX = 300
EVIDENCE_CAP = 256
CONTEXT_CAP = 64
SEGMENTS_CAP = 256
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

    @field_validator("occurred_at", "ended_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        # One clock for ordering keys and ids: any aware input is converted to UTC.
        return v.astimezone(timezone.utc) if v is not None else None

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
    # Resumes after a broadcast RECORDER gap (outage/restart): not attention, so not a return.
    source_gap_resumes: int = 0
    suspended_by_gap: bool = False
    cumulative_dwell_sec: float = 0.0
    segments: list[ArcSegmentV1] = Field(default_factory=list)
    segments_merged: int = 0  # point segments folded into their neighbour past the cap
    interruptions: list[str] = Field(default_factory=list)  # arc_ids that suspended this one
    interruptions_overflow: int = 0
    # OUTWARD boundary only: juniper_chat when the label or subject comes from Juniper's turns.
    privacy_class: PrivacyClass = "orion_internal"
    carried_from_previous_day: bool = False
    carried_from_arc_id: str | None = None
    # A strain arc carried over midnight begins at the day boundary; this keeps when the episode
    # itself began (the regulation reading's ``since``). None for every other kind.
    episode_began_at: AwareDatetime | None = None
    # A strain arc's causes present at its newest row (the label keeps every cause seen in the
    # episode). [] while a strain only holds before clearing; None when not recorded (every other
    # kind, and arcs folded before this field existed).
    causes_now: list[str] | None = None
    related_refs: list[str] = Field(default_factory=list)  # e.g. offered prior ids
    evidence_refs: list[str] = Field(default_factory=list)  # refs that SHARE subject_ref
    evidence_overflow: int = 0
    context_event_ids: list[str] = Field(default_factory=list)  # subject-less, bound by time/ref
    context_overflow: int = 0
    expectation_event_ids: list[str] = Field(default_factory=list)
    expectation_overflow: int = 0
    constraint_event_ids: list[str] = Field(default_factory=list)
    constraint_overflow: int = 0
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
        if len(self.segments) > SEGMENTS_CAP:
            raise ValueError("segments over cap")
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
    # Patch 4: the arc's newest evidence ("last turn 3 min ago"). Optional so frames and closed
    # days stored before it still validate; readers render the clause only when it is present.
    last_seen_at: AwareDatetime | None = None
    # Strain loop part 1: when a strain episode began, even if that was on an earlier day.
    episode_began_at: AwareDatetime | None = None
    # Strain loop: the causes present at the strain arc's newest row (see TemporalSelfArcV1).
    causes_now: list[str] | None = None


class OpenThreadV1(_Forbid):
    """A concern loop raised in conversation and not yet given a verdict."""

    # The concern arc this thread is (``TemporalSelfArcV1.arc_id``), so every reader can cite it
    # as ``[arc:<id>]`` even when the arc fell off ``arcs_today``'s cap. Optional: frames and
    # closed days stored before it still validate (readers fall back to a subject_ref lookup).
    # Consumer-first rollout: readers (orion-thought, orion-hub) deploy before the producer
    # (orion-durable-runs), since this model forbids unknown fields.
    arc_id: str | None = None
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
    # Items that sorted at or before the fold's watermark and were not folded (a driver
    # re-read, or a row committed late with an older timestamp). Visible, never silent.
    skipped_at_or_before_watermark: int = 0
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
    skipped_today: int = 0
    correlation_overflow: int = 0
    cursors: dict[str, str] = Field(default_factory=dict)
    pending_closed_days: list[TemporalSelfDayV1] = Field(default_factory=list)
    reducer_version: str = TEMPORAL_SELF_REDUCER_VERSION


# --- Patch 4: Orion uses its own day (readers + citation gate) --------------------------------
# Spec: docs/superpowers/specs/2026-10-11-temporal-self-patch4-orion-uses-its-day.md. Both
# renderers are deterministic templates over stored rows (no LLM); every clause carries the
# ``[arc:<id>]`` it came from, and whatever Orion writes back is checked against what it was shown.

CueSkipReason = Literal["disabled", "fetch_failed", "render_failed", "invalid_frame", "wrong_day", "stale", "clock_skew"]
DayCitationLabel = Literal["verified", "UNVERIFIED", "no_day_claims"]
STANCE_CUE_MAX_CHARS = 400
DAY_BLOCK_MAX_CHARS = 2400
UNCITED_SENTENCES_SHOWN = 5


class TemporalSelfStanceCueV1(_Forbid):
    """The "where am I in my day" line shown to stance_react. ``text`` is "" when skipped."""

    schema_version: Literal["temporal_self.stance_cue.v1"] = "temporal_self.stance_cue.v1"
    text: str = Field(default="", max_length=STANCE_CUE_MAX_CHARS)
    day_id: str | None = None
    frame_as_of: AwareDatetime | None = None
    rendered_at: AwareDatetime
    arc_ids: list[str] = Field(default_factory=list)  # every [arc:<id>] in ``text``, in order
    # Shown subject label -> the arc ids it names (the citation gate's match set, not a word list).
    labels: dict[str, list[str]] = Field(default_factory=dict)
    skip_reason: CueSkipReason | None = None
    clauses_dropped: int = 0  # dropped to fit STANCE_CUE_MAX_CHARS, or for a silent broadcast log


class TemporalSelfDayBlockV1(_Forbid):
    """A closed day, rendered for a reader of "yesterday" (Orion's Day letter).

    ``records`` maps every ref in ``text`` (``arc:<id>``, ``day:<day_id>``) to a compact copy of
    the stored row it names, so a citation in the letter resolves after the fact without the
    700 KB day record."""

    schema_version: Literal["temporal_self.day_block.v1"] = "temporal_self.day_block.v1"
    day_id: str
    closed_at: AwareDatetime
    text: str = Field(min_length=1, max_length=DAY_BLOCK_MAX_CHARS)
    records: dict[str, dict[str, Any]] = Field(default_factory=dict)
    labels: dict[str, list[str]] = Field(default_factory=dict)
    lines_dropped: int = 0


class DayCitationCheckV1(_Forbid):
    """What the citation gate found in text a reader of Orion's day wrote back.

    Unknown ``arc:`` ids are always removed and counted. A sentence that names something from the
    shown record (a subject label, a clock time) without citing an arc is not blocked or retried:
    it is counted and the whole check is labelled ``UNVERIFIED`` where Juniper reads it."""

    schema_version: Literal["temporal_self.day_citations.v1"] = "temporal_self.day_citations.v1"
    shown: int = 0
    cited_valid: list[str] = Field(default_factory=list)
    rejected_unknown: list[str] = Field(default_factory=list)
    uncited_day_sentences: int = 0
    uncited_examples: list[str] = Field(default_factory=list)  # first few, clipped
    label: DayCitationLabel = "no_day_claims"
    action: Literal["kept", "stripped"] = "kept"
