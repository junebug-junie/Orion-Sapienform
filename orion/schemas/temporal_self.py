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

Episodes from the street (2026-10-23, walkway follow-ups spec PR #2763 item 1) added
``street_event`` / arc kind ``street``: a notable outside event (a published motion story that passed
Juniper's spot-check gate, a doorbell ring, a graded handoff that passed the street attention gate)
is one closed process arc, labelled "Outside (<camera>): ..." so it is never read as room company.
Same consumer-first rollout: orion-thought and orion-hub before orion-durable-runs.

``causes_now`` (2026-10-23, on the arc and its summary) follows the same order: every stored frame
carries it (``null`` on non-strain arcs), so an orion-thought or orion-hub still on older code
rejects the whole frame and the stance cue goes dark until they are rebuilt.

Felt state, "where Orion sees it" (2026-11-03, body economy design Organ 5,
docs/superpowers/specs/2026-10-23-organism-body-economy-design.md) added ``body_binding`` / arc kind
``binding`` and ``binding`` (``ArcBindingV1``) on the arc and its summary: the physiology node's own
rows (one per onset, bound-state change, keepalive and release of a body-economy constraint whose
price binds) become one arc per constraint episode. Same consumer-first rollout: every stored arc
and frame now carries ``binding`` (``null`` on other kinds), so orion-thought and orion-hub deploy
before orion-durable-runs.

Self-monitor organ patch 2 (2026-11-03, docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md)
added ``faculty_status_changed`` / arc kind ``faculty``: the self_monitor node's own status-change rows
(already in ``temporal_self_event``) become one arc per stretch a faculty spends broken, unreliable or
miscalibrated ("my walkway grading broke on day X and came back on day Y"). ``episode_began_at`` holds
when the status began. Same consumer-first rollout: orion-thought, orion-hub and orion-dream before
orion-durable-runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, GetJsonSchemaHandler, SerializerFunctionWrapHandler,
                      field_validator, model_serializer, model_validator)

TEMPORAL_SELF_REDUCER_VERSION = "temporal_self.reducer.v1"

TemporalSourceKind = Literal[
    # process boundaries
    "chat_turn",
    "curiosity_run",
    # retired 2026-11-24 (reveries that ask, Patch 1): no producer; kept so history rows still parse
    "reverie_chain",
    "visual_run",
    "dream_cycle",
    "field_dominance_run",
    "attention_loop_raised",
    "attention_loop_verdict",
    # expectations and self-change
    "dream_hypothesis",
    # retired 2026-11-24 with the room-camera expectations: no producer; kept so history rows still parse
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
    # one note Orion stored while a person played (orion-hub retro_watch_look) that a line turn checked
    # against frames of the same moment and found to hold (game company patch 3, spec rev 4 section 8).
    # Opens/refreshes the sitting's play arc; the human_play retro_play row closes it. No salience.
    # ADDITIVE on TemporalNowItemV1 (forbid): deploy the now's readers (orion-thought, orion-hub, orion-mind,
    # orion-prediction) before orion-durable-runs writes the first retro_look event.
    "retro_look",
    # one notable outside (walkway / doorbell) event: story, ring, or gated handoff (PR #2763 item 1)
    "street_event",
    # the physiology node's binding-constraint rows (felt state: onset, change, keepalive, release)
    "body_binding",
    # the self-monitor's held faculty status changes and keepalives (self-monitor organ patch 2)
    "faculty_status_changed",
    # one final office visit (sql-writer vision_office_session): labels the office company arc
    "office_session",
    # Orion's now, patch 1 (docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md
    # A1): experiences only. No arc folds them (arcs._fold_event falls through); they are read back
    # through GET /temporal-self/now. The spec's other three (intention_close, day_letter,
    # personality_choice) are patch 7 and are not named here until a reader exists.
    # one finished reading: a closed hub.world_pulse_read act on the act ledger (grammar_acts)
    "reading_turn",
    # one scored exceedance forecast (orion-prediction prediction_exceedance), with its surprisal
    "prediction_surprise",
    # one Prime thought: a closed prime/infer act on the act ledger, its outcome note as label
    "prime_thought",
    # one finished randomised experiment run (orion-prediction prediction_effect_estimate, estimator
    # family_c_pooled_run): "I held my town sessions to see what they cost: GPU3 +x W, tier-1 +y s"
    # (Family C Experiment 1, docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md)
    "experiment_result",
    # one night's day dream (orion-durable-runs dream_day, the day-dream spec Patch 2): the paragraph of what the
    # day was, and the ids of the memories Orion kept of it. Experience only (no arc folds it).
    # ADDITIVE on TemporalNowItemV1 (forbid): deploy the now's readers (orion-thought, orion-hub, orion-mind,
    # orion-prediction) before orion-durable-runs writes the first dream_day row (~01:00 Denver).
    "dream_day",
    # one reverie (orion-thought, reveries that ask Patch 1): a reverie_question row -- the question it left,
    # or why there was none, or why code refused it. Experience only (no arc folds it).
    # ADDITIVE on TemporalNowItemV1 (forbid): deploy the now's readers (orion-thought, orion-hub, orion-mind,
    # orion-prediction) before orion-durable-runs writes the first reverie_question event.
    "reverie_question",
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
    "street",
    "binding",
    "faculty",
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


# The body-economy price states that mean a constraint binds (orion/physiology/budget.py
# price_constraint; the same set the body report calls BINDING, orion/physiology/report_text.py).
BindingState = Literal["binding", "exogenous_bind", "stuck_at_ceiling"]


class ArcBindingV1(_Forbid):
    """A binding arc's constraint as of its newest row (the physiology node's priced reading)."""

    constraint: str  # heat | energy | lane:<role> | attention:juniper | claude
    bind_state: BindingState
    unit: str | None = None
    level: float | None = None
    line: float | None = None  # the reading's budget: soft line, envelope, capacity, daily cap
    price: float | None = None
    ceiling: float | None = None
    # Orion spending its own budget (orion.physiology.budget.SELF_SPENT_CONSTRAINTS), not the world.
    self_spent: bool = False
    why: str = Field(default="", max_length=LABEL_MAX)
    # True once a release row (slack for 240 s) ended the arc. A closed arc without it ended by the
    # stale timeout (unread): its real end is unknown, so readers say "last read binding at".
    released: bool = False


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
    # A binding arc's constraint at its newest row. None for every other kind.
    binding: ArcBindingV1 | None = None
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
    # Felt state: a binding arc's constraint at its newest row (see TemporalSelfArcV1).
    binding: ArcBindingV1 | None = None


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
    # The newest broadcast tick intervals in seconds, oldest first, capped (arcs.TICK_INTERVALS_CAP).
    # The gap bar is 3 times their median (arcs.tick_gap_sec, #3400). Kept across midnight.
    tick_recent_intervals_sec: list[float] = Field(default_factory=list)
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


# --- Orion's now (integration design Part A, patch 1) -----------------------------------------
# Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, A2-A4. The now is
# the few most salient experiences (``temporal_self_event`` rows) of OTHER organs since a reader last
# looked. Every read is stored (``experience_shown``); a reader's cites are checked against that row
# and only valid ones are stored (``experience_use``). Built by ``orion/temporal_self/now.py`` (pure)
# and served by orion-durable-runs ``GET /temporal-self/now``. No bus channel: HTTP and Postgres only.

NOW_LABEL_MAX = 200
NOW_MAX_ITEMS = 12
NOW_MAX_PER_ORGAN = 2
NOW_MAX_CITES = 3
NOW_RELATED_REFS_CAP = 8
NOW_HELD_REFS_CAP = 4


class TemporalNowItemV1(_Forbid):
    """One experience in the now. ``label`` is code-written from the row (never model prose)."""

    event_id: str
    organ: str
    source_kind: TemporalSourceKind
    label: str = Field(default="", max_length=NOW_LABEL_MAX)
    occurred_at: AwareDatetime
    # Share of the same source kind's events in the 30 days before this one that were strictly
    # smaller (orion.physiology.event_wants.source_salience). None: under 20 past events, or the
    # kind declares no magnitude (config/integration/organs.yaml). Never 0 for "unknown".
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    surprise_nats: float | None = None  # only where a real prediction exists
    related_refs: list[str] = Field(default_factory=list, max_length=NOW_RELATED_REFS_CAP)
    # #3783: the refs the reader holds (``GET /temporal-self/now?held=...``; Prime: a concern's event_ref, ``arc:<id>``
    # or ``prime_concern:<id>``) that this row's own ids match exactly. Such a row ranks first. Left out of the body
    # while empty, so a read without ``held`` (reverie, curiosity, the thought service, dream day) is byte-for-byte
    # the old body and an image built before this field still validates it.
    held_refs: list[str] = Field(default_factory=list, max_length=NOW_HELD_REFS_CAP)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_empty_held(self, handler: SerializerFunctionWrapHandler) -> Any:
        out = handler(self)
        if isinstance(out, dict) and not out.get("held_refs"):
            out.pop("held_refs", None)
        return out


class TemporalNowV1(_Forbid):
    """What one reader was shown. ``shown_id`` is the ``experience_shown`` row a cite must match."""

    schema_version: Literal["temporal_self.now.v1"] = "temporal_self.now.v1"
    shown_id: str
    reader_organ: str
    reader_run_ref: str | None = None
    since: AwareDatetime
    as_of: AwareDatetime
    items: list[TemporalNowItemV1] = Field(default_factory=list, max_length=NOW_MAX_ITEMS)
    candidates: int = 0  # other organs' experiences recorded since ``since``
    dropped_by_organ_cap: int = 0
    dropped_by_total_cap: int = 0
    unregistered_kinds: list[str] = Field(default_factory=list)  # kinds no organ claims (never shown)


class ExperienceCitesV1(_Forbid):
    """A reader's cites at its run's close (``POST /temporal-self/now/use``). At most 3 count."""

    shown_id: str = Field(min_length=1)
    citing_ref: str = Field(min_length=1, max_length=300)  # the citing run's own output ref
    cites: list[str] = Field(default_factory=list, max_length=32)  # event ids, bare or "experience:<id>"


class ExperienceUseRowV1(_Forbid):
    use_id: str
    event_id: str
    citing_organ: str
    citing_ref: str
    shown_id: str
    cited_at: AwareDatetime


class ExperienceUseResultV1(_Forbid):
    """``record_use``'s verdict. Invalid cites are counted and dropped, never stored."""

    schema_version: Literal["temporal_self.experience_use.v1"] = "temporal_self.experience_use.v1"
    shown_id: str
    citing_organ: str
    uses: list[ExperienceUseRowV1] = Field(default_factory=list, max_length=NOW_MAX_CITES)
    rejected_not_shown: list[str] = Field(default_factory=list)
    rejected_over_cap: list[str] = Field(default_factory=list)


class OrganUseV1(_Forbid):
    organ: str
    experiences: int = 0
    used_by_other: int = 0  # distinct experiences at least one different organ cited, validly
    cites_made: int = 0  # valid cites this organ made of other organs' experiences in the window
    rate: float | None = None  # None when the organ recorded no experience


class ExperienceUseRateV1(_Forbid):
    """The integration metric: the share of experiences a different organ used within 24 h."""

    schema_version: Literal["temporal_self.experience_use_rate.v1"] = "temporal_self.experience_use_rate.v1"
    window_start: AwareDatetime
    window_end: AwareDatetime
    experiences: int = 0
    used_by_other: int = 0
    rate: float | None = None
    closed_organs: list[str] = Field(default_factory=list)
    registered_organs: int = 0
    organs: list[OrganUseV1] = Field(default_factory=list)
    unregistered_kinds: dict[str, int] = Field(default_factory=dict)


# --- Prime hunger inputs (integration design Part B, patch 5) ----------------------------------
# orion-durable-runs GET /temporal-self/hunger?organ=<id>: the two inputs of Prime's needs that live in
# Postgres (orion/prime/hunger.py computes the drives; this is only the read). Read-only: no
# experience_shown row is written, so polling it is never a "shown" now. HTTP only, no bus channel.

HUNGER_SURPRISES_CAP = 10
HUNGER_JUNIPER_TIMES_CAP = 2000
HUNGER_PROMPT_CAP = 240


class HungerSurpriseV1(_Forbid):
    """One experience of another organ whose ``surprise_nats`` is strictly above its own source kind's
    30-day 90th percentile, and that no organ has cited (``experience_use``)."""

    event_id: str
    organ: str
    source_kind: TemporalSourceKind
    label: str = Field(default="", max_length=NOW_LABEL_MAX)
    occurred_at: AwareDatetime
    surprise_nats: float = Field(ge=0.0)
    source_p90: float = Field(ge=0.0)  # the kind's 90th percentile over the 30 days before as_of
    source_n: int = Field(ge=0)  # how many of the kind's events carried surprise_nats in those 30 days


class HungerJuniperV1(_Forbid):
    """Juniper's own messages to Orion in Hub (``chat_history_log`` source ``hub_orion``, prompt not blank),
    30 days, oldest first; and the last one, quoted, with its Temporal Self experience id when ingested."""

    message_times: list[AwareDatetime] = Field(default_factory=list, max_length=HUNGER_JUNIPER_TIMES_CAP)
    last_chat_id: str | None = None
    last_prompt: str = Field(default="", max_length=HUNGER_PROMPT_CAP)
    # temporal_self_event.event_id of the chat_turn row for the last message; None until the chronicle ingests it.
    last_event_id: str | None = None


class TemporalHungerInputsV1(_Forbid):
    schema_version: Literal["temporal_self.hunger_inputs.v1"] = "temporal_self.hunger_inputs.v1"
    organ: str
    as_of: AwareDatetime
    surprises: list[HungerSurpriseV1] = Field(default_factory=list, max_length=HUNGER_SURPRISES_CAP)
    juniper: HungerJuniperV1 = Field(default_factory=HungerJuniperV1)
