"""Substrate Atlas visual grammar schemas (distinct from schema_kernel ConceptAtomV1)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GrammarEventKind = Literal[
    "trace_started",
    "trace_ended",
    "atom_emitted",
    "edge_emitted",
    "temporal_hop_emitted",
    "compaction_emitted",
    "projection_emitted",
    "annotation_emitted",
    # Doing on one track (rung 2, docs/superpowers/specs/2026-10-14-orion-
    # thinking-architecture-design.md section 3.2): an efference copy of an act.
    # Opened before the act runs (with what it expects), closed by act_id when
    # its outcome is known. Persisted to grammar_acts by orion-sql-writer.
    "act_opened",
    "act_closed",
]

# Closed, versioned verb vocabulary (design section 3.2). The same at every
# level. Producers as of rung 2: make (orion-thought paintings), say (hub chat
# reply, outreach), ask (outreach), look (reading), play (hub retro leisure
# session, 2026-10-22: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md
# section E). The others are reserved by the design and have no producer yet; a
# new producer must ship with a test.
ActVerb = Literal[
    "sense",
    "look",
    "recall",
    "infer",
    "predict",
    "say",
    "ask",
    "wait",
    "make",
    "maintain",
    "regulate",
    "play",
]

ActLevel = Literal["metabolic", "organ", "faculty", "deliberate"]

# What an act can be checked against. Each is a BINARY "did this happen by the
# horizon" event, so a horizon that passes with no evidence is a real 0, not a
# guess (that is what lets the writer close expired acts honestly).
#   juniper_replied   -- Juniper sent Orion a chat turn after the act, before
#                        the horizon, by when her turn STARTED. Resolved by
#                        orion-sql-writer from the hub ingress hop
#                        (cockpit_turn_sighting; chat_history_log created_at
#                        fallback); producers never self-report it.
#   artifact_produced -- the act produced its artifact (an image stored, a
#                        reading with real content) before the horizon.
#                        Reported by the producer on close.
ActSignal = Literal["juniper_replied", "artifact_produced"]

# Who supplied an act's outcome.
ActOutcomeSource = Literal["producer", "resolver", "expired"]

AtomType = Literal[
    "raw_span",
    "observation",
    "signal",
    "entity",
    "claim",
    "affective_cue",
    "salience_marker",
    "uncertainty_marker",
    "memory_claim",
    "stance_influence",
    "reasoning_step",
    "action_candidate",
    "spoken_output",
    "scene_state",
    "compaction",
    "projection",
]

RelationType = Literal[
    "derived_from",
    "supports",
    "contradicts",
    "temporal_successor",
    "compacted_into",
    "expands_to",
    "influenced",
    "recalled_by",
    "projected_from",
    "caused_candidate",
    "rendered_as",
    "same_as",
    "near",
    "contains",
    "references",
]

TemporalHopType = Literal[
    "same_turn",
    "prior_turn",
    "prior_session",
    "memory_recall",
    "dream_reentry",
    "projection",
    "counterfactual",
    "compaction_window",
    "future_candidate",
]


class GrammarProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_service: str
    source_component: str | None = None
    source_event_id: str | None = None
    source_trace_id: str | None = None
    source_payload_ref: str | None = None
    code_version: str | None = None
    model_id: str | None = None
    prompt_id: str | None = None


class TimeRangeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime | None = None


class GrammarAtomV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_atom.v1"] = "grammar_atom.v1"
    atom_id: str
    trace_id: str
    atom_type: AtomType
    semantic_role: str
    layer: str
    dimensions: list[str] = Field(default_factory=list)
    summary: str
    text_value: str | None = None
    confidence: float | None = None
    salience: float | None = None
    uncertainty: float | None = None
    time_range: TimeRangeV1 | None = None
    source_event_id: str | None = None
    payload_ref: str | None = None
    renderer_hint: str | None = None


class GrammarEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_edge.v1"] = "grammar_edge.v1"
    edge_id: str
    trace_id: str
    from_atom_id: str
    to_atom_id: str
    relation_type: RelationType
    confidence: float | None = None
    salience: float | None = None
    layer_from: str | None = None
    layer_to: str | None = None
    temporal_relation: str | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)


class TemporalHopV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_temporal_hop.v1"] = "grammar_temporal_hop.v1"
    hop_id: str
    trace_id: str
    from_atom_id: str
    to_atom_id: str | None = None
    hop_type: TemporalHopType
    direction: Literal["backward", "forward", "lateral"]
    reason: str
    confidence: float | None = None
    turn_distance: int | None = None
    session_distance: int | None = None
    target_time_range: TimeRangeV1 | None = None


class GrammarCompactionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_compaction.v1"] = "grammar_compaction.v1"
    compaction_id: str
    trace_id: str
    source_atom_ids: list[str]
    output_atom_id: str
    compaction_type: str
    method: str
    summary: str
    preserves: list[str] = Field(default_factory=list)
    drops: list[str] = Field(default_factory=list)
    confidence: float | None = None


class GrammarProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_projection.v1"] = "grammar_projection.v1"
    projection_id: str
    trace_id: str
    source_atom_ids: list[str]
    projection_type: str
    summary: str
    confidence: float | None = None
    expires_at: datetime | None = None
    projected_atom_id: str | None = None


class ActExpectV1(BaseModel):
    """What an act expects, written BEFORE it runs (the efference copy).

    A producer declares only what it genuinely knows: which signal the act is
    about, which way it should move, and by when. The magnitude
    (``predicted``) is NOT hand-written: orion-sql-writer stamps the current
    EffectPosterior mean for (verb, actor, signal) when it persists the open,
    exactly as ExpectedEffectV1 does for dispatches. ``basis`` says, in words,
    where the horizon and direction came from, and must not be empty.
    """

    model_config = ConfigDict(extra="forbid")

    signal_id: ActSignal
    direction: Literal["increase", "decrease", "no_change"] = "increase"
    horizon_sec: float = Field(gt=0.0)
    basis: str = Field(min_length=1)
    # Writer-stamped. A producer leaves these unset.
    predicted: float | None = None
    predicted_n: int | None = Field(default=None, ge=0)
    cold_start: bool | None = None


class ActCostV1(BaseModel):
    """Currency the act spent. None means not measured, never zero."""

    model_config = ConfigDict(extra="forbid")

    seconds: float | None = Field(default=None, ge=0.0)
    gpu_seconds: float | None = Field(default=None, ge=0.0)
    tokens: int | None = Field(default=None, ge=0)
    watts: float | None = Field(default=None, ge=0.0)


class ActCostV2(BaseModel):
    """ActCostV1 plus the body economy's per-constraint currencies (body economy Patch A2,
    docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Organ 3 "ActCostV2").

    Additive and optional: every V1 field, same meaning. None means not measured, never zero.

    * ``seconds``: wall time from act_opened to act_closed. The act ledger fills it on a PRODUCER
      close when the producer left it unset and the open time is known. A resolver or expired
      close is stamped at the horizon (or at Juniper's reply), so its wall time is not what the
      act spent: left None.
    * ``gpu_slot_s_by_lane``: GPU pool slot-seconds held, per lane (pool role). Filled on close
      from ``gpu_pool_events`` leases whose ``turn_correlation_id`` is the act's correlation_id.
    * ``wh_by_host``: watt-hours drawn, per host. Not measured yet.
    * ``juniper_msgs``: messages that reached Juniper. Not measured yet.
    * ``claude_limit_events``: Claude limit events the act triggered. Not measured yet.

    CONSUMER-FIRST: ``ActV1.cost`` accepts V1 or V2, but V1 and ActV1 are extra="forbid", so a
    reader running older code rejects a V2 payload outright. Producers keep sending V1 (or no cost)
    until every reader of ``orion:grammar:event`` (sql-writer, and substrate-runtime reading
    ``grammar_events``) runs code that knows V2. The writer-side fill below never crosses the bus:
    it lands only in ``grammar_acts.cost_json``.
    """

    model_config = ConfigDict(extra="forbid")

    seconds: float | None = Field(default=None, ge=0.0)
    gpu_seconds: float | None = Field(default=None, ge=0.0)
    tokens: int | None = Field(default=None, ge=0)
    watts: float | None = Field(default=None, ge=0.0)
    gpu_slot_s_by_lane: dict[str, float] | None = None
    wh_by_host: dict[str, float] | None = None
    juniper_msgs: int | None = Field(default=None, ge=0)
    claude_limit_events: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _non_negative_maps(self) -> "ActCostV2":
        for name in ("gpu_slot_s_by_lane", "wh_by_host"):
            for key, value in (getattr(self, name) or {}).items():
                if value < 0:
                    raise ValueError(f"{name}[{key!r}] must be >= 0, got {value!r}")
        return self

    @classmethod
    def from_any(cls, cost: "ActCostV1 | ActCostV2 | None") -> "ActCostV2 | None":
        if cost is None:
            return None
        if isinstance(cost, cls):
            return cost
        return cls(**cost.model_dump())


class ActOutcomeV1(BaseModel):
    """What actually happened, on the expectation's own scale."""

    model_config = ConfigDict(extra="forbid")

    value: float
    observed_at: datetime
    source: ActOutcomeSource = "producer"
    note: str = ""

    @model_validator(mode="after")
    def _binary_value(self) -> "ActOutcomeV1":
        # Every ActSignal is a binary event (see ActSignal above).
        if self.value not in (0.0, 1.0):
            raise ValueError(f"act outcome value must be 0 or 1, got {self.value!r}")
        return self


class ActV1(BaseModel):
    """One act, as a TOTE unit (Test -> Operate -> Test -> Exit).

    act_opened carries identity, verb, test_before and expect.
    act_closed carries the same act_id plus outcome (and cost if measured).
    ``error`` (outcome - predicted) is computed by the writer at close; a
    producer never fills it, because it never sees ``predicted``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_act.v1"] = "grammar_act.v1"
    act_id: str = Field(min_length=1)
    parent_act_id: str | None = None
    level: ActLevel
    actor: str = Field(min_length=1)
    verb: ActVerb
    object_ref: str | None = None
    test_before: str | None = None
    expect: ActExpectV1 | None = None
    # V1 or V2 (consumer-first: see ActCostV2). A dict with only V1 keys parses as V1.
    cost: ActCostV1 | ActCostV2 | None = None
    outcome: ActOutcomeV1 | None = None
    error: float | None = None
    goal_ref: str | None = None
    workspace_ref: str | None = None

    @model_validator(mode="after")
    def _expect_required_above_metabolic(self) -> "ActV1":
        if self.level != "metabolic" and self.expect is None:
            raise ValueError(f"{self.level} act {self.act_id!r} must carry an expect")
        return self


class GrammarEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_event.v1"] = "grammar_event.v1"
    event_id: str
    event_kind: GrammarEventKind
    trace_id: str
    parent_event_id: str | None = None
    root_event_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    correlation_id: str | None = None
    emitted_at: datetime
    observed_at: datetime | None = None
    layer: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    atom: GrammarAtomV1 | None = None
    edge: GrammarEdgeV1 | None = None
    temporal_hop: TemporalHopV1 | None = None
    compaction: GrammarCompactionV1 | None = None
    projection: GrammarProjectionV1 | None = None
    act: ActV1 | None = None
    provenance: GrammarProvenanceV1

    @model_validator(mode="after")
    def _act_events_carry_an_act(self) -> "GrammarEventV1":
        if self.event_kind in ("act_opened", "act_closed"):
            if self.act is None:
                raise ValueError(f"{self.event_kind} requires act")
            if self.event_kind == "act_closed" and self.act.outcome is None:
                raise ValueError("act_closed requires act.outcome")
        return self


class GrammarTraceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["grammar_trace.v1"] = "grammar_trace.v1"
    trace_id: str
    trace_type: str
    session_id: str | None = None
    turn_id: str | None = None
    root_event_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: Literal["open", "closed"] = "open"
    summary: str | None = None
