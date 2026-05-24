"""Substrate Atlas visual grammar schemas (distinct from schema_kernel ConceptAtomV1)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GrammarEventKind = Literal[
    "trace_started",
    "trace_ended",
    "atom_emitted",
    "edge_emitted",
    "temporal_hop_emitted",
    "compaction_emitted",
    "projection_emitted",
    "annotation_emitted",
]

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
    provenance: GrammarProvenanceV1


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
