# orion/schemas/metacog_entry.py
"""Metacog self-observation entry schema (real-artifact model).

This is a NEW, INDEPENDENT schema. It is not a subclass or remap of
CollapseMirrorEntryV2 (orion/schemas/collapse_mirror.py) -- collapse_mirror is
now strict-lane-only (Juniper's manually-authored journal entries via the Hub
form). Orion's machine-generated self-observation entries live here instead,
sourced from real, live-computed turn artifacts rather than an LLM self-report.

Deliberately DROPPED, no replacement: the old `numeric_sisters` field
(valence/arousal/clarity/overload/risk_score). It was proven contaminated --
primed by `phi_hint` (phi's own live bands) before the LLM "self-reported" the
same numbers back. No sisters. See
docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metacog_event_id() -> str:
    return f"metacog_{uuid4().hex}"


class MetacogWhatChanged(BaseModel):
    """Structured diff placeholder -- kept minimal by design, not over-built."""

    model_config = ConfigDict(extra="ignore")

    summary: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)


class MetacogRepairEvidence(BaseModel):
    """Lighter summary shape mirroring orion/schemas/repair_evidence.py::RepairEvidenceV1's
    evidence_kind/score/confidence. Deliberately NOT importing that schema directly --
    this is a compact evidence-list entry for the metacog entry, not the full
    repair-pressure appraisal record."""

    model_config = ConfigDict(extra="ignore")

    evidence_kind: str
    score: float
    confidence: float


class MetacogRepairPressure(BaseModel):
    """Real, live-computed repair_pressure_v2 evidence (Task 4's hub wiring),
    not a self-rating."""

    model_config = ConfigDict(extra="ignore")

    level: float
    level_label: str
    confidence: float
    evidence: list[MetacogRepairEvidence] = Field(default_factory=list)
    behavior_applied: Optional[str] = None


class MetacogRealState(BaseModel):
    """Only real, live-computed artifacts. No self-rating of any kind."""

    model_config = ConfigDict(extra="ignore")

    biometrics: Optional[dict[str, Any]] = None
    turn_effect: Optional[dict[str, Any]] = None
    turn_effect_evidence: Optional[dict[str, Any]] = None
    substrate_eventfulness_score: Optional[float] = None
    substrate_eventfulness_reasons: Optional[list[str]] = None
    # Only populated when the trigger itself is about LLM surface
    # instability (trigger_kind=llm_surface_instability, copied from its
    # upstream). The metacog writer's own logprob probe that used to fill
    # this on ~every row was removed 2026-09-24: it measured the writer, not
    # the event.
    llm_uncertainty: Optional[dict[str, Any]] = None
    reasoning_excerpt: Optional[str] = None
    repair_pressure: Optional[MetacogRepairPressure] = None


class MetacogCausalDensity(BaseModel):
    """score = the triggering event's own normalized magnitude (0..1), from
    orion.metacog.evidence_map (2026-09-24). Previously a blend of global
    state fields that only ever read 0 or 0.25."""

    model_config = ConfigDict(extra="ignore")

    label: str = "ambient"
    score: float = 0.0
    rationale: Optional[str] = None


class MetacogProvenance(BaseModel):
    """docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md
    lines ~107-116: 'came from X, produces Y, impacts [Z,A,...]'."""

    model_config = ConfigDict(extra="ignore")

    source: str
    produces: str
    impacts: list[str] = Field(default_factory=list)
    # The metacog pipeline's own step log ("exec -> X", "ok <- X", ...). Moved
    # here out of what_changed.evidence on 2026-09-24: it describes how the row
    # was produced, not what the triggering event was. Additive + optional;
    # extra="ignore" means an un-rebuilt sql-writer drops it silently rather
    # than failing, so rebuild orion-sql-writer alongside cortex-exec.
    pipeline_steps: list[str] = Field(default_factory=list)


class MetacogEntryV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=_metacog_event_id)
    timestamp: str = Field(default_factory=_utc_now_iso)
    environment: Optional[str] = None

    # From the upstream MetacogTriggerV1 that caused this entry.
    trigger_kind: str
    trigger_reason: str

    # Authored/LLM-narrative fields -- the "ritual half," explicitly meant to
    # stay non-deterministic.
    summary: str
    mantra: str

    what_changed: MetacogWhatChanged = Field(default_factory=MetacogWhatChanged)
    state: MetacogRealState = Field(default_factory=MetacogRealState)

    # Discrete severity read off the TRIGGER's own upstream magnitude
    # (2026-09-24 metric-definition change; previously the writing LLM's
    # logprob margin + the pipeline's failed-step count). Same band as
    # causal_density.score: nominal <0.3, degraded <0.6, critical >=0.6.
    # See orion.metacog.evidence_map.
    severity: Literal["nominal", "degraded", "critical"] = "nominal"

    # Topology, not severity: the services / channels / artifacts named in
    # the trigger's own upstream (e.g. "cortex-exec",
    # "orion:state:request", "channel:failure_pressure"). See
    # orion.metacog.evidence_map (2026-09-24; previously which global `state`
    # fields happened to be populated).
    touches: list[str] = Field(default_factory=list)

    causal_density: MetacogCausalDensity = Field(default_factory=MetacogCausalDensity)
    is_causally_dense: bool = False

    # Constrained, not free string. The old collapse_mirror table's
    # snapshot_kind column has 38 distinct garbage free-text values live in
    # production from years of being an unconstrained string -- learn from
    # that here.
    snapshot_kind: Literal["baseline", "confirmed_dense"] = "baseline"

    provenance: MetacogProvenance

    tags: list[str] = Field(default_factory=list)

    source_service: Optional[str] = None
    source_node: Optional[str] = None

    epistemic_status: str = "observed"
    visibility: str = "internal"
    redaction_level: str = "low"
