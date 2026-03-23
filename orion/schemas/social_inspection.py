from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialInspectionSectionKind = Literal[
    "context_window",
    "claims",
    "commitments",
    "routing",
    "repair",
    "deliberation",
    "floor",
    "calibration",
    "freshness",
    "resumptive",
    "epistemic",
    "artifact_dialogue",
    "gif",
    "safety",
]

SocialInspectionDecisionState = Literal["selected", "softened", "excluded", "active", "omitted"]


class SocialInspectionDecisionTraceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trace_id: str = Field(default_factory=lambda: f"social-inspection-trace-{uuid4()}")
    trace_kind: str
    decision_state: SocialInspectionDecisionState = "active"
    summary: str
    why_it_mattered: str = ""
    source_ref: str = ""
    freshness_hint: Optional[str] = None
    confidence_hint: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialInspectionSectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    section_id: str = Field(default_factory=lambda: f"social-inspection-section-{uuid4()}")
    section_kind: SocialInspectionSectionKind
    included_artifact_summaries: List[str] = Field(default_factory=list)
    selected_state: List[str] = Field(default_factory=list)
    softened_state: List[str] = Field(default_factory=list)
    excluded_state: List[str] = Field(default_factory=list)
    freshness_hints: List[str] = Field(default_factory=list)
    confidence_hints: List[str] = Field(default_factory=list)
    why_this_mattered: str = ""
    decision_traces: List[SocialInspectionDecisionTraceV1] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialInspectionSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    snapshot_id: str = Field(default_factory=lambda: f"social-inspection-snapshot-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    participant_id: Optional[str] = None
    summary: str
    sections: List[SocialInspectionSectionV1] = Field(default_factory=list)
    decision_traces: List[SocialInspectionDecisionTraceV1] = Field(default_factory=list)
    built_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
