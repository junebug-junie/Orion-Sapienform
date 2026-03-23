from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialContextCandidateKind = Literal[
    "thread",
    "claim",
    "consensus",
    "divergence",
    "calibration",
    "commitment",
    "repair",
    "ritual",
    "style",
    "freshness_hint",
    "handoff",
    "deliberation",
    "peer_continuity",
    "room_continuity",
    "episode_snapshot",
    "reentry_anchor",
]
SocialContextPriorityBand = Literal["critical", "high", "medium", "low", "background"]
SocialContextFreshnessBand = Literal["fresh", "aging", "stale", "refresh_needed", "expired"]
SocialContextInclusionDecision = Literal["include", "soften", "exclude"]


class SocialContextCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: str = Field(default_factory=lambda: f"social-context-candidate-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    participant_id: Optional[str] = None
    candidate_kind: SocialContextCandidateKind
    reference_key: str = ""
    summary: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_band: SocialContextPriorityBand = "background"
    freshness_band: SocialContextFreshnessBand = "fresh"
    inclusion_decision: SocialContextInclusionDecision = "include"
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    max_window_budget: int = Field(default=6, ge=1)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialContextSelectionDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision_id: str = Field(default_factory=lambda: f"social-context-selection-decision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    selected_candidate_ids: List[str] = Field(default_factory=list)
    total_candidates_considered: int = Field(default=0, ge=0)
    included_count: int = Field(default=0, ge=0)
    softened_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0)
    budget_max: int = Field(default=6, ge=1)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    assembled_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialContextWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    window_id: str = Field(default_factory=lambda: f"social-context-window-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    participant_id: Optional[str] = None
    selected_candidates: List[SocialContextCandidateV1] = Field(default_factory=list)
    budget_max: int = Field(default=6, ge=1)
    total_candidates_considered: int = Field(default=0, ge=0)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    assembled_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialEpisodeSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    snapshot_id: str = Field(default_factory=lambda: f"social-episode-snapshot-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    participant_id: Optional[str] = None
    summary: str
    resumptive_hint: str = ""
    focus_topics: List[str] = Field(default_factory=list)
    last_active_at: str = Field(default_factory=_utcnow_iso)
    freshness_band: SocialContextFreshnessBand = "fresh"
    superseded_by_live_state: bool = False
    rationale: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialReentryAnchorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    anchor_id: str = Field(default_factory=lambda: f"social-reentry-anchor-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    participant_id: Optional[str] = None
    source_snapshot_id: Optional[str] = None
    anchor_text: str
    freshness_band: SocialContextFreshnessBand = "fresh"
    reentry_style: str = "grounded"
    rationale: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
