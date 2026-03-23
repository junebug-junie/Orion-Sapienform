from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialFreshnessState = Literal["fresh", "aging", "stale", "expired", "refresh_needed"]
SocialDecayLevel = Literal["none", "light", "moderate", "strong"]
SocialRegroundingAction = Literal["keep", "soften", "reopen", "expire", "refresh_needed"]
SocialMemoryArtifactKind = Literal[
    "participant_continuity",
    "room_continuity",
    "claim_consensus",
    "peer_calibration",
    "trust_boundary",
    "peer_style",
    "room_ritual",
    "commitment",
    "deliberation_summary",
    "handoff_closure",
    "unknown",
]


class SocialDecaySignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    signal_id: str = Field(default_factory=lambda: f"social-decay-signal-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    artifact_kind: SocialMemoryArtifactKind = "unknown"
    freshness_state: SocialFreshnessState = "fresh"
    decay_level: SocialDecayLevel = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    last_updated_at: str = Field(default_factory=_utcnow_iso)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialRegroundingDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision_id: str = Field(default_factory=lambda: f"social-regrounding-decision-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    artifact_kind: SocialMemoryArtifactKind = "unknown"
    freshness_state: SocialFreshnessState = "fresh"
    decay_level: SocialDecayLevel = "none"
    decision: SocialRegroundingAction = "keep"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    last_updated_at: str = Field(default_factory=_utcnow_iso)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialMemoryFreshnessV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    freshness_id: str = Field(default_factory=lambda: f"social-memory-freshness-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    artifact_kind: SocialMemoryArtifactKind = "unknown"
    freshness_state: SocialFreshnessState = "fresh"
    decay_level: SocialDecayLevel = "none"
    regrounding_decision: SocialRegroundingAction = "keep"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    last_updated_at: str = Field(default_factory=_utcnow_iso)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
