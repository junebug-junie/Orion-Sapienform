from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


RitualStyle = Literal["light", "warm", "direct", "brief", "playful", "grounded"]


class SocialPeerStyleHintV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    peer_style_key: str
    platform: str
    room_id: str
    participant_id: str
    participant_name: Optional[str] = None
    style_hints_summary: str = ""
    preferred_directness: float = Field(default=0.5, ge=0.0, le=1.0)
    preferred_depth: float = Field(default=0.5, ge=0.0, le=1.0)
    question_appetite: float = Field(default=0.5, ge=0.0, le=1.0)
    playfulness_tendency: float = Field(default=0.3, ge=0.0, le=1.0)
    formality_tendency: float = Field(default=0.5, ge=0.0, le=1.0)
    summarization_preference: float = Field(default=0.3, ge=0.0, le=1.0)
    evidence_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated_at: str = Field(default_factory=_utcnow_iso)


class SocialRoomRitualSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ritual_key: str
    platform: str
    room_id: str
    greeting_style: RitualStyle = "warm"
    reentry_style: RitualStyle = "grounded"
    thread_revival_style: RitualStyle = "direct"
    pause_handoff_style: RitualStyle = "brief"
    summary_cadence_preference: float = Field(default=0.3, ge=0.0, le=1.0)
    room_tone_summary: str = ""
    culture_summary: str = ""
    evidence_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated_at: str = Field(default_factory=_utcnow_iso)


class SocialStyleAdaptationSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    snapshot_id: str
    platform: Optional[str] = None
    room_id: Optional[str] = None
    participant_id: Optional[str] = None
    core_identity_anchor: str
    peer_adaptation_hint: str = ""
    room_ritual_hint: str = ""
    directness_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    depth_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    question_frequency_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    playfulness_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    summarization_tendency_delta: float = Field(default=0.0, ge=-0.35, le=0.35)
    guardrail: str = "Adapt lightly to the peer and room while remaining Orion."
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utcnow_iso)
