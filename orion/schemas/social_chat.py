from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialRoomProfile = Literal["social_room"]
RedactionLevel = Literal["low", "medium", "high"]


class SocialConceptEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ref_id: str
    source_kind: str
    summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SocialGroundingStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    profile: SocialRoomProfile = "social_room"
    identity_label: str = "Oríon"
    relationship_frame: str = "peer"
    self_model_hint: str = "distributed social presence"
    continuity_anchor: Optional[str] = None
    stance: str = "warm, direct, grounded"


class SocialRedactionScoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    prompt_score: float = Field(default=0.0, ge=0.0, le=1.0)
    response_score: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_safe: bool = True
    redaction_level: RedactionLevel = "low"
    reasons: List[str] = Field(default_factory=list)


class SocialRoomTurnV1(BaseModel):
    """Append-only social_room turn persistence payload."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    turn_id: str = Field(default_factory=lambda: f"social-turn-{uuid4()}")
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    source: str = "hub"
    profile: SocialRoomProfile = "social_room"
    prompt: str
    response: str
    text: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    recall_profile: Optional[str] = None
    trace_verb: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    concept_evidence: List[SocialConceptEvidenceV1] = Field(default_factory=list)
    grounding_state: SocialGroundingStateV1 = Field(default_factory=SocialGroundingStateV1)
    redaction: SocialRedactionScoreV1 = Field(default_factory=SocialRedactionScoreV1)
    client_meta: Dict[str, object] = Field(default_factory=dict)


class SocialRoomTurnStoredV1(SocialRoomTurnV1):
    """Post-commit stored event emitted by sql-writer."""

    stored_at: str = Field(default_factory=_utcnow_iso)

