from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_thread import SocialAudienceScope


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialFloorDecisionKind = Literal[
    "yield_to_peer",
    "invite_peer",
    "invite_room",
    "close_thread",
    "leave_open",
    "no_handoff",
]
SocialClosureKind = Literal["resolved", "left_open", "none"]


class SocialTurnHandoffV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    handoff_id: str = Field(default_factory=lambda: f"social-turn-handoff-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    decision_kind: SocialFloorDecisionKind = "no_handoff"
    handoff_text: str = ""
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialClosureSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    closure_signal_id: str = Field(default_factory=lambda: f"social-closure-signal-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    closure_kind: SocialClosureKind = "none"
    resolved: bool = False
    closure_text: str = ""
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialFloorDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    floor_decision_id: str = Field(default_factory=lambda: f"social-floor-decision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    decision_kind: SocialFloorDecisionKind = "no_handoff"
    handoff_id: Optional[str] = None
    closure_signal_id: Optional[str] = None
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
