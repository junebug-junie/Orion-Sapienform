from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialAudienceScope = Literal["peer", "room", "thread", "summary", "none"]
SocialThreadRoutingKind = Literal["reply_to_peer", "reply_to_room", "wait", "summarize_room", "revive_thread"]
SocialHandoffKind = Literal["to_orion", "yield_to_peer", "room_summary", "thread_wrap", "none"]
SocialThreadAmbiguityLevel = Literal["low", "medium", "high"]


class SocialThreadStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    thread_key: str
    platform: str
    room_id: str
    thread_id: Optional[str] = None
    active_participants: List[str] = Field(default_factory=list)
    audience_scope: SocialAudienceScope = "thread"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    last_speaker: str
    last_addressed_participant_id: Optional[str] = None
    last_addressed_participant_name: Optional[str] = None
    open_question: bool = False
    handoff_flag: bool = False
    orion_involved: bool = False
    thread_summary: str = ""
    last_activity_at: str = Field(default_factory=_utcnow_iso)
    expires_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialHandoffSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    handoff_id: str = Field(default_factory=lambda: f"social-handoff-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    handoff_kind: SocialHandoffKind = "none"
    audience_scope: SocialAudienceScope = "none"
    from_participant_id: Optional[str] = None
    from_participant_name: Optional[str] = None
    to_participant_id: Optional[str] = None
    to_participant_name: Optional[str] = None
    detected: bool = False
    rationale: str = ""
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialThreadRoutingDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    routing_id: str = Field(default_factory=lambda: f"social-thread-routing-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    audience_scope: SocialAudienceScope = "none"
    routing_decision: SocialThreadRoutingKind
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    last_speaker: Optional[str] = None
    last_addressed_participant_id: Optional[str] = None
    open_question: bool = False
    handoff_flag: bool = False
    thread_summary: str = ""
    primary_thread_key: Optional[str] = None
    primary_thread_summary: str = ""
    candidate_thread_summaries: List[str] = Field(default_factory=list)
    ambiguity_level: SocialThreadAmbiguityLevel = "low"
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
