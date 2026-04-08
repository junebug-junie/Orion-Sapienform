from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_thread import SocialAudienceScope


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialCommitmentType = Literal[
    "answer_pending_question",
    "return_to_thread",
    "summarize_room",
    "respect_memory_scope",
    "yield_then_reenter",
]
SocialCommitmentState = Literal["open", "fulfilled", "superseded", "dropped", "expired"]
SocialCommitmentDueState = Literal["fresh", "due_soon", "stale"]


class SocialCommitmentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    commitment_id: str = Field(default_factory=lambda: f"social-commitment-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    commitment_type: SocialCommitmentType
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    summary: str
    state: SocialCommitmentState = "open"
    source_turn_id: Optional[str] = None
    source_correlation_id: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
    expires_at: str = Field(default_factory=_utcnow_iso)
    due_state: SocialCommitmentDueState = "fresh"
    resolution_reason: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialCommitmentResolutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    resolution_id: str = Field(default_factory=lambda: f"social-commitment-resolution-{uuid4()}")
    commitment_id: str
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    commitment_type: SocialCommitmentType
    state: SocialCommitmentState
    summary: str
    resolution_reason: str = ""
    resolved_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
