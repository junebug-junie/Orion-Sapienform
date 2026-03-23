from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialSkillName = Literal[
    "social_summarize_thread",
    "social_safe_recall",
    "social_self_ground",
    "social_followup_question",
    "social_room_reflection",
    "social_exit_or_pause",
    "social_artifact_dialogue",
]


class SocialSkillRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"social-skill-request-{uuid4()}")
    profile: str = "social_room"
    room_id: Optional[str] = None
    thread_id: Optional[str] = None
    prompt: str
    recent_messages: List[str] = Field(default_factory=list)
    allowlist: List[SocialSkillName] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)


class SocialSkillResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    result_id: str = Field(default_factory=lambda: f"social-skill-result-{uuid4()}")
    skill_name: SocialSkillName
    used: bool = True
    summary: str
    snippets: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)


class SocialSkillSelectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    selection_id: str = Field(default_factory=lambda: f"social-skill-selection-{uuid4()}")
    considered_skills: List[SocialSkillName] = Field(default_factory=list)
    selected_skill: Optional[SocialSkillName] = None
    used: bool = False
    selection_reason: str = ""
    suppressed_reason: Optional[str] = None
    request_id: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
