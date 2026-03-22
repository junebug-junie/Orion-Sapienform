from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialRepairType = Literal[
    "peer_correction",
    "thread_mismatch",
    "audience_mismatch",
    "commitment_contradiction",
    "scope_correction",
    "redirect",
    "clarification_after_misalignment",
]
SocialRepairTrigger = Literal[
    "peer_message",
    "thread_routing",
    "active_commitment",
    "scope_boundary",
    "handoff_redirect",
    "low_confidence",
]
SocialRepairAction = Literal["repair", "clarify", "yield", "ignore", "reset_thread"]


class SocialRepairSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    repair_id: str = Field(default_factory=lambda: f"social-repair-signal-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    repair_type: SocialRepairType
    trigger: SocialRepairTrigger = "peer_message"
    source_participant_id: Optional[str] = None
    source_participant_name: Optional[str] = None
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detected: bool = False
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialRepairDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision_id: str = Field(default_factory=lambda: f"social-repair-decision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    repair_type: Optional[SocialRepairType] = None
    trigger: Optional[SocialRepairTrigger] = None
    signal_id: Optional[str] = None
    decision: SocialRepairAction = "ignore"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
