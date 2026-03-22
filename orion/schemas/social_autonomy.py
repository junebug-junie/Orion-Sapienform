from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_epistemic import SocialEpistemicDecisionV1, SocialEpistemicSignalV1
from orion.schemas.social_repair import SocialRepairDecisionV1, SocialRepairSignalV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadRoutingDecisionV1

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialAutonomyMode = Literal["addressed_only", "responsive", "light_initiative"]
SocialTurnDecision = Literal["reply", "wait", "ask_follow_up", "initiate_lightly", "skip"]


class SocialOpenThreadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    topic_key: str
    platform: str
    room_id: str
    summary: str
    last_speaker: str
    open_question: bool = False
    orion_involved: bool = False
    last_activity_at: str = Field(default_factory=_utcnow_iso)
    expires_at: str = Field(default_factory=_utcnow_iso)
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_count: int = 0


class SocialTurnPolicyDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision_id: str = Field(default_factory=lambda: f"social-turn-policy-{uuid4()}")
    mode: SocialAutonomyMode = "responsive"
    platform: str
    room_id: str
    thread_id: Optional[str] = None
    participant_id: Optional[str] = None
    decision: SocialTurnDecision
    should_speak: bool = False
    reasons: List[str] = Field(default_factory=list)
    addressed: bool = False
    cooldown_active: bool = False
    consecutive_limit_hit: bool = False
    quiet_room: bool = False
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    open_thread_key: Optional[str] = None
    thread_routing: Optional[SocialThreadRoutingDecisionV1] = None
    handoff_signal: Optional[SocialHandoffSignalV1] = None
    repair_signal: Optional[SocialRepairSignalV1] = None
    repair_decision: Optional[SocialRepairDecisionV1] = None
    epistemic_signal: Optional[SocialEpistemicSignalV1] = None
    epistemic_decision: Optional[SocialEpistemicDecisionV1] = None
    created_at: str = Field(default_factory=_utcnow_iso)
