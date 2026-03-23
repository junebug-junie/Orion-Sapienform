from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_thread import SocialAudienceScope


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialClaimKind = Literal[
    "recall",
    "summary",
    "inference",
    "speculation",
    "proposal",
    "clarification_needed",
]
SocialEpistemicConfidenceLevel = Literal["high", "medium", "low"]
SocialEpistemicAmbiguityLevel = Literal["low", "medium", "high"]
SocialEpistemicSourceBasis = Literal[
    "recent_turns",
    "social_memory",
    "active_thread",
    "explicit_peer_request",
    "open_commitment",
    "pending_artifact",
    "repair_context",
    "low_evidence",
]
SocialEpistemicAction = Literal[
    "answer_recall",
    "answer_summary",
    "answer_inference",
    "answer_speculation",
    "ask_clarifying_question",
    "defer_narrowly",
]


class SocialEpistemicSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    epistemic_id: str = Field(default_factory=lambda: f"social-epistemic-signal-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    claim_kind: SocialClaimKind
    confidence_level: SocialEpistemicConfidenceLevel = "low"
    ambiguity_level: SocialEpistemicAmbiguityLevel = "low"
    source_basis: SocialEpistemicSourceBasis = "recent_turns"
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialEpistemicDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision_id: str = Field(default_factory=lambda: f"social-epistemic-decision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    signal_id: Optional[str] = None
    claim_kind: SocialClaimKind
    decision: SocialEpistemicAction = "defer_narrowly"
    confidence_level: SocialEpistemicConfidenceLevel = "low"
    ambiguity_level: SocialEpistemicAmbiguityLevel = "low"
    source_basis: SocialEpistemicSourceBasis = "recent_turns"
    audience_scope: SocialAudienceScope = "none"
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
