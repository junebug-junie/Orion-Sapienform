from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialDeliberationDecisionKind = Literal["bridge_summary", "ask_clarifying_question", "normal_peer_reply", "normal_room_reply", "stay_narrow", "wait"]
SocialDeliberationAmbiguity = Literal["low", "medium", "high"]


class SocialBridgeSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bridge_summary_id: str = Field(default_factory=lambda: f"social-bridge-summary-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    active_claim_ids: List[str] = Field(default_factory=list)
    active_claim_keys: List[str] = Field(default_factory=list)
    trigger: Literal["partial_agreement", "contested_shared_core", "crosstalk", "explicit_landing_request"] = "partial_agreement"
    shared_core: str = ""
    disagreement_edge: str = ""
    attributed_views: List[str] = Field(default_factory=list)
    agreement_points: List[str] = Field(default_factory=list)
    disagreement_points: List[str] = Field(default_factory=list)
    attributed_participants: Dict[str, str] = Field(default_factory=dict)
    summary_text: str = ""
    proposed_bridge_framing: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_level: SocialDeliberationAmbiguity = "medium"
    preserve_disagreement: bool = True
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialClarifyingQuestionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    clarifying_question_id: str = Field(default_factory=lambda: f"social-clarifying-question-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    active_claim_ids: List[str] = Field(default_factory=list)
    active_claim_keys: List[str] = Field(default_factory=list)
    trigger: Literal["ambiguity", "cross_talk", "scope_unclear"] = "ambiguity"
    question_focus: Literal["scope", "target", "shared_core", "decision_criteria"] = "scope"
    question_text: str
    attributed_participants: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_level: SocialDeliberationAmbiguity = "high"
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialDeliberationDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    deliberation_decision_id: str = Field(default_factory=lambda: f"social-deliberation-decision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    active_claim_ids: List[str] = Field(default_factory=list)
    active_claim_keys: List[str] = Field(default_factory=list)
    decision_kind: SocialDeliberationDecisionKind = "stay_narrow"
    trigger: Optional[str] = None
    bridge_summary_id: Optional[str] = None
    clarifying_question_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_level: SocialDeliberationAmbiguity = "medium"
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
