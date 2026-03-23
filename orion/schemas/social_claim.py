from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialClaimKind = Literal["peer_claim", "orion_claim", "shared_summary", "inferred_claim"]
SocialClaimState = Literal["provisional", "accepted", "disputed", "corrected", "revised", "withdrawn"]
SocialParticipantClaimStance = Literal["support", "question", "dispute", "correct", "withdraw", "unknown"]
SocialConsensusLabel = Literal["none", "emerging", "partial", "contested", "consensus", "corrected"]
SocialClaimSourceBasis = Literal[
    "recent_turns",
    "social_memory",
    "repair_context",
    "epistemic_context",
    "shared_artifact_confirmation",
]
SocialClaimRevisionType = Literal["disputed", "corrected", "revised", "withdrawn"]


class SocialClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    claim_id: str = Field(default_factory=lambda: f"social-claim-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    source_participant_id: Optional[str] = None
    source_participant_name: Optional[str] = None
    claim_text: str
    normalized_summary: str
    claim_kind: SocialClaimKind
    stance: SocialClaimState = "provisional"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_basis: SocialClaimSourceBasis = "recent_turns"
    related_claim_ids: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialClaimRevisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision_id: str = Field(default_factory=lambda: f"social-claim-revision-{uuid4()}")
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    claim_id: str
    revision_type: SocialClaimRevisionType
    prior_stance: SocialClaimState
    new_stance: SocialClaimState
    source_participant_id: Optional[str] = None
    source_participant_name: Optional[str] = None
    revised_summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_basis: SocialClaimSourceBasis = "recent_turns"
    related_claim_ids: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialClaimStanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stance_id: str = Field(default_factory=lambda: f"social-claim-stance-{uuid4()}")
    claim_id: str
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    source_participant_id: Optional[str] = None
    source_participant_name: Optional[str] = None
    claim_kind: SocialClaimKind
    normalized_summary: str
    current_stance: SocialClaimState = "provisional"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_basis: SocialClaimSourceBasis = "recent_turns"
    related_claim_ids: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialClaimAttributionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    attribution_id: str = Field(default_factory=lambda: f"social-claim-attribution-{uuid4()}")
    claim_id: str
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    normalized_claim_key: str
    attributed_participant_ids: List[str] = Field(default_factory=list)
    attributed_participant_names: Dict[str, str] = Field(default_factory=dict)
    participant_stances: Dict[str, SocialParticipantClaimStance] = Field(default_factory=dict)
    orion_stance: SocialParticipantClaimStance = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_count: int = 0
    updated_at: str = Field(default_factory=_utcnow_iso)
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialConsensusStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    consensus_id: str = Field(default_factory=lambda: f"social-consensus-state-{uuid4()}")
    claim_id: str
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    normalized_claim_key: str
    consensus_state: SocialConsensusLabel = "none"
    supporting_participant_ids: List[str] = Field(default_factory=list)
    disputing_participant_ids: List[str] = Field(default_factory=list)
    questioning_participant_ids: List[str] = Field(default_factory=list)
    orion_stance: SocialParticipantClaimStance = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_count: int = 0
    updated_at: str = Field(default_factory=_utcnow_iso)
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialDivergenceSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    divergence_id: str = Field(default_factory=lambda: f"social-divergence-signal-{uuid4()}")
    claim_id: str
    platform: str
    room_id: str
    thread_key: Optional[str] = None
    normalized_claim_key: str
    divergence_detected: bool = True
    consensus_state: SocialConsensusLabel = "none"
    participant_stances: Dict[str, SocialParticipantClaimStance] = Field(default_factory=dict)
    orion_stance: SocialParticipantClaimStance = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_count: int = 0
    updated_at: str = Field(default_factory=_utcnow_iso)
    reasons: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
