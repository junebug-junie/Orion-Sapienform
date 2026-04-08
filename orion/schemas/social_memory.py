from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_artifact import (
    SocialArtifactConfirmationV1,
    SocialArtifactProposalV1,
    SocialArtifactRevisionV1,
)
from orion.schemas.social_commitment import SocialCommitmentV1
from orion.schemas.social_calibration import SocialCalibrationSignalV1, SocialPeerCalibrationV1, SocialTrustBoundaryV1
from orion.schemas.social_freshness import SocialDecaySignalV1, SocialMemoryFreshnessV1, SocialRegroundingDecisionV1
from orion.schemas.social_deliberation import (
    SocialBridgeSummaryV1,
    SocialClarifyingQuestionV1,
    SocialDeliberationDecisionV1,
)
from orion.schemas.social_floor import (
    SocialClosureSignalV1,
    SocialFloorDecisionV1,
    SocialTurnHandoffV1,
)
from orion.schemas.social_gif import SocialGifUsageStateV1
from orion.schemas.social_claim import (
    SocialClaimAttributionV1,
    SocialClaimRevisionV1,
    SocialClaimStanceV1,
    SocialConsensusStateV1,
    SocialDivergenceSignalV1,
)
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadStateV1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


TrustTier = Literal["new", "known", "steady"]
SharedArtifactStatus = Literal["unknown", "accepted", "declined", "deferred"]
SharedArtifactScope = Literal["peer_local", "room_local"]


class SocialParticipantContinuityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    peer_key: str
    platform: str
    room_id: str
    participant_id: str
    participant_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    participant_kind: str = "peer_ai"
    recent_shared_topics: List[str] = Field(default_factory=list)
    interaction_tone_summary: str = ""
    safe_continuity_summary: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    last_seen_at: str = Field(default_factory=_utcnow_iso)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_tier: TrustTier = "new"
    shared_artifact_scope: SharedArtifactScope = "peer_local"
    shared_artifact_status: SharedArtifactStatus = "unknown"
    shared_artifact_summary: str = ""
    shared_artifact_reason: str = ""
    shared_artifact_proposal: Optional[SocialArtifactProposalV1] = None
    shared_artifact_revision: Optional[SocialArtifactRevisionV1] = None
    shared_artifact_confirmation: Optional[SocialArtifactConfirmationV1] = None
    calibration_signals: List[SocialCalibrationSignalV1] = Field(default_factory=list)
    peer_calibration: Optional[SocialPeerCalibrationV1] = None
    trust_boundary: Optional[SocialTrustBoundaryV1] = None
    memory_freshness: List[SocialMemoryFreshnessV1] = Field(default_factory=list)
    decay_signals: List[SocialDecaySignalV1] = Field(default_factory=list)
    regrounding_decisions: List[SocialRegroundingDecisionV1] = Field(default_factory=list)


class SocialRoomContinuityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    room_key: str
    platform: str
    room_id: str
    recurring_topics: List[str] = Field(default_factory=list)
    active_participants: List[str] = Field(default_factory=list)
    recent_thread_summary: str = ""
    room_tone_summary: str = ""
    open_threads: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    last_updated_at: str = Field(default_factory=_utcnow_iso)
    shared_artifact_scope: SharedArtifactScope = "room_local"
    shared_artifact_status: SharedArtifactStatus = "unknown"
    shared_artifact_summary: str = ""
    shared_artifact_reason: str = ""
    shared_artifact_proposal: Optional[SocialArtifactProposalV1] = None
    shared_artifact_revision: Optional[SocialArtifactRevisionV1] = None
    shared_artifact_confirmation: Optional[SocialArtifactConfirmationV1] = None
    active_threads: List[SocialThreadStateV1] = Field(default_factory=list)
    current_thread_key: Optional[str] = None
    current_thread_summary: str = ""
    handoff_signal: Optional[SocialHandoffSignalV1] = None
    active_claims: List[SocialClaimStanceV1] = Field(default_factory=list)
    recent_claim_revisions: List[SocialClaimRevisionV1] = Field(default_factory=list)
    claim_attributions: List[SocialClaimAttributionV1] = Field(default_factory=list)
    claim_consensus_states: List[SocialConsensusStateV1] = Field(default_factory=list)
    claim_divergence_signals: List[SocialDivergenceSignalV1] = Field(default_factory=list)
    bridge_summary: Optional[SocialBridgeSummaryV1] = None
    clarifying_question: Optional[SocialClarifyingQuestionV1] = None
    deliberation_decision: Optional[SocialDeliberationDecisionV1] = None
    turn_handoff: Optional[SocialTurnHandoffV1] = None
    closure_signal: Optional[SocialClosureSignalV1] = None
    floor_decision: Optional[SocialFloorDecisionV1] = None
    gif_usage_state: Optional[SocialGifUsageStateV1] = None
    active_commitments: List[SocialCommitmentV1] = Field(default_factory=list)
    calibration_signals: List[SocialCalibrationSignalV1] = Field(default_factory=list)
    peer_calibrations: List[SocialPeerCalibrationV1] = Field(default_factory=list)
    trust_boundaries: List[SocialTrustBoundaryV1] = Field(default_factory=list)
    memory_freshness: List[SocialMemoryFreshnessV1] = Field(default_factory=list)
    decay_signals: List[SocialDecaySignalV1] = Field(default_factory=list)
    regrounding_decisions: List[SocialRegroundingDecisionV1] = Field(default_factory=list)


class SocialStanceSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stance_id: str = "orion-social-room"
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    warmth: float = Field(default=0.7, ge=0.0, le=1.0)
    directness: float = Field(default=0.6, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.3, ge=0.0, le=1.0)
    caution: float = Field(default=0.4, ge=0.0, le=1.0)
    depth_preference: float = Field(default=0.5, ge=0.0, le=1.0)
    recent_social_orientation_summary: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    last_updated_at: str = Field(default_factory=_utcnow_iso)


class SocialRelationalMemoryUpdateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    update_id: str = Field(default_factory=lambda: f"social-memory-update-{uuid4()}")
    turn_id: str
    correlation_id: Optional[str] = None
    platform: Optional[str] = None
    room_id: Optional[str] = None
    peer_key: Optional[str] = None
    participant_updated: bool = False
    room_updated: bool = False
    stance_updated: bool = False
    claim_count: int = 0
    claim_revision_count: int = 0
    evidence_count: int = 0
    emitted_at: str = Field(default_factory=_utcnow_iso)
