from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialCalibrationKind = Literal[
    "reliable_continuity",
    "revised_often",
    "cautious_scope",
    "strong_summary_partner",
    "disagreement_prone",
    "unknown",
]
SocialCalibrationScope = Literal["peer_thread", "peer_room", "room_thread", "room"]
SocialCalibrationDecayHint = Literal[
    "decay_after_two_quiet_turns",
    "decay_after_topic_shift",
    "decay_after_unreinforced_repair",
    "manual_review",
]


class SocialCalibrationSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    signal_id: str = Field(default_factory=lambda: f"social-calibration-signal-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    participant_name: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    scope: SocialCalibrationScope = "peer_room"
    calibration_kind: SocialCalibrationKind = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    reversible: bool = True
    decay_hint: SocialCalibrationDecayHint = "decay_after_two_quiet_turns"
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialPeerCalibrationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    calibration_id: str = Field(default_factory=lambda: f"social-peer-calibration-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    participant_name: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    scope: SocialCalibrationScope = "peer_room"
    calibration_kind: SocialCalibrationKind = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    reversible: bool = True
    decay_hint: SocialCalibrationDecayHint = "decay_after_two_quiet_turns"
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    active_signal_ids: List[str] = Field(default_factory=list)
    caution_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    attribution_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialTrustBoundaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    boundary_id: str = Field(default_factory=lambda: f"social-trust-boundary-{uuid4()}")
    platform: str
    room_id: str
    participant_id: Optional[str] = None
    participant_name: Optional[str] = None
    thread_key: Optional[str] = None
    topic_scope: Optional[str] = None
    scope: SocialCalibrationScope = "peer_room"
    calibration_kind: SocialCalibrationKind = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    reversible: bool = True
    decay_hint: SocialCalibrationDecayHint = "decay_after_two_quiet_turns"
    treat_claims_as_provisional: bool = False
    summary_anchor: bool = False
    use_narrower_attribution: bool = False
    require_clarification_before_shared_ground: bool = False
    caution_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    attribution_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    reasons: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
