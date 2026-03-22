from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialArtifactType = Literal["shared_takeaway", "room_norm", "peer_cue"]
SocialArtifactScope = Literal["session_only", "room_local", "peer_local", "no_persistence"]
SocialArtifactDecisionState = Literal["proposed", "clarify_scope", "revised", "accepted", "declined", "deferred"]


class SocialArtifactProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    proposal_id: str = Field(default_factory=lambda: f"social-artifact-proposal-{uuid4()}")
    artifact_type: SocialArtifactType = "shared_takeaway"
    proposed_summary_text: str
    proposed_scope: SocialArtifactScope = "session_only"
    decision_state: SocialArtifactDecisionState = "proposed"
    confirmation_needed: bool = True
    rationale: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)


class SocialArtifactRevisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision_id: str = Field(default_factory=lambda: f"social-artifact-revision-{uuid4()}")
    proposal_id: Optional[str] = None
    artifact_type: SocialArtifactType = "shared_takeaway"
    prior_summary_text: str
    prior_scope: SocialArtifactScope = "session_only"
    revised_summary_text: str
    revised_scope: SocialArtifactScope = "session_only"
    decision_state: SocialArtifactDecisionState = "revised"
    confirmation_needed: bool = True
    rationale: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)


class SocialArtifactConfirmationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    confirmation_id: str = Field(default_factory=lambda: f"social-artifact-confirmation-{uuid4()}")
    proposal_id: Optional[str] = None
    artifact_type: SocialArtifactType = "shared_takeaway"
    decision_state: SocialArtifactDecisionState
    confirmed_summary_text: str = ""
    confirmed_scope: SocialArtifactScope = "no_persistence"
    confirmation_needed: bool = False
    rationale: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)
