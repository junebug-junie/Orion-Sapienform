"""Room contracts for Claude as a third social-room participant.

Two payloads on two channels, both consumed by `orion-room-companion`:

    orion:room:claude:request    RoomClaudeRequestV1     Hub -> companion
    orion:room:claude:utterance  RoomClaudeUtteranceV1   companion -> Hub

Why a responder block exists at all: every social-room turn persisted today
identifies who spoke *to* Orion (`client_meta.external_participant`) and
assumes Orion answered. That is true of the live 4-participant `aitown-town`
room -- four NPCs, all addressing Orion -- but it cannot express a room where
a third participant answers Juniper directly. `ExternalRoomResponderV1` is the
missing half. It rides in `client_meta` alongside `external_participant`, so
absent still means Orion and no stored row needs migrating.

Deliberately NOT reusing `SocialRoomTurnV1`: that is the persistence contract
for a completed prompt/response pair. These two are the in-flight request and
reply between two services, carrying things a stored turn has no business
holding (a Claude session id, a per-turn dollar cost, a subprocess duration).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.social_bridge import ParticipantKind, RoomPlatform


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExternalRoomResponderV1(BaseModel):
    """Who produced a room reply. Mirrors `ExternalRoomParticipantV1`'s
    identity fields so a reader can treat prompter and responder the same
    way. Rides in `client_meta.external_responder`; absent means Orion."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    participant_id: str
    participant_name: Optional[str] = None
    participant_kind: ParticipantKind = "peer_ai"


class RoomTranscriptEntryV1(BaseModel):
    """One prior utterance, as Claude sees it.

    Speaker-shaped rather than prompt/response-shaped: this is the room's
    history from a third participant's seat, where "who said it" is the only
    thing that distinguishes two lines of text.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    speaker_id: str
    speaker_name: str
    speaker_kind: ParticipantKind = "human"
    text: str
    created_at: str = Field(default_factory=_utcnow_iso)


class RoomClaudeRequestV1(BaseModel):
    """Hub asks the companion for one Claude turn.

    v1 is explicit-invite only, so `invited_by` is always a human participant
    id. When Orion gains the ability to invite (v2), that is the field that
    changes meaning -- and the field a budget policy would key on.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"room-claude-req-{uuid4()}")
    correlation_id: Optional[str] = None
    platform: RoomPlatform = "hub"
    room_id: str
    session_id: Optional[str] = None
    invited_by: str
    prompt: str
    transcript: List[RoomTranscriptEntryV1] = Field(default_factory=list)
    # The social_memory /summary block (roster, room continuity, stance, open
    # threads) exactly as any room participant receives it. Left untyped here
    # on purpose: orion-social-memory owns that shape and this contract should
    # not fork a second copy of it that can drift.
    social_memory_summary: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)


class RoomClaudeUtteranceV1(BaseModel):
    """Claude's reply, plus the evidence that it was really Claude.

    `model` and `cost_usd` are copied verbatim from the CLI's own result
    object, never recomputed. They are not decoration: a turn that silently
    fell back to a local model through the FCC gateway would still return
    text, and these two fields are how that is caught. `cost_usd == 0.0` with
    non-empty text is a failure signal, not a free lunch.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    utterance_id: str = Field(default_factory=lambda: f"room-claude-utt-{uuid4()}")
    request_id: str
    correlation_id: Optional[str] = None
    platform: RoomPlatform = "hub"
    room_id: str
    session_id: Optional[str] = None
    responder: ExternalRoomResponderV1
    text: str
    claude_session_id: Optional[str] = None
    model: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    ok: bool = True
    error: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
