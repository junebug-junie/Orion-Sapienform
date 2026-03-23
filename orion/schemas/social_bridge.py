from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


RoomPlatform = Literal["callsyne"]
ParticipantKind = Literal["peer_ai", "human", "system"]
RoomDirection = Literal["inbound", "outbound", "skipped"]


class CallSyneRoomMessageV1(BaseModel):
    """Thin transport contract for inbound CallSyne-style room traffic."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    platform: RoomPlatform = "callsyne"
    room_id: str
    thread_id: Optional[str] = None
    message_id: str
    sender_id: str
    sender_name: Optional[str] = None
    sender_kind: ParticipantKind = "peer_ai"
    text: str
    created_at: str = Field(default_factory=_utcnow_iso)
    reply_to_message_id: Optional[str] = None
    mentions_orion: bool = False
    reply_to_sender_id: Optional[str] = None
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    mentioned_participant_ids: List[str] = Field(default_factory=list)
    mentioned_participant_names: List[str] = Field(default_factory=list)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExternalRoomParticipantV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    participant_ref: str = Field(default_factory=lambda: f"participant-{uuid4()}")
    platform: str
    room_id: str
    participant_id: str
    participant_name: Optional[str] = None
    participant_kind: ParticipantKind = "peer_ai"
    last_message_id: Optional[str] = None
    last_seen_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExternalRoomMessageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: str = Field(default_factory=lambda: f"external-room-event-{uuid4()}")
    correlation_id: Optional[str] = None
    platform: str
    room_id: str
    thread_id: Optional[str] = None
    direction: RoomDirection = "inbound"
    event_type: str = "message"
    transport_message_id: str
    reply_to_message_id: Optional[str] = None
    sender_id: str
    sender_name: Optional[str] = None
    sender_kind: ParticipantKind = "peer_ai"
    text: str
    source: str = "orion-social-room-bridge"
    observed_at: str = Field(default_factory=_utcnow_iso)
    transport_ts: Optional[str] = None
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    delivery_ok: Optional[bool] = None
    delivery_error: Optional[str] = None
    skip_reason: Optional[str] = None


class ExternalRoomPostRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    platform: str
    room_id: str
    thread_id: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExternalRoomPostResultV1(ExternalRoomMessageV1):
    direction: RoomDirection = "outbound"
    event_type: str = "post_result"
    delivery_ok: bool = True
    posted_at: str = Field(default_factory=_utcnow_iso)


class ExternalRoomTurnSkippedV1(ExternalRoomMessageV1):
    direction: RoomDirection = "skipped"
    event_type: str = "skipped"
    skip_reason: str
