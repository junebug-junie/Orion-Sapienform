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
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from orion.schemas.social_bridge import ParticipantKind, RoomPlatform


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _drop_defaults(data: Any, defaults: Dict[str, Any]) -> Any:
    if isinstance(data, dict):
        for key, default in defaults.items():
            if key in data and data[key] == default:
                data.pop(key)
    return data


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
    # When the line was said. Hub fills it from chat_history_log.created_at
    # (2026-11-12); before that it defaulted to "now" for every line. The
    # companion's per-session "sent up to here" mark compares these.
    created_at: str = Field(default_factory=_utcnow_iso)
    # First 8 characters of the turn's correlation id, so Claude can name a
    # turn ("turn:1a2b3c4d") and a later tool can look it up.
    turn_ref: Optional[str] = None
    # Orion spoke without being asked (client_meta.unsolicited). Labelled so
    # Claude does not read it as an answer to the line above it.
    unprompted: bool = False
    # Part of a side exchange with Claude itself (Juniper's @claude prompt or
    # Claude's own reply). A resumed Claude session already holds these, so
    # the companion never re-sends them on --resume.
    side_exchange: bool = False

    @model_serializer(mode="wrap")
    def _omit_unset_new_fields(self, handler):  # type: ignore[no-untyped-def]
        # Every consumer validates with extra="forbid". Dropping the 2026-11-12
        # fields when they hold their defaults keeps a payload that does not use
        # them byte-identical to before, so a companion that has not been rebuilt
        # yet never sees a key it would reject. Same pattern as
        # orion/schemas/pre_turn_appraisal.py TurnWindowMessageV1.
        return _drop_defaults(handler(self), {"turn_ref": None, "unprompted": False, "side_exchange": False})


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
    # "manual" = a human clicked Ask Claude. "auto" = Claude was invited
    # without a human in the loop, and is therefore licensed to stay quiet
    # (`passed=True`) rather than owing a reply to a direct question.
    #
    # NOTHING PRODUCES "auto" TODAY. Its only producer was a post-turn hook in
    # websocket_handler.py that fired after EVERY Hub chat turn; that was
    # removed 2026-09-08 because it shipped Juniper's private conversations to
    # Claude and billed a call per turn. The next producer is the endogenous
    # stuck-prior trigger (orion/autonomy/ask_claude_trigger.py), read-only
    # until deliberately armed -- which is why the field and the pass licence
    # both survive rather than being deleted with the hook.
    trigger: Literal["manual", "auto"] = "manual"
    prompt: str
    transcript: List[RoomTranscriptEntryV1] = Field(default_factory=list)
    # The social_memory /summary block (roster, room continuity, stance, open
    # threads) exactly as any room participant receives it. Left untyped here
    # on purpose: orion-social-memory owns that shape and this contract should
    # not fork a second copy of it that can drift.
    social_memory_summary: Dict[str, Any] = Field(default_factory=dict)
    # "What is open in Hub now": names and counts only, never content, at most
    # 800 characters, built by Hub at invite time. Shown on a session's first
    # turn only. None = Hub did not send one.
    hub_index: Optional[str] = None
    # One line about older turns that did not fit the transcript cap, e.g.
    # "[Earlier: 323 turns from 2026-09-18 to 2026-11-05 not shown.]".
    transcript_note: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)

    @model_serializer(mode="wrap")
    def _omit_unset_new_fields(self, handler):  # type: ignore[no-untyped-def]
        # See RoomTranscriptEntryV1: unset 2026-11-12 fields are left out.
        return _drop_defaults(handler(self), {"hub_index": None, "transcript_note": None})


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
    # Claude chose to stay quiet. ok=True and text="" -- NOT a failure.
    #
    # A pass is still published and still carries its real cost_usd, because
    # deciding not to speak costs the same tokens as speaking. Suppressing the
    # event would hide that spend from the meter, which is the one thing the
    # v2 budget work cannot afford. Consumers render nothing; accounting still
    # sees it.
    passed: bool = False
    error: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)
