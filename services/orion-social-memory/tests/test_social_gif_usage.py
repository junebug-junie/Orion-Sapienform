from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app.synthesizer import update_social_gif_usage_state
from orion.schemas.social_chat import SocialGroundingStateV1, SocialRedactionScoreV1, SocialRoomTurnStoredV1


def _turn(*, turn_id: str, gif: bool, intent_kind: str | None = None) -> SocialRoomTurnStoredV1:
    return SocialRoomTurnStoredV1(
        turn_id=turn_id,
        correlation_id=f"corr-{turn_id}",
        session_id="scenario:gifs",
        user_id="peer-1",
        source="test",
        profile="social_room",
        prompt="lol yes exactly",
        response="Absolutely.",
        text="User: lol yes exactly\nOrion: Absolutely.",
        created_at="2026-03-22T12:00:00+00:00",
        stored_at="2026-03-22T12:00:01+00:00",
        recall_profile="social.room.v1",
        trace_verb="chat_social_room",
        tags=["social_room"],
        grounding_state=SocialGroundingStateV1(),
        redaction=SocialRedactionScoreV1(),
        client_meta={
            "chat_profile": "social_room",
            "external_room": {
                "platform": "callsyne",
                "room_id": "room-alpha",
                "thread_id": "thread-gif",
            },
            "external_participant": {
                "participant_id": "peer-1",
                "participant_name": "CallSyne Peer",
                "participant_kind": "peer_ai",
            },
            "social_gif_policy": {
                "gif_allowed": gif,
                "decision_kind": "text_plus_gif" if gif else "text_only",
                "intent_kind": intent_kind,
                "metadata": {"transport_degraded": "false"},
            },
            "social_gif_intent": (
                {
                    "intent_kind": intent_kind,
                    "gif_query": "warm laugh with you reaction gif",
                }
                if gif and intent_kind
                else None
            ),
        },
    )


def test_social_gif_usage_state_tracks_density_and_cooldown() -> None:
    first = update_social_gif_usage_state(
        None,
        _turn(turn_id="turn-1", gif=True, intent_kind="laugh_with"),
        platform="callsyne",
        room_id="room-alpha",
        thread_key="callsyne:room-alpha:thread:thread-gif",
        participant_id="peer-1",
        participant_name="CallSyne Peer",
    )
    second = update_social_gif_usage_state(
        first,
        _turn(turn_id="turn-2", gif=False),
        platform="callsyne",
        room_id="room-alpha",
        thread_key="callsyne:room-alpha:thread:thread-gif",
        participant_id="peer-1",
        participant_name="CallSyne Peer",
    )

    assert first.consecutive_gif_turns == 1
    assert first.turns_since_last_orion_gif == 0
    assert first.recent_gif_turn_count == 1
    assert first.last_intent_kind == "laugh_with"
    assert second.consecutive_gif_turns == 0
    assert second.turns_since_last_orion_gif == 1
    assert second.recent_turn_was_gif[-2:] == [True, False]
