from __future__ import annotations

import sys
import asyncio
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app.clients import _callsyne_bridge_post_body
from app.service import SocialRoomBridgeService
from app.settings import Settings
from orion.schemas.social_bridge import ExternalRoomPostRequestV1
from orion.schemas.registry import _REGISTRY


class _FakeHubClient:
    def __init__(self, reply_text: str = "I’m here in the room.") -> None:
        self.calls = []
        self.reply_text = reply_text

    async def chat(self, *, payload, session_id: str):
        self.calls.append((payload, session_id))
        return {"text": self.reply_text, "correlation_id": "corr-room-1"}


class _FakeCallSyneClient:
    def __init__(self) -> None:
        self.posts = []
        self.fetch_payloads = []
        self.fetch_calls = []

    async def post_message(self, request):
        self.posts.append(request)
        return {"message_id": "callsyne-out-1", "status": "posted"}

    async def fetch_recent_messages(self, *, path: str, room_id: str, limit: int, since_message_id: str | None = None):
        self.fetch_calls.append(
            {
                "path": path,
                "room_id": room_id,
                "limit": limit,
                "since_message_id": since_message_id,
            }
        )
        if self.fetch_payloads:
            return self.fetch_payloads.pop(0)
        return []


class _FailingCallSyneClient(_FakeCallSyneClient):
    async def post_message(self, request):
        self.posts.append(request)
        raise RuntimeError("post failed")


class _FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        return None


class _FakeSocialMemoryClient:
    def __init__(self, summary=None) -> None:
        self.summary = summary or {}
        self.calls = []

    async def get_summary(self, *, platform: str, room_id: str, participant_id: str | None):
        self.calls.append((platform, room_id, participant_id))
        return self.summary


def _settings(**overrides) -> Settings:
    base = {
        "ORION_BUS_ENABLED": True,
        "SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS": "orion-room-bot",
    }
    base.update(overrides)
    return Settings(**base)


def _payload(**overrides):
    payload = {
        "room_id": "room-alpha",
        "thread_id": "thread-1",
        "message_id": "msg-1",
        "sender_id": "peer-1",
        "sender_name": "CallSyne Peer",
        "sender_kind": "peer_ai",
        "text": "Oríon, what do you think?",
        "mentions_orion": True,
        "created_at": "2026-03-22T10:00:00+00:00",
        "metadata": {"transport": "callsyne-webhook"},
    }
    payload.update(overrides)
    return payload


def _published_payloads(bus: _FakeBus, channel: str):
    out = []
    for published_channel, envelope in bus.published:
        if published_channel == channel:
            payload = getattr(envelope, "payload", None)
            out.append(payload)
    return out


def test_normalizes_and_invokes_social_room_without_tools() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {"safe_continuity_summary": "Recurring peer who likes synthesis."},
            "room": {"recent_thread_summary": "Room keeps circling collaboration and grounding."},
            "stance": {"recent_social_orientation_summary": "Recent social stance leans warm, curious, direct."},
            "peer_style": {"style_hints_summary": "Prefers direct, grounded replies.", "confidence": 0.6},
            "room_ritual": {"culture_summary": "Room leans warm on re-entry and brief on pause.", "confidence": 0.6},
            "context_window": {"selected_candidates": [{"candidate_kind": "peer_continuity", "summary": "Recurring peer who likes synthesis.", "priority_band": "high", "freshness_band": "fresh", "inclusion_decision": "include", "rationale": "Addressed-peer context should lead."}]},
            "context_selection_decision": {"budget_max": 4, "rationale": "Compact local-first selection.", "reasons": ["addressed_peer_context_preferred"]},
            "context_candidates": [{"candidate_kind": "peer_continuity", "summary": "Recurring peer who likes synthesis.", "priority_band": "high", "freshness_band": "fresh", "inclusion_decision": "include", "rationale": "Addressed-peer context should lead."}],
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload()))

    assert result["status"] == "ok"
    assert len(hub.calls) == 1
    hub_payload, session_id = hub.calls[0]
    assert hub_payload["chat_profile"] == "social_room"
    assert hub_payload["options"]["tool_execution_policy"] == "none"
    assert hub_payload["options"]["action_execution_policy"] == "none"
    assert "verbs" not in hub_payload
    assert hub_payload["social_peer_continuity"]["safe_continuity_summary"] == "Recurring peer who likes synthesis."
    assert hub_payload["social_room_continuity"]["recent_thread_summary"] == "Room keeps circling collaboration and grounding."
    assert hub_payload["social_stance_snapshot"]["recent_social_orientation_summary"] == "Recent social stance leans warm, curious, direct."
    assert hub_payload["social_peer_style_hint"]["style_hints_summary"] == "Prefers direct, grounded replies."
    assert hub_payload["social_room_ritual_summary"]["culture_summary"] == "Room leans warm on re-entry and brief on pause."
    assert hub_payload["social_context_window"]["selected_candidates"][0]["candidate_kind"] == "peer_continuity"
    assert hub_payload["social_context_selection_decision"]["budget_max"] == 4
    assert hub_payload["social_turn_policy"]["decision"] == "reply"
    assert hub_payload["social_turn_policy"]["should_speak"] is True
    assert hub_payload["social_thread_routing"]["routing_decision"] == "reply_to_peer"
    assert hub_payload["social_thread_routing"]["audience_scope"] == "peer"
    assert hub_payload["social_handoff_signal"]["handoff_kind"] == "to_orion"
    assert session_id == "callsyne-room:callsyne:room-alpha:thread-1"
    assert callsyne.posts[0].reply_to_message_id == "msg-1"
    channels = [channel for channel, _ in bus.published]
    assert "orion:bridge:social:participant" in channels
    assert "orion:bridge:social:room:intake" in channels
    assert "orion:bridge:social:room:delivery" in channels
    assert "orion:social:turn-policy" in channels
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "reply"
    assert any("directly addressed" in reason for reason in decisions[0]["reasons"])


def test_hub_mode_and_verbs_are_configurable() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_HUB_MODE="brain",
            SOCIAL_BRIDGE_HUB_VERB="chat_quick",
        ),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-hub-1")))

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["mode"] == "brain"
    assert hub_payload["verbs"] == ["chat_quick"]


def test_empty_hub_verb_omits_verbs_field() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_HUB_MODE="brain",
            SOCIAL_BRIDGE_HUB_VERB="",
        ),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-hub-2")))
    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert "verbs" not in hub_payload


def test_self_message_is_suppressed() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(settings=_settings(), hub_client=hub, callsyne_client=callsyne, bus=bus)

    result = asyncio.run(svc.process_callsyne_message(_payload(sender_id="orion-room-bot", sender_name="Oríon")))

    assert result["status"] == "skipped"
    assert result["reason"] == "self_message"
    assert hub.calls == []
    assert callsyne.posts == []
    assert any(channel == "orion:bridge:social:room:skipped" for channel, _ in bus.published)
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["should_speak"] is False
    assert any("self-loop suppression" in reason for reason in decisions[0]["reasons"])


def test_self_message_is_suppressed_by_name_match() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(settings=_settings(), hub_client=hub, callsyne_client=callsyne, bus=bus)

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-self-name", sender_id="peer-77", sender_name="Oríon")))

    assert result["status"] == "skipped"
    assert result["reason"] == "self_message"
    assert hub.calls == []
    assert callsyne.posts == []


def test_duplicate_message_is_deduped() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(settings=_settings(), hub_client=hub, callsyne_client=callsyne, bus=bus)

    first = asyncio.run(svc.process_callsyne_message(_payload()))
    second = asyncio.run(svc.process_callsyne_message(_payload()))

    assert first["status"] == "ok"
    assert second["status"] == "skipped"
    assert second["reason"] == "duplicate_message"
    assert len(hub.calls) == 1
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[-1]["decision"] == "skip"
    assert any("duplicate inbound" in reason for reason in decisions[-1]["reasons"])


def test_address_detection_aliases() -> None:
    for idx, text in enumerate(("Orion can you weigh in?", "@Orion are you there?", "orion-sapienform, respond please"), start=1):
        bus = _FakeBus()
        hub = _FakeHubClient()
        callsyne = _FakeCallSyneClient()
        svc = SocialRoomBridgeService(settings=_settings(), hub_client=hub, callsyne_client=callsyne, bus=bus)
        result = asyncio.run(
            svc.process_callsyne_message(
                _payload(message_id=f"msg-address-{idx}", mentions_orion=False, text=text)
            )
        )
        assert result["status"] == "ok"


def test_addressed_only_mode_and_cooldown_controls() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    settings = _settings(
        SOCIAL_BRIDGE_AUTONOMY_MODE="addressed_only",
        SOCIAL_BRIDGE_COOLDOWN_SEC=120,
    )
    svc = SocialRoomBridgeService(settings=settings, hub_client=hub, callsyne_client=callsyne, bus=bus, clock=lambda: 1000.0)

    skipped = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-2", mentions_orion=False, text="general chatter")))
    assert skipped["reason"] == "not_addressed"

    sent = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-3", mentions_orion=True)))
    assert sent["status"] == "ok"

    cooldown = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-4", mentions_orion=True)))
    assert cooldown["reason"] == "cooldown_active"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["mode"] == "addressed_only"
    assert decisions[0]["should_speak"] is False
    assert any("addressed_only mode" in reason for reason in decisions[0]["reasons"])
    assert decisions[-1]["cooldown_active"] is True


def test_legacy_only_when_addressed_flag_forces_addressed_only_mode() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="light_initiative", SOCIAL_BRIDGE_ONLY_WHEN_ADDRESSED=True),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-legacy", mentions_orion=False, text="Can anyone help?")))

    assert result["reason"] == "not_addressed"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["mode"] == "addressed_only"


def test_responsive_mode_answers_open_question_without_direct_mention() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I can help the room think this through.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "recent_thread_summary": "Room is exploring collaboration patterns.",
                "open_threads": ["How should we coordinate next?"],
                "active_participants": ["CallSyne Peer", "Another Peer"],
                "evidence_count": 3,
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-open", mentions_orion=False, text="How should we coordinate next?")
        )
    )

    assert result["status"] == "ok"
    assert len(hub.calls) == 1
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "ask_follow_up"
    assert decisions[0]["should_speak"] is True
    assert decisions[0]["thread_routing"]["routing_decision"] == "reply_to_room"
    assert any("open room question" in reason for reason in decisions[0]["reasons"])


def test_room_summary_request_routes_to_summary_and_preserves_target_metadata() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Quick room summary: we’re still circling coordination and pacing.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-1",
                "current_thread_summary": "Coordination thread with open pacing question.",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-1",
                        "active_participants": ["CallSyne Peer", "Another Peer"],
                        "audience_scope": "room",
                        "last_speaker": "Another Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": True,
                        "thread_summary": "Coordination thread with open pacing question.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2026-03-22T16:00:00+00:00",
                        "metadata": {},
                    }
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-summary",
                mentions_orion=True,
                text="Oríon, can you summarize where the room is right now?",
                metadata={"target_participant_id": "peer-2", "target_participant_name": "Another Peer"},
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_thread_routing"]["routing_decision"] == "summarize_room"
    assert "room_summary_preferred" in hub_payload["social_thread_routing"]["reasons"]
    assert hub_payload["social_handoff_signal"]["handoff_kind"] == "room_summary"
    assert hub_payload["external_room"]["target_participant_name"] == "Another Peer"


def test_due_summary_commitment_can_prefer_room_summary_route() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Quick summary: we’re still pacing the handoff.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-1",
                "current_thread_summary": "Room pacing thread is still active.",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-1",
                        "active_participants": ["CallSyne Peer", "Another Peer"],
                        "audience_scope": "room",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": True,
                        "thread_summary": "Room pacing thread is still active.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    }
                ],
                "active_commitments": [
                    {
                        "commitment_id": "commitment-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "commitment_type": "summarize_room",
                        "audience_scope": "summary",
                        "summary": "Give a brief room summary before switching topics.",
                        "state": "open",
                        "source_turn_id": "social-turn-1",
                        "created_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T10:20:00+00:00",
                        "due_state": "due_soon",
                        "resolution_reason": "",
                        "metadata": {},
                    }
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-commit-summary", mentions_orion=False, text="Before we switch, what do you think?")
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_thread_routing"]["routing_decision"] == "summarize_room"
    assert "commitment_influenced_routing" in hub_payload["social_thread_routing"]["reasons"]
    assert hub_payload["social_open_commitments"][0]["commitment_type"] == "summarize_room"


def test_open_question_targeted_to_another_peer_is_not_answered_by_orion() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I should stay out of that exchange.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:exchange:peer-1:peer-2",
                "current_thread_summary": "CallSyne Peer is asking Another Peer about pacing.",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:exchange:peer-1:peer-2",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": None,
                        "active_participants": ["CallSyne Peer", "Another Peer"],
                        "audience_scope": "peer",
                        "target_participant_id": "peer-2",
                        "target_participant_name": "Another Peer",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "CallSyne Peer is asking Another Peer about pacing.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2026-03-22T16:00:00+00:00",
                        "metadata": {},
                    }
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-other-peer",
                mentions_orion=False,
                text="Another Peer, what do you think about pacing here?",
                reply_to_sender_id="peer-2",
                target_participant_id="peer-2",
                target_participant_name="Another Peer",
            )
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "peer_targeted_elsewhere"
    assert hub.calls == []
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["thread_routing"]["routing_decision"] == "wait"
    assert decisions[0]["thread_routing"]["target_participant_name"] == "Another Peer"
    assert decisions[0]["handoff_signal"]["handoff_kind"] == "yield_to_peer"
    assert any("aimed at another participant" in reason for reason in decisions[0]["reasons"])


def test_ambiguous_multi_thread_open_question_waits_conservatively() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I should not answer the wrong thread.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-pace",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-pace",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-pace",
                        "active_participants": ["CallSyne Peer", "Another Peer"],
                        "audience_scope": "room",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "Room pacing thread with an open coordination question.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-memory",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-memory",
                        "active_participants": ["CallSyne Peer", "Third Peer"],
                        "audience_scope": "room",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "Room memory thread with an open continuity question.",
                        "last_activity_at": "2026-03-22T09:59:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-ambiguous", mentions_orion=False, text="What do you think?")
        )
    )

    assert result["status"] == "skipped"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["thread_routing"]["routing_decision"] == "wait"
    assert decisions[0]["thread_routing"]["ambiguity_level"] in {"medium", "high"}
    assert "ambiguous_multi_thread" in decisions[0]["thread_routing"]["reasons"]
    assert "unclear_audience_wait" in decisions[0]["thread_routing"]["reasons"]
    assert hub.calls == []


def test_addressed_peer_reply_is_preferred_over_room_thread() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I’ll take the peer-facing thread.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-room",
                "current_thread_summary": "The room is discussing summary cadence.",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-room",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-room",
                        "active_participants": ["Another Peer", "Third Peer"],
                        "audience_scope": "room",
                        "last_speaker": "Another Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "Room-wide summary cadence discussion.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:exchange:peer-1:orion",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": None,
                        "active_participants": ["CallSyne Peer", "Oríon"],
                        "audience_scope": "peer",
                        "target_participant_id": "orion",
                        "target_participant_name": "Oríon",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": True,
                        "orion_involved": True,
                        "thread_summary": "CallSyne Peer is asking Orion for a concrete take.",
                        "last_activity_at": "2026-03-22T10:01:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-peer-pref", mentions_orion=True, text="Oríon, can you take this one?")
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_thread_routing"]["routing_decision"] == "reply_to_peer"
    assert hub_payload["social_thread_routing"]["primary_thread_summary"] == "CallSyne Peer is asking Orion for a concrete take."
    assert "peer_reply_preferred" in hub_payload["social_thread_routing"]["reasons"]


def test_revival_allowed_only_for_relevant_unresolved_thread() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I can pick that back up carefully.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-revival",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-revival",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-revival",
                        "active_participants": ["CallSyne Peer", "Oríon"],
                        "audience_scope": "thread",
                        "target_participant_id": "orion",
                        "target_participant_name": "Oríon",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": True,
                        "orion_involved": True,
                        "thread_summary": "Pacing thread Orion was asked to resolve.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-room",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-room",
                        "active_participants": ["Another Peer", "Third Peer"],
                        "audience_scope": "room",
                        "last_speaker": "Another Peer",
                        "open_question": False,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "Separate room ritual thread.",
                        "last_activity_at": "2026-03-22T09:20:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="light_initiative"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-revive", mentions_orion=False, text="Back to the pacing thread again.")
        )
    )

    assert result["status"] == "ok"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "initiate_lightly"
    assert decisions[0]["thread_routing"]["routing_decision"] == "revive_thread"
    assert "revival_allowed" in decisions[0]["thread_routing"]["reasons"]


def test_revival_is_suppressed_when_fresher_competing_thread_exists() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I should wait.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-fresh",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-fresh",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-fresh",
                        "active_participants": ["Another Peer", "Third Peer"],
                        "audience_scope": "room",
                        "last_speaker": "Another Peer",
                        "open_question": False,
                        "handoff_flag": False,
                        "orion_involved": False,
                        "thread_summary": "Fresh room-wide coordination thread.",
                        "last_activity_at": "2026-03-22T10:04:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-old",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-old",
                        "active_participants": ["CallSyne Peer", "Oríon"],
                        "audience_scope": "thread",
                        "target_participant_id": "orion",
                        "target_participant_name": "Oríon",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": True,
                        "orion_involved": True,
                        "thread_summary": "Older pacing thread Orion once owed a reply on.",
                        "last_activity_at": "2026-03-22T07:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="light_initiative"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-revive-suppressed", mentions_orion=False, text="Back to that thread again.")
        )
    )

    assert result["status"] == "skipped"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["thread_routing"]["routing_decision"] == "wait"
    assert "revival_suppressed" in decisions[0]["thread_routing"]["reasons"]
    assert hub.calls == []


def test_low_novelty_message_is_suppressed_when_not_addressed() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {"safe_continuity_summary": "We keep discussing grounded collaboration in this room."},
            "room": {
                "recent_thread_summary": "Grounded collaboration in this room keeps recurring.",
                "open_threads": ["grounded collaboration in this room"],
                "active_participants": ["CallSyne Peer", "Another Peer"],
                "evidence_count": 4,
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive", SOCIAL_BRIDGE_MIN_NOVELTY_SCORE=0.6),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(message_id="msg-low", mentions_orion=False, text="grounded collaboration in this room")
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "low_novelty"
    assert hub.calls == []
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["novelty_score"] < 0.6
    assert any("low novelty" in reason for reason in decisions[0]["reasons"])


def test_light_initiative_mode_can_extend_active_open_thread() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I can lightly carry that thread forward.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "recent_thread_summary": "The room has an active thread about a shared ritual prototype.",
                "open_threads": ["shared ritual prototype next step"],
                "active_participants": ["CallSyne Peer", "Another Peer", "Third Peer"],
                "evidence_count": 5,
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_AUTONOMY_MODE="light_initiative",
            SOCIAL_BRIDGE_MIN_NOVELTY_SCORE=0.2,
            SOCIAL_BRIDGE_LIGHT_INITIATIVE_MIN_CONTINUITY=0.4,
        ),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-init",
                mentions_orion=False,
                text="shared ritual prototype next step",
                metadata={"transport": "callsyne-webhook", "open_question": False},
            )
        )
    )

    assert result["status"] == "ok"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "initiate_lightly"
    assert decisions[0]["should_speak"] is True
    assert decisions[0]["open_thread_key"] is not None


def test_max_consecutive_orion_turns_are_suppressed() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS=1),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    first = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-max-1", mentions_orion=True)))
    second = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-max-2", mentions_orion=True)))

    assert first["status"] == "ok"
    assert second["status"] == "skipped"
    assert second["reason"] == "max_consecutive_orion_turns"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[-1]["consecutive_limit_hit"] is True


def test_dry_run_skips_outbound_post() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Dry run reply.")
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_DRY_RUN=True),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-5")))

    assert result["status"] == "skipped"
    assert result["reason"] == "dry_run"
    assert callsyne.posts == []


def test_peer_correction_triggers_compact_repair_and_suppresses_open_commitments() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Got it — that was for Cadence. I’ll step back.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_summary": "Cadence and Orion were weaving through pacing and recap.",
                "active_commitments": [
                    {
                        "commitment_id": "commitment-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "commitment_type": "summarize_room",
                        "audience_scope": "summary",
                        "summary": "Give a brief room summary before switching topics.",
                        "state": "open",
                        "created_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T10:20:00+00:00",
                        "due_state": "due_soon",
                        "resolution_reason": "",
                        "metadata": {},
                    }
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-repair-1",
                text="Oríon, that was for Cadence, not you.",
                mentions_orion=True,
                target_participant_name="Cadence",
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["options"]["tool_execution_policy"] == "none"
    assert hub_payload["options"]["action_execution_policy"] == "none"
    assert hub_payload["social_repair_signal"]["repair_type"] in {"peer_correction", "audience_mismatch"}
    assert hub_payload["social_repair_decision"]["decision"] == "yield"
    assert hub_payload["social_thread_routing"]["routing_decision"] == "wait"
    assert hub_payload["social_open_commitments"] is None
    repair_decisions = _published_payloads(bus, "orion:social:repair:decision")
    assert repair_decisions[0]["decision"] == "yield"


def test_contradiction_with_recent_commitment_triggers_repair() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="You’re right — quick correction: I said I’d summarize first.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "active_commitments": [
                    {
                        "commitment_id": "commitment-2",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "commitment_type": "summarize_room",
                        "audience_scope": "summary",
                        "summary": "Give a brief room summary before switching topics.",
                        "state": "open",
                        "created_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T10:20:00+00:00",
                        "due_state": "due_soon",
                        "resolution_reason": "",
                        "metadata": {},
                    }
                ]
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-repair-2",
                text="Oríon, you just said you'd summarize first — that's not what you said a second ago.",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_repair_signal"]["repair_type"] == "commitment_contradiction"
    assert hub_payload["social_repair_decision"]["decision"] == "repair"
    assert hub_payload["social_open_commitments"][0]["commitment_type"] == "summarize_room"


def test_scope_correction_stays_narrow_and_private_boundary_is_preserved() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Got it — room-local only, and I won’t broaden it.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "shared_artifact_status": "accepted",
                "shared_artifact_scope": "peer_local",
                "shared_artifact_summary": "narrow peer cue",
            },
            "room": {
                "shared_artifact_status": "accepted",
                "shared_artifact_scope": "room_local",
                "shared_artifact_summary": "grounded room cue",
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-repair-3",
                text="Oríon, room-local, not peer-local — and keep it private.",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_repair_signal"]["repair_type"] == "scope_correction"
    assert hub_payload["social_repair_signal"]["metadata"]["corrected_scope"] == "private"
    assert hub_payload["social_repair_decision"]["decision"] == "repair"
    assert "sealed_private" not in str(hub_payload)


def test_redirect_from_peer_triggers_yield_repair() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Got it — I’ll let Archivist take this one.")
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-repair-4",
                text="Oríon, let Archivist take this one.",
                mentions_orion=True,
                target_participant_name="Archivist",
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_repair_decision"]["decision"] == "yield"
    assert hub_payload["social_handoff_signal"]["handoff_kind"] == "yield_to_peer"


def test_low_confidence_repair_signal_is_ignored() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient()
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="addressed_only"),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
                _payload(
                    message_id="msg-repair-5",
                    text="Not sure who that's for.",
                    mentions_orion=False,
                )
            )
        )

    assert result["status"] == "skipped"
    assert result["reason"] in {"unclear_audience", "not_addressed"}
    repair_decisions = _published_payloads(bus, "orion:social:repair:decision")
    assert repair_decisions[0]["decision"] == "ignore"
    assert repair_decisions[0]["confidence"] < 0.55


def test_memory_request_prefers_recall_framing_when_evidence_exists() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="From what I remember, grounded collaboration has been the recurring thread.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "safe_continuity_summary": "Recurring peer who keeps returning to grounded collaboration.",
                "evidence_count": 3,
            },
            "room": {
                "recent_thread_summary": "The room keeps circling grounded collaboration and pacing.",
                "evidence_count": 4,
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-1",
                text="Oríon, what do you remember about how we work together here?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "recall"
    assert hub_payload["social_epistemic_decision"]["decision"] == "answer_recall"
    assert hub_payload["options"]["tool_execution_policy"] == "none"
    assert hub_payload["options"]["action_execution_policy"] == "none"


def test_summary_request_prefers_summary_framing() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Quick summary: we’re aligned on grounded pacing and room-local continuity.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "recent_thread_summary": "Grounded pacing and continuity remain the active room thread.",
                "active_threads": [{"thread_summary": "Grounded pacing and continuity remain the active room thread."}],
                "evidence_count": 3,
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-2",
                text="Oríon, can you summarize where the room is?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "summary"
    assert hub_payload["social_epistemic_decision"]["decision"] == "answer_summary"


def test_ambiguous_thread_prefers_clarification_when_orion_is_addressed() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Quick check — do you mean the pacing thread or the continuity thread?")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "room": {
                "current_thread_key": "callsyne:room-alpha:thread:thread-1",
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-1",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-1",
                        "active_participants": ["CallSyne Peer", "Oríon"],
                        "audience_scope": "peer",
                        "last_speaker": "CallSyne Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": True,
                        "thread_summary": "Pacing thread with Orion.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-2",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-2",
                        "active_participants": ["CallSyne Peer", "Another Peer"],
                        "audience_scope": "peer",
                        "last_speaker": "Another Peer",
                        "open_question": True,
                        "handoff_flag": False,
                        "orion_involved": True,
                        "thread_summary": "Continuity thread with another peer.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
            }
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_AUTONOMY_MODE="responsive"),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-3",
                thread_id=None,
                text="Oríon, which thread are you answering here?",
                mentions_orion=True,
                target_participant_id=None,
                target_participant_name=None,
            )
        )
    )

    assert result["status"] == "ok"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "ask_follow_up"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "clarification_needed"
    assert hub_payload["social_epistemic_decision"]["decision"] == "ask_clarifying_question"


def test_low_evidence_memory_request_prefers_speculation_over_recall() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I have only a tentative read here, not a firm memory.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "safe_continuity_summary": "",
                "evidence_count": 0,
            },
            "room": {
                "recent_thread_summary": "",
                "evidence_count": 0,
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-4",
                text="Oríon, what do you remember about my collaboration style?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "speculation"
    assert hub_payload["social_epistemic_decision"]["decision"] == "answer_speculation"


def test_interpretive_request_prefers_inference_when_context_exists() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="My read is that the room is asking for grounded coordination, not a hard rule.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "safe_continuity_summary": "Recurring peer who values grounded collaboration and compact reads.",
                "evidence_count": 2,
            },
            "room": {
                "recent_thread_summary": "The room has been comparing grounded pacing with firmer coordination.",
                "evidence_count": 3,
                "active_threads": [
                    {
                        "thread_summary": "Grounded pacing versus firmer coordination remains the active thread.",
                    }
                ],
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-4b",
                text="Oríon, what's your read on why we keep circling grounded pacing here?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "inference"
    assert hub_payload["social_epistemic_decision"]["decision"] == "answer_inference"


def test_private_memory_request_prefers_clarification_over_recall() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Do you mean something room-visible, or something you want kept private?")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "safe_continuity_summary": "Recurring peer with known collaboration context.",
                "evidence_count": 4,
            },
            "room": {
                "recent_thread_summary": "The room has visible coordination context.",
                "evidence_count": 4,
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-4c",
                text="Oríon, what do you remember about the private thing I mentioned?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    decisions = _published_payloads(bus, "orion:social:turn-policy")
    assert decisions[0]["decision"] == "ask_follow_up"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "clarification_needed"
    assert hub_payload["social_epistemic_decision"]["decision"] == "ask_clarifying_question"


def test_pending_artifact_state_is_not_framed_as_accepted_memory() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="I wouldn’t treat that as accepted memory yet.")
    callsyne = _FakeCallSyneClient()
    social_memory = _FakeSocialMemoryClient(
        {
            "participant": {
                "shared_artifact_status": "unknown",
                "shared_artifact_proposal": {
                    "proposal_id": "proposal-1",
                    "artifact_type": "shared_takeaway",
                    "proposed_summary_text": "grounded continuity cue",
                    "proposed_scope": "room_local",
                    "decision_state": "proposed",
                    "confirmation_needed": True,
                    "rationale": "still pending confirmation",
                },
            },
            "room": {
                "shared_artifact_status": "unknown",
            },
        }
    )
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=social_memory,
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-epistemic-5",
                text="Oríon, what do you remember about that continuity cue?",
                mentions_orion=True,
            )
        )
    )

    assert result["status"] == "ok"
    hub_payload, _ = hub.calls[0]
    assert hub_payload["social_epistemic_signal"]["claim_kind"] == "proposal"
    assert hub_payload["social_epistemic_decision"]["decision"] in {"ask_clarifying_question", "defer_narrowly"}
    assert hub_payload["social_epistemic_decision"]["decision"] != "answer_recall"


def test_normalize_channel_key_aliases_room_id() -> None:
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=_FakeHubClient(),
        callsyne_client=_FakeCallSyneClient(),
        bus=_FakeBus(),
    )
    for key_field in ("channel_key", "channelKey"):
        msg = svc.normalize_callsyne_message(
            {
                key_field: "world-general",
                "message_id": "m-ch",
                "sender_id": "peer-9",
                "text": "ping",
            }
        )
        assert msg.room_id == "world-general"


def test_normalize_channel_key_in_metadata() -> None:
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=_FakeHubClient(),
        callsyne_client=_FakeCallSyneClient(),
        bus=_FakeBus(),
    )
    msg = svc.normalize_callsyne_message(
        {
            "message_id": "m2",
            "sender_id": "p1",
            "text": "hi",
            "metadata": {"channel_key": "zip-123-social"},
        }
    )
    assert msg.room_id == "zip-123-social"


def test_callsyne_bridge_post_body_minimal_and_media_hint() -> None:
    req = ExternalRoomPostRequestV1(
        platform="callsyne",
        room_id="room-1",
        text="hello",
        reply_to_message_id="prev-1",
        thread_id="t-a",
        correlation_id="corr-x",
        metadata={"gif_intent": "laugh_with", "media_hint": {"kind": "gif", "query": "lol"}},
    )
    body = _callsyne_bridge_post_body(req)
    assert body == {
        "room_id": "room-1",
        "text": "hello",
        "thread_id": "t-a",
        "media_hint": {"kind": "gif", "query": "lol"},
        "metadata": {"gif_intent": "laugh_with", "correlation_id": "corr-x"},
    }

    bare = _callsyne_bridge_post_body(
        ExternalRoomPostRequestV1(platform="callsyne", room_id="r", text="x", metadata={})
    )
    assert bare == {"room_id": "r", "text": "x"}


def test_callsyne_bridge_post_body_numeric_reply_id_only() -> None:
    numeric = _callsyne_bridge_post_body(
        ExternalRoomPostRequestV1(
            platform="callsyne",
            room_id="room-1",
            text="hello",
            reply_to_message_id="12345",
        )
    )
    assert numeric["reply_to_message_id"] == 12345
    non_numeric = _callsyne_bridge_post_body(
        ExternalRoomPostRequestV1(
            platform="callsyne",
            room_id="room-1",
            text="hello",
            reply_to_message_id="live-public-test-xyz",
        )
    )
    assert "reply_to_message_id" not in non_numeric


def test_delivery_failure_returns_structured_200_payload_when_enabled() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="bridge message")
    callsyne = _FailingCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_RETURN_2XX_ON_DELIVERY_FAILURE=True),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-delivery-fail")))
    assert result["status"] == "delivery_failed"
    assert result["message_id"] == "msg-delivery-fail"


def test_delivery_failure_raises_when_disabled() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="bridge message")
    callsyne = _FailingCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_RETURN_2XX_ON_DELIVERY_FAILURE=False),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    try:
        asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-delivery-fail-raise")))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_polling_fallback_dedupes_and_skips_self_messages() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="poll reply")
    callsyne = _FakeCallSyneClient()
    callsyne.fetch_payloads = [[
        {"id": "9", "room_id": "world", "sender_id": "peer-1", "sender_name": "CallSyne Peer", "text": "Orion ping"},
        {"id": "10", "room_id": "world", "sender_id": "orion-room-bot", "sender_name": "Oríon", "text": "self"},
        {"id": "11", "room_id": "world", "sender_id": "peer-2", "sender_name": "Peer Two", "text": "@Orion answer this"},
    ]]
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_CALLSYNE_POLL_ENABLED=True,
            SOCIAL_BRIDGE_CALLSYNE_POLL_ROOM_ID="world",
            SOCIAL_BRIDGE_CALLSYNE_POLL_SINCE_MESSAGE_ID="9",
            SOCIAL_BRIDGE_CALLSYNE_POLL_SKIP_SELF=True,
            SOCIAL_BRIDGE_CALLSYNE_POLL_PATH="/api/bridge/messages/read",
        ),
        hub_client=hub,
        callsyne_client=callsyne,
        bus=bus,
    )

    asyncio.run(svc.poll_callsyne_once())

    assert len(hub.calls) == 1
    assert len(callsyne.posts) == 1


def test_polling_fetch_failure_does_not_raise() -> None:
    class _ErroringPollClient(_FakeCallSyneClient):
        async def fetch_recent_messages(self, *, path: str, room_id: str, limit: int, since_message_id: str | None = None):
            raise RuntimeError("fetch failure")

    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_CALLSYNE_POLL_ENABLED=True,
            SOCIAL_BRIDGE_CALLSYNE_POLL_PATH="/api/bridge/messages/read",
        ),
        hub_client=_FakeHubClient(),
        callsyne_client=_ErroringPollClient(),
        bus=_FakeBus(),
    )

    asyncio.run(svc.poll_callsyne_once())


def test_polling_is_hard_blocked_for_post_only_bridge_path() -> None:
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_CALLSYNE_POLL_ENABLED=True,
            SOCIAL_BRIDGE_CALLSYNE_POLL_PATH="/api/bridge/messages",
        ),
        hub_client=_FakeHubClient(),
        callsyne_client=callsyne,
        bus=_FakeBus(),
    )

    asyncio.run(svc.start())

    assert svc._poll_task is None
    assert callsyne.fetch_calls == []


def test_polling_allows_non_post_only_paths() -> None:
    callsyne = _FakeCallSyneClient()
    callsyne.fetch_payloads = [[]]
    svc = SocialRoomBridgeService(
        settings=_settings(
            SOCIAL_BRIDGE_CALLSYNE_POLL_ENABLED=True,
            SOCIAL_BRIDGE_CALLSYNE_POLL_PATH="/api/bridge/messages/read",
        ),
        hub_client=_FakeHubClient(),
        callsyne_client=callsyne,
        bus=_FakeBus(),
    )

    asyncio.run(svc.poll_callsyne_once())

    assert len(callsyne.fetch_calls) == 1
    assert callsyne.fetch_calls[0]["path"] == "/api/bridge/messages/read"


def test_settings_defaults_and_schema_registration() -> None:
    settings = _settings()

    assert settings.social_bridge_platform == "callsyne"
    assert settings.social_bridge_autonomy_mode == "responsive"
    assert settings.social_bridge_max_consecutive_orion_turns == 2
    assert settings.room_intake_channel == "orion:bridge:social:room:intake"
    assert settings.room_repair_signal_channel == "orion:social:repair:signal"
    assert settings.room_repair_decision_channel == "orion:social:repair:decision"
    assert settings.room_epistemic_signal_channel == "orion:social:epistemic:signal"
    assert settings.room_epistemic_decision_channel == "orion:social:epistemic:decision"
    assert settings.room_turn_policy_channel == "orion:social:turn-policy"
    assert settings.callsyne_post_path_template == "/api/bridge/messages"
    assert _REGISTRY["CallSyneRoomMessageV1"].__name__ == "CallSyneRoomMessageV1"
    assert _REGISTRY["ExternalRoomPostResultV1"].__name__ == "ExternalRoomPostResultV1"
    assert _REGISTRY["SocialEpistemicSignalV1"].__name__ == "SocialEpistemicSignalV1"
    assert _REGISTRY["SocialEpistemicDecisionV1"].__name__ == "SocialEpistemicDecisionV1"
    assert _REGISTRY["SocialRepairSignalV1"].__name__ == "SocialRepairSignalV1"
    assert _REGISTRY["SocialRepairDecisionV1"].__name__ == "SocialRepairDecisionV1"
    assert _REGISTRY["SocialGifPolicyDecisionV1"].__name__ == "SocialGifPolicyDecisionV1"
    assert _REGISTRY["SocialGifIntentV1"].__name__ == "SocialGifIntentV1"
    assert _REGISTRY["SocialGifUsageStateV1"].__name__ == "SocialGifUsageStateV1"
