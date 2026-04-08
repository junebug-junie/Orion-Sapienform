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

from app.gif_policy import evaluate_social_gif_policy, reconcile_gif_policy_with_reply_text
from app.service import SocialRoomBridgeService
from app.settings import Settings
from orion.schemas.social_gif import SocialGifUsageStateV1


class _FakeHubClient:
    def __init__(self, reply_text: str = "Absolutely.") -> None:
        self.calls = []
        self.reply_text = reply_text

    async def chat(self, *, payload, session_id: str):
        self.calls.append((payload, session_id))
        return {"text": self.reply_text, "correlation_id": "corr-room-gif"}


class _FakeCallSyneClient:
    def __init__(self) -> None:
        self.posts = []

    async def post_message(self, request):
        self.posts.append(request)
        return {"message_id": f"callsyne-out-{len(self.posts)}", "status": "posted"}


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

    async def get_summary(self, *, platform: str, room_id: str, participant_id: str | None):
        return self.summary


def _settings(**overrides) -> Settings:
    base = {
        "ORION_BUS_ENABLED": False,
        "SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS": "orion-room-bot",
    }
    base.update(overrides)
    return Settings(**base)


def _payload(**overrides):
    payload = {
        "room_id": "room-alpha",
        "thread_id": "thread-gif",
        "message_id": "msg-gif-1",
        "sender_id": "peer-1",
        "sender_name": "CallSyne Peer",
        "sender_kind": "peer_ai",
        "text": "Oríon lol yes exactly",
        "mentions_orion": True,
        "created_at": "2026-03-22T10:00:00+00:00",
        "metadata": {
            "transport": "callsyne-webhook",
            "peer_used_gif": "true",
            "supports_media_hints": "true",
        },
    }
    payload.update(overrides)
    return payload


def _eligible_summary(room_overrides: dict | None = None, **extra_room_overrides):
    overrides = dict(room_overrides or {})
    overrides.update(extra_room_overrides)
    return {
        "room": {
            "room_id": "room-alpha",
            "current_thread_key": "callsyne:room-alpha:thread:thread-gif",
            "current_thread_summary": "Light, playful agreement thread.",
            "room_tone_summary": "Warm and playful.",
            "active_participants": ["CallSyne Peer", "Oríon"],
            "active_threads": [
                {
                    "thread_key": "callsyne:room-alpha:thread:thread-gif",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_id": "thread-gif",
                    "active_participants": ["CallSyne Peer", "Oríon"],
                    "audience_scope": "peer",
                    "last_speaker": "CallSyne Peer",
                    "open_question": False,
                    "handoff_flag": False,
                    "orion_involved": True,
                    "thread_summary": "Light, playful agreement thread.",
                    "last_activity_at": "2026-03-22T10:00:00+00:00",
                    "expires_at": "2099-03-22T16:00:00+00:00",
                    "metadata": {},
                }
            ],
            "gif_usage_state": {
                "platform": "callsyne",
                "room_id": "room-alpha",
                "thread_key": "callsyne:room-alpha:thread:thread-gif",
                "consecutive_gif_turns": 0,
                "turns_since_last_orion_gif": 3,
                "recent_gif_density": 0.0,
                "recent_gif_turn_count": 0,
                "recent_turn_window_size": 10,
                "orion_turn_count": 3,
                "recent_turn_was_gif": [False, False, False],
                "recent_intent_kinds": [],
                "recent_target_participant_ids": [],
                "recent_target_participant_names": [],
                "metadata": {"source": "test"},
            },
            **overrides,
        },
        "room_ritual": {
            "culture_summary": "Warm, playful, lightly expressive.",
        },
    }


def _policy_and_message(*, text: str, summary: dict | None = None, usage_state: SocialGifUsageStateV1 | None = None, metadata: dict | None = None):
    active_summary = summary or _eligible_summary()
    resolved_usage_state = usage_state
    if resolved_usage_state is None:
        raw_usage = ((active_summary.get("room") or {}).get("gif_usage_state") if isinstance(active_summary, dict) else None) or {}
        if raw_usage:
            resolved_usage_state = SocialGifUsageStateV1.model_validate(raw_usage)
    svc = SocialRoomBridgeService(
        settings=_settings(),
        hub_client=_FakeHubClient(),
        callsyne_client=_FakeCallSyneClient(),
        social_memory_client=_FakeSocialMemoryClient(active_summary),
        bus=_FakeBus(),
    )
    message = svc.normalize_callsyne_message(_payload(text=text, metadata=metadata or _payload()["metadata"]))
    turn_policy = svc._policy_decision(message, social_memory=active_summary)
    gif_policy = evaluate_social_gif_policy(
        message=message,
        turn_policy=turn_policy,
        social_memory=active_summary,
        usage_state=resolved_usage_state,
    )
    return turn_policy, gif_policy


def test_light_affiliative_exchange_allows_text_plus_gif() -> None:
    turn_policy, gif_policy = _policy_and_message(text="Oríon lol yes exactly")

    assert turn_policy.decision == "reply"
    assert gif_policy.decision_kind == "text_plus_gif"
    assert gif_policy.gif_allowed is True
    assert gif_policy.intent_kind in {"laugh_with", "dramatic_agreement"}


def test_playful_room_can_allow_gif_even_without_peer_gif_when_orion_has_been_text_only() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon yes exactly, that lands",
        metadata={"transport": "callsyne-webhook", "peer_used_gif": "false", "supports_media_hints": "true"},
        summary={
            **_eligible_summary(),
            "peer_style": {"playfulness_tendency": 0.82},
            "room_ritual": {
                "culture_summary": "Warm, playful, lightly expressive.",
                "room_tone_summary": "Playful and warm.",
                "confidence": 0.84,
                "evidence_count": 4,
            },
        },
    )

    assert gif_policy.decision_kind == "text_plus_gif"
    assert gif_policy.gif_allowed is True
    assert "fresh_room_ritual_supports_playfulness" in gif_policy.reasons


def test_cooldown_blocks_too_soon_gif() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon lol yes exactly",
        usage_state=SocialGifUsageStateV1(
            platform="callsyne",
            room_id="room-alpha",
            thread_key="callsyne:room-alpha:thread:thread-gif",
            consecutive_gif_turns=0,
            turns_since_last_orion_gif=1,
            recent_gif_density=0.1,
            recent_gif_turn_count=1,
            recent_turn_window_size=10,
            orion_turn_count=5,
            recent_turn_was_gif=[True, False, False],
            recent_intent_kinds=["laugh_with"],
            recent_target_participant_ids=["peer-1"],
            recent_target_participant_names=["CallSyne Peer"],
        ),
    )

    assert gif_policy.decision_kind == "text_only"
    assert "gif_cooldown_active" in gif_policy.reasons


def test_rolling_density_cap_blocks_gif() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon lol yes exactly",
        usage_state=SocialGifUsageStateV1(
            platform="callsyne",
            room_id="room-alpha",
            thread_key="callsyne:room-alpha:thread:thread-gif",
            consecutive_gif_turns=0,
            turns_since_last_orion_gif=4,
            recent_gif_density=0.3,
            recent_gif_turn_count=2,
            recent_turn_window_size=10,
            orion_turn_count=7,
            recent_turn_was_gif=[True, False, False, True, False, False, False],
            recent_intent_kinds=["laugh_with", "celebrate"],
            recent_target_participant_ids=["peer-1", "peer-2"],
            recent_target_participant_names=["CallSyne Peer", "Another Peer"],
        ),
    )

    assert gif_policy.decision_kind == "text_only"
    assert "gif_density_cap_reached" in gif_policy.reasons


def test_medium_chaos_room_tightens_density_and_scope() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon lol yes exactly",
        summary=_eligible_summary(
            room_overrides={
                "active_participants": ["CallSyne Peer", "Oríon", "Cadence"],
                "active_threads": [
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-gif",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-gif",
                        "active_participants": ["CallSyne Peer", "Oríon"],
                        "audience_scope": "peer",
                        "last_speaker": "CallSyne Peer",
                        "open_question": False,
                        "handoff_flag": False,
                        "orion_involved": True,
                        "thread_summary": "Light, playful agreement thread.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                    {
                        "thread_key": "callsyne:room-alpha:thread:thread-side",
                        "platform": "callsyne",
                        "room_id": "room-alpha",
                        "thread_id": "thread-side",
                        "active_participants": ["Cadence", "Mika"],
                        "audience_scope": "room",
                        "last_speaker": "Cadence",
                        "open_question": True,
                        "handoff_flag": True,
                        "orion_involved": False,
                        "thread_summary": "Side thread still open.",
                        "last_activity_at": "2026-03-22T10:00:00+00:00",
                        "expires_at": "2099-03-22T16:00:00+00:00",
                        "metadata": {},
                    },
                ],
                "gif_usage_state": {
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_key": "callsyne:room-alpha:thread:thread-gif",
                    "consecutive_gif_turns": 0,
                    "turns_since_last_orion_gif": 4,
                    "recent_gif_density": 0.16,
                    "recent_gif_turn_count": 1,
                    "recent_turn_window_size": 10,
                    "orion_turn_count": 6,
                    "recent_turn_was_gif": [True, False, False, False, False, False],
                    "recent_intent_kinds": ["laugh_with"],
                    "recent_target_participant_ids": ["peer-1"],
                    "recent_target_participant_names": ["CallSyne Peer"],
                    "metadata": {"source": "test"},
                },
            }
        ),
    )

    assert gif_policy.decision_kind == "text_only"
    assert "gif_density_cap_reached" in gif_policy.reasons
    assert "medium_chaos_downrank" not in gif_policy.reasons  # density block should fire first


def test_repair_clarification_and_artifact_boundary_force_text_only() -> None:
    repair_summary = _eligible_summary(room_overrides={"shared_artifact_status": "deferred"})
    turn_policy, gif_policy = _policy_and_message(text="Oríon, that was for Cadence, not you.", summary=repair_summary)

    assert turn_policy.repair_signal is not None
    assert gif_policy.decision_kind == "text_only"
    assert {"repair_active_turn", "shared_artifact_or_scope_boundary"} & set(gif_policy.reasons)

    clarify_summary = _eligible_summary(
        deliberation_decision={"decision_kind": "ask_clarifying_question"},
        clarifying_question={"question_text": "Which part do you mean?"},
    )
    turn_policy, gif_policy = _policy_and_message(text="Oríon, which one do you mean?", summary=clarify_summary)

    assert turn_policy.decision == "ask_follow_up"
    assert gif_policy.decision_kind == "text_only"
    assert "clarification_turn_text_only" in gif_policy.reasons


def test_contested_or_epistemic_sensitive_turn_stays_text_only() -> None:
    summary = _eligible_summary(
        claim_divergence_signals=[{"normalized_claim_key": "pacing versus thread split"}],
        claim_consensus_states=[{"consensus_state": "contested"}],
    )
    turn_policy, gif_policy = _policy_and_message(
        text="Oríon, what do you remember about the private thing?",
        summary=summary,
        metadata={"transport": "callsyne-webhook", "peer_used_gif": "true", "supports_media_hints": "true"},
    )

    assert turn_policy.epistemic_signal is not None
    assert gif_policy.decision_kind == "text_only"
    assert "epistemically_sensitive_or_contested" in gif_policy.reasons


def test_repeated_intent_suppression_blocks_same_joke_pattern() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon lol yes exactly",
        usage_state=SocialGifUsageStateV1(
            platform="callsyne",
            room_id="room-alpha",
            thread_key="callsyne:room-alpha:thread:thread-gif",
            consecutive_gif_turns=0,
            turns_since_last_orion_gif=4,
            recent_gif_density=0.1,
            recent_gif_turn_count=1,
            recent_turn_window_size=10,
            orion_turn_count=6,
            recent_turn_was_gif=[False, False, True, False, False],
            recent_intent_kinds=["laugh_with"],
            recent_target_participant_ids=["peer-1"],
            recent_target_participant_names=["CallSyne Peer"],
            last_intent_kind="laugh_with",
            last_target_participant_id="peer-1",
            last_target_participant_name="CallSyne Peer",
        ),
    )

    assert gif_policy.decision_kind == "text_only"
    assert {"repeated_intent_suppressed", "gif_intent_loop_detected"} & set(gif_policy.reasons)


def test_intent_loop_detection_blocks_repeating_same_reaction_to_same_peer() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon lol yes exactly",
        usage_state=SocialGifUsageStateV1(
            platform="callsyne",
            room_id="room-alpha",
            thread_key="callsyne:room-alpha:thread:thread-gif",
            consecutive_gif_turns=0,
            turns_since_last_orion_gif=5,
            recent_gif_density=0.1,
            recent_gif_turn_count=1,
            recent_turn_window_size=10,
            orion_turn_count=8,
            recent_turn_was_gif=[False, True, False, False],
            recent_intent_kinds=["laugh_with", "celebrate", "laugh_with"],
            recent_target_participant_ids=["peer-1", "peer-2", "peer-1"],
            recent_target_participant_names=["CallSyne Peer", "Another Peer", "CallSyne Peer"],
            last_intent_kind="laugh_with",
            last_target_participant_id="peer-1",
            last_target_participant_name="CallSyne Peer",
        ),
    )

    assert gif_policy.decision_kind == "text_only"
    assert "gif_intent_loop_detected" in gif_policy.reasons


def test_stale_ritual_hint_does_not_push_gif_over_fresh_caution_context() -> None:
    _, gif_policy = _policy_and_message(
        text="Oríon yes exactly",
        metadata={"transport": "callsyne-webhook", "peer_used_gif": "false", "supports_media_hints": "true"},
        summary={
            **_eligible_summary(),
            "room": {
                **_eligible_summary()["room"],
                "memory_freshness": [
                    {
                        "artifact_kind": "room_ritual",
                        "freshness_state": "stale",
                        "rationale": "Older room ritual read is fading.",
                    },
                    {
                        "artifact_kind": "claim_consensus",
                        "freshness_state": "refresh_needed",
                        "rationale": "Consensus needs refresh before style cues.",
                    },
                ],
            },
            "context_candidates": [
                {
                    "candidate_kind": "ritual",
                    "inclusion_decision": "soften",
                    "freshness_band": "stale",
                },
                {
                    "candidate_kind": "freshness_hint",
                    "inclusion_decision": "include",
                    "freshness_band": "refresh_needed",
                },
            ],
            "room_ritual": {
                "culture_summary": "Warm, playful, lightly expressive.",
                "confidence": 0.8,
                "evidence_count": 3,
            },
        },
    )

    assert gif_policy.decision_kind == "text_only"
    assert "stale_ritual_does_not_push_gif" in gif_policy.reasons
    assert "fresh_caution_context_prefers_text" in gif_policy.reasons


def test_transport_degrades_cleanly_without_media_support() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Absolutely.")
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_COOLDOWN_SEC=0, SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS=5),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=_FakeSocialMemoryClient(_eligible_summary()),
        bus=bus,
    )

    result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-gif-degrade",
                metadata={"transport": "callsyne-webhook", "peer_used_gif": "true", "supports_media_hints": "false"},
            )
        )
    )

    assert result["status"] == "ok"
    assert result["gif_policy"]["decision_kind"] == "text_only"
    assert "transport_degraded_to_text_only" in result["gif_policy"]["reasons"]
    assert "gif_intent" not in callsyne.posts[0].metadata
    assert hub.calls[0][0]["options"]["tool_execution_policy"] == "none"
    assert hub.calls[0][0]["options"]["action_execution_policy"] == "none"


def test_reply_text_redundancy_downgrades_allowed_gif_to_text_only() -> None:
    _, gif_policy = _policy_and_message(text="Oríon lol yes exactly")

    reconciled = reconcile_gif_policy_with_reply_text(policy=gif_policy, reply_text="lol yes exactly 😂😂")

    assert reconciled.decision_kind == "text_only"
    assert reconciled.gif_allowed is False
    assert "reply_text_already_carries_reaction" in reconciled.reasons
    assert reconciled.selected_intent is None


def test_service_blocks_successive_orion_gif_turns_and_private_turns() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="Absolutely.")
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_COOLDOWN_SEC=0, SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS=5),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=_FakeSocialMemoryClient(_eligible_summary()),
        bus=bus,
    )

    first = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-gif-1")))
    second = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-gif-2")))
    private_result = asyncio.run(
        svc.process_callsyne_message(
            _payload(
                message_id="msg-gif-3",
                text="Oríon lol yes exactly about the private thing",
                metadata={"transport": "callsyne-webhook", "peer_used_gif": "true", "supports_media_hints": "true"},
            )
        )
    )

    assert first["gif_policy"]["decision_kind"] == "text_plus_gif"
    assert callsyne.posts[0].metadata["gif_intent"] in {"laugh_with", "dramatic_agreement"}
    assert second["gif_policy"]["decision_kind"] == "text_only"
    assert "gif_intent" not in callsyne.posts[1].metadata
    assert private_result["gif_policy"]["decision_kind"] == "text_only"
    assert "private_or_blocked_material" in private_result["gif_policy"]["reasons"]
    assert "gif_intent" not in callsyne.posts[2].metadata


def test_service_drops_gif_transport_hint_when_reply_text_is_already_redundant() -> None:
    bus = _FakeBus()
    hub = _FakeHubClient(reply_text="lol yes exactly 😂😂")
    callsyne = _FakeCallSyneClient()
    svc = SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_COOLDOWN_SEC=0, SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS=5),
        hub_client=hub,
        callsyne_client=callsyne,
        social_memory_client=_FakeSocialMemoryClient(_eligible_summary()),
        bus=bus,
    )

    result = asyncio.run(svc.process_callsyne_message(_payload(message_id="msg-gif-redundant")))

    assert result["status"] == "ok"
    assert result["gif_policy"]["decision_kind"] == "text_only"
    assert "reply_text_already_carries_reaction" in result["gif_policy"]["reasons"]
    assert "gif_intent" not in callsyne.posts[0].metadata
