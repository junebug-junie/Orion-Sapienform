from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app.gif_proxy import (  # noqa: E402
    build_social_gif_proxy_context,
    extract_social_gif_observed_signal,
    interpret_social_gif_proxy,
)
from app.service import SocialRoomBridgeService  # noqa: E402
from app.settings import Settings  # noqa: E402


class _FakeHubClient:
    def __init__(self, reply_text: str = "Absolutely.") -> None:
        self.calls = []
        self.reply_text = reply_text

    async def chat(self, *, payload, session_id: str):
        self.calls.append((payload, session_id))
        return {"text": self.reply_text, "correlation_id": "corr-room-gif-proxy"}


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
        "thread_id": "thread-gif-proxy",
        "message_id": "msg-gif-proxy-1",
        "sender_id": "peer-1",
        "sender_name": "CallSyne Peer",
        "sender_kind": "peer_ai",
        "text": "we did it lol",
        "mentions_orion": True,
        "created_at": "2026-03-23T10:00:00+00:00",
        "metadata": {
            "transport": "callsyne-webhook",
            "peer_used_gif": "true",
            "supports_media_hints": "true",
            "gif_provider": "giphy",
            "gif_query": "celebration reaction gif",
            "gif_title": "tiny victory celebration",
            "gif_tags": "celebrate,win,party",
        },
    }
    payload.update(overrides)
    return payload


def _summary(**room_overrides):
    return {
        "room": {
            "room_id": "room-alpha",
            "current_thread_key": "callsyne:room-alpha:thread:thread-gif-proxy",
            "current_thread_summary": "A light celebratory thread.",
            "room_tone_summary": "Warm and playful.",
            "active_participants": ["CallSyne Peer", "Oríon"],
            "active_threads": [
                {
                    "thread_key": "callsyne:room-alpha:thread:thread-gif-proxy",
                    "platform": "callsyne",
                    "room_id": "room-alpha",
                    "thread_id": "thread-gif-proxy",
                    "active_participants": ["CallSyne Peer", "Oríon"],
                    "audience_scope": "peer",
                    "last_speaker": "CallSyne Peer",
                    "open_question": False,
                    "handoff_flag": False,
                    "orion_involved": True,
                    "thread_summary": "A light celebratory thread.",
                    "last_activity_at": "2026-03-23T10:00:00+00:00",
                    "expires_at": "2099-03-23T16:00:00+00:00",
                    "metadata": {},
                }
            ],
            **room_overrides,
        }
    }


def _service(summary=None, *, reply_text: str = "Absolutely.") -> SocialRoomBridgeService:
    return SocialRoomBridgeService(
        settings=_settings(SOCIAL_BRIDGE_COOLDOWN_SEC=0, SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS=5),
        hub_client=_FakeHubClient(reply_text=reply_text),
        callsyne_client=_FakeCallSyneClient(),
        social_memory_client=_FakeSocialMemoryClient(summary or _summary()),
        bus=_FakeBus(),
    )


def test_clear_metadata_yields_conservative_reaction_class() -> None:
    svc = _service()
    message = svc.normalize_callsyne_message(_payload())
    observed = extract_social_gif_observed_signal(message)
    assert observed is not None
    proxy = build_social_gif_proxy_context(message=message, social_memory=_summary(), observed_signal=observed)
    decision = svc._policy_decision(message, social_memory=_summary())
    interpretation = interpret_social_gif_proxy(
        message=message,
        turn_policy=decision,
        social_memory=_summary(),
        observed_signal=observed,
        proxy_context=proxy,
    )

    assert interpretation.reaction_class == "celebrate"
    assert interpretation.confidence_level in {"low", "medium"}
    assert interpretation.cue_disposition in {"used", "softened"}


def test_weak_metadata_stays_unknown_and_high_ambiguity() -> None:
    svc = _service()
    message = svc.normalize_callsyne_message(
        _payload(
            text="",
            metadata={
                "transport": "callsyne-webhook",
                "peer_used_gif": "true",
                "supports_media_hints": "true",
                "gif_filename": "reaction.gif",
            },
        )
    )
    observed = extract_social_gif_observed_signal(message)
    assert observed is not None
    proxy = build_social_gif_proxy_context(message=message, social_memory=_summary(), observed_signal=observed)
    interpretation = interpret_social_gif_proxy(
        message=message,
        turn_policy=svc._policy_decision(message, social_memory=_summary()),
        social_memory=_summary(),
        observed_signal=observed,
        proxy_context=proxy,
    )

    assert interpretation.reaction_class == "unknown"
    assert interpretation.confidence_level == "low"
    assert interpretation.ambiguity_level == "high"


def test_surrounding_text_can_disambiguate_laughter() -> None:
    svc = _service()
    message = svc.normalize_callsyne_message(
        _payload(
            text="lol this absolutely got me",
            metadata={
                "transport": "callsyne-webhook",
                "peer_used_gif": "true",
                "supports_media_hints": "true",
                "gif_provider": "giphy",
            },
        )
    )
    observed = extract_social_gif_observed_signal(message)
    assert observed is not None
    proxy = build_social_gif_proxy_context(message=message, social_memory=_summary(), observed_signal=observed)
    interpretation = interpret_social_gif_proxy(
        message=message,
        turn_policy=svc._policy_decision(message, social_memory=_summary()),
        social_memory=_summary(),
        observed_signal=observed,
        proxy_context=proxy,
    )

    assert interpretation.reaction_class == "laugh_with"
    assert interpretation.confidence_level == "low"


def test_contested_context_keeps_gif_proxy_secondary() -> None:
    summary = _summary(
        claim_divergence_signals=[{"normalized_claim_key": "private thing"}],
        claim_consensus_states=[{"consensus_state": "contested"}],
    )
    svc = _service(summary)
    message = svc.normalize_callsyne_message(_payload(text="what do you mean exactly?"))
    observed = extract_social_gif_observed_signal(message)
    assert observed is not None
    proxy = build_social_gif_proxy_context(message=message, social_memory=summary, observed_signal=observed)
    interpretation = interpret_social_gif_proxy(
        message=message,
        turn_policy=svc._policy_decision(message, social_memory=summary),
        social_memory=summary,
        observed_signal=observed,
        proxy_context=proxy,
    )

    assert interpretation.cue_disposition == "ignored"
    assert "stronger_live_cues_override_gif_proxy" in interpretation.reasons


def test_blocked_private_proxy_text_is_not_widened() -> None:
    svc = _service()
    message = svc.normalize_callsyne_message(
        _payload(
            metadata={
                "transport": "callsyne-webhook",
                "peer_used_gif": "true",
                "supports_media_hints": "true",
                "gif_caption": "private sealed note",
                "gif_query": "celebration reaction gif",
            },
        )
    )
    observed = extract_social_gif_observed_signal(message)
    assert observed is not None
    proxy = build_social_gif_proxy_context(message=message, social_memory=_summary(), observed_signal=observed)

    assert observed.caption_text == ""
    assert all("private" not in fragment.lower() for fragment in proxy.proxy_text_fragments)


def test_service_emits_proxy_metadata_without_enabling_tools() -> None:
    svc = _service()

    result = asyncio.run(svc.process_callsyne_message(_payload()))

    assert result["status"] == "ok"
    assert result["gif_interpretation"]["reaction_class"] == "celebrate"
    assert svc.hub_client.calls[0][0]["social_gif_observed_signal"]["media_present"] is True
    assert svc.hub_client.calls[0][0]["social_gif_interpretation"]["reaction_class"] == "celebrate"
    assert svc.hub_client.calls[0][0]["options"]["tool_execution_policy"] == "none"
    assert svc.hub_client.calls[0][0]["options"]["action_execution_policy"] == "none"
