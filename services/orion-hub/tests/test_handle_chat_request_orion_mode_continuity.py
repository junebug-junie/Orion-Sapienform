from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def test_handle_chat_request_orion_mode_builds_continuity_messages(monkeypatch) -> None:
    """Regression for the parity gap: the mode=="orion" HTTP branch of /api/chat
    previously called execute_unified_turn with no continuity_messages at all,
    unlike the sibling non-orion branch a few lines below (which calls
    build_continuity_messages(history=user_messages, ...)) and unlike the
    WebSocket path (the real Hub UI), which already built it correctly. This
    was inert while turn_orchestrator dropped continuity_messages on the
    floor, but now that HarnessRunRequestV1.recent_turns threads it into the
    harness prompt, an HTTP-driven orion-mode turn must build it the same way
    the sibling branch does or it silently gets no history at all.

    NOTE: scripts.api_routes/scripts.main must be imported fresh inside this
    test body -- conftest.py's autouse _hub_service_isolation fixture clears
    scripts.*/app.* from sys.modules before every test.
    """
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    from orion.hub import turn_orchestrator

    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "harness_step_relay", None, raising=False)
    monkeypatch.setattr(hub_main, "rpc_bus", None, raising=False)

    captured_kwargs: dict = {}

    async def _fake_execute_unified_turn(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            {
                "type": "final",
                "correlation_id": "corr-continuity",
                "llm_response": "a fully reflected answer",
                "finalize_ran": True,
            }
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    payload = {
        "mode": "orion",
        "messages": [
            {"role": "user", "content": "what's the weather like?"},
            {"role": "assistant", "content": "I don't have live weather access."},
            {"role": "user", "content": "ok what about tomorrow?"},
        ],
    }
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-continuity", no_write=True))

    assert out["type"] == "final"
    continuity_messages = captured_kwargs.get("continuity_messages")
    assert continuity_messages, "orion-mode HTTP branch must build continuity_messages, not pass None"
    assert continuity_messages == [
        {"role": "user", "content": "what's the weather like?"},
        {"role": "assistant", "content": "I don't have live weather access."},
        {"role": "user", "content": "ok what about tomorrow?"},
    ]
