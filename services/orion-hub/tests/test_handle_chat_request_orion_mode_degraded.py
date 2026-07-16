from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def test_handle_chat_request_orion_mode_surfaces_degraded_reason(monkeypatch) -> None:
    """The mode=="orion" HTTP branch collapses execute_unified_turn's frame list down to
    a single dict (frames[-1]) for its JSON response. Without the fix, a degraded turn's
    turn_degraded frame (delivered fine over the websocket, which sends every frame) would
    silently vanish for this HTTP call path -- the caller would see finalize_ran=False with
    no explanation. This confirms the reason survives via finalize_degraded_reason instead.

    NOTE: scripts.api_routes/scripts.main must be imported fresh inside this test body, not
    at module top level -- conftest.py's autouse _hub_service_isolation fixture clears
    scripts.*/app.* from sys.modules before every test, so a module-level import here would
    bind to a stale module/settings singleton the monkeypatches below never touch.
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

    async def _fake_execute_unified_turn(**_kwargs):
        return [
            {
                "type": "turn_degraded",
                "correlation_id": "corr-degraded",
                "reason": "substrate appraisal unavailable (RPC timeout)",
            },
            {
                "type": "final",
                "correlation_id": "corr-degraded",
                "llm_response": "internal draft delivered as final",
                "finalize_ran": False,
            },
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-degraded", no_write=True))

    assert out["type"] == "final"
    assert out["llm_response"] == "internal draft delivered as final"
    assert out["finalize_ran"] is False
    assert out["finalize_degraded_reason"] == "substrate appraisal unavailable (RPC timeout)"


def test_handle_chat_request_orion_mode_normal_success_has_no_degraded_reason(monkeypatch) -> None:
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

    async def _fake_execute_unified_turn(**_kwargs):
        return [
            {
                "type": "final",
                "correlation_id": "corr-normal",
                "llm_response": "a fully reflected answer",
                "finalize_ran": True,
            }
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(api_routes.handle_chat_request(object(), payload, "sid-orion-normal", no_write=True))

    assert out["type"] == "final"
    assert out["finalize_ran"] is True
    assert "finalize_degraded_reason" not in out
