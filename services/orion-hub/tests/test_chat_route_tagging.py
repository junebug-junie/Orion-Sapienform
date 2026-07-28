"""Every branch of handle_chat_request must stamp an explicit chat_route tag.

Ground truth: the four branches (classic PlanRunner RPC, unified-turn harness,
agent-claude, context-exec agent lane) were previously distinguished only by
which response fields happened to be populated -- no explicit tag. This is
the first slice of Runtime-Trace-Nexus/Grammar-Atlas fusion work: a later
correlation-keyed trace lookup needs chat_route to tell "no stance row
because this route never produces one" from "no stance row because
persistence broke."
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.hub.chat_route import (
    CHAT_ROUTE_AGENT_CLAUDE,
    CHAT_ROUTE_CLASSIC_PLANRUNNER,
    CHAT_ROUTE_CONTEXT_EXEC_AGENT,
    CHAT_ROUTE_UNIFIED_TURN_HARNESS,
)
from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from scripts import api_routes
from scripts.api_routes import handle_chat_request


class _ClassicClient:
    async def chat(self, req, correlation_id=None):
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="ok",
            correlation_id=correlation_id or "corr-classic",
        )
        return CortexChatResult(cortex_result=result, final_text="ok")


def test_classic_planrunner_path_tags_chat_route() -> None:
    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_ClassicClient(), payload, "sid-classic", no_write=True))
    assert out["chat_route"] == CHAT_ROUTE_CLASSIC_PLANRUNNER


def test_unified_turn_path_tags_chat_route(monkeypatch) -> None:
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
                "correlation_id": "corr-unified",
                "llm_response": "a fully reflected answer",
                "finalize_ran": True,
            }
        ]

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_execute_unified_turn)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(object(), payload, "sid-unified", no_write=True))
    assert out["chat_route"] == CHAT_ROUTE_UNIFIED_TURN_HARNESS


def test_unified_turn_config_disabled_error_still_tags_chat_route(monkeypatch) -> None:
    monkeypatch.setattr(api_routes.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(api_routes.settings, "ORION_HARNESS_GOVERNOR_ENABLED", False, raising=False)

    payload = {"mode": "orion", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(object(), payload, "sid-unified-disabled", no_write=True))
    assert out["chat_route"] == CHAT_ROUTE_UNIFIED_TURN_HARNESS
    assert out["error"] == "harness_governor_disabled"


def test_agent_claude_path_tags_chat_route(monkeypatch) -> None:
    async def _fake_run_agent_claude_http(*, prompt, fcc_model_label, correlation_id):
        return {
            "mode": "agent-claude",
            "correlation_id": correlation_id,
            "llm_response": "claude says hi",
            "text": "claude says hi",
        }

    monkeypatch.setattr(api_routes, "_run_agent_claude_http", _fake_run_agent_claude_http)
    monkeypatch.setattr(
        api_routes, "catalog_from_settings", lambda **_kwargs: {"default_label": "MODEL_SONNET"}
    )

    payload = {"mode": "agent-claude", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(object(), payload, "sid-agent-claude", no_write=True))
    assert out["chat_route"] == CHAT_ROUTE_AGENT_CLAUDE


def test_context_exec_agent_lane_tags_chat_route(monkeypatch) -> None:
    monkeypatch.setattr(api_routes, "should_use_context_exec_agent_lane", lambda req: True)

    async def _fake_run_hub_agent_via_context_exec(*, req, prompt, correlation_id, route_debug):
        return {
            "llm_response": "context-exec answer",
            "agent_trace": None,
            "raw": {},
            "routing_debug": route_debug,
            "context_exec_run": {"status": "ok"},
            "operator_summary": None,
        }

    monkeypatch.setattr(
        api_routes, "run_hub_agent_via_context_exec", _fake_run_hub_agent_via_context_exec
    )

    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(object(), payload, "sid-context-exec", no_write=True))
    assert out["chat_route"] == CHAT_ROUTE_CONTEXT_EXEC_AGENT
    assert out["context_exec_lane"] is True
