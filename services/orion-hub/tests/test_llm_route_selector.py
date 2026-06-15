"""Hub LLM route selector and context-exec agent lane tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts.context_exec_agent_bridge import (
    build_context_exec_request,
    normalize_llm_profile,
    should_use_context_exec_agent_lane,
)
from scripts.cortex_request_builder import build_chat_request
from orion.schemas.cortex.contracts import CortexChatRequest


def test_normalize_llm_profile_defaults_to_chat() -> None:
    assert normalize_llm_profile(None) == "chat"
    assert normalize_llm_profile("AGENT") == "agent"
    assert normalize_llm_profile("bogus") == "chat"


def test_build_chat_request_includes_llm_route_in_options() -> None:
    req, debug, _ = build_chat_request(
        payload={"mode": "brain", "llm_route": "quick"},
        session_id="s1",
        user_id="u1",
        trace_id="t1",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="test",
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.options.get("llm_route") == "quick"
    assert debug.get("llm_route") == "quick"


def test_build_chat_request_defaults_llm_route_to_chat() -> None:
    req, debug, _ = build_chat_request(
        payload={"mode": "brain"},
        session_id="s1",
        user_id=None,
        trace_id="t1",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="test",
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.options.get("llm_route") == "chat"
    assert debug.get("llm_route") == "chat"


def test_should_use_context_exec_agent_lane_when_enabled() -> None:
    req = CortexChatRequest(prompt="x", mode="agent")
    with patch("scripts.context_exec_agent_bridge.agent_lane_enabled", return_value=True):
        assert should_use_context_exec_agent_lane(req) is True
    with patch("scripts.context_exec_agent_bridge.agent_lane_enabled", return_value=False):
        assert should_use_context_exec_agent_lane(req) is False


def test_build_context_exec_request_sets_llm_profile() -> None:
    req = CortexChatRequest(prompt="trace autopsy corr abc", mode="agent", trace_id="corr-1")
    body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="chat")
    assert body.llm_profile == "chat"
    assert body.mode == "trace_autopsy"


@pytest.mark.asyncio
async def test_fetch_routes_normalizes_catalog() -> None:
    from scripts import llm_gateway_client as client

    payload = {
        "default_route": "chat",
        "routes": [
            {
                "id": "chat",
                "served_by": "atlas-worker-1",
                "backend": "llamacpp",
                "status": "up",
                "latency_ms": 12,
                "last_checked_at": "2026-06-14T00:00:00+00:00",
            },
            {
                "id": "agent",
                "served_by": "atlas-worker-agent-1",
                "backend": "llamacpp",
                "status": "down",
                "latency_ms": None,
                "last_checked_at": "2026-06-14T00:00:00+00:00",
            },
        ],
    }

    class _Resp:
        status = 200

        async def json(self):
            return payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        def get(self, url):
            assert url.endswith("/routes")
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    with patch.object(client.aiohttp, "ClientSession", return_value=_Session()):
        result = await client.fetch_routes()
    assert result["default_route"] == "chat"
    by_id = {r["id"]: r for r in result["routes"]}
    assert by_id["chat"]["status"] == "up"
    assert by_id["agent"]["status"] == "down"
    assert "quick" in by_id and "metacog" in by_id


@pytest.mark.asyncio
async def test_run_hub_agent_via_context_exec_uses_chat_profile() -> None:
    from scripts.context_exec_agent_bridge import run_hub_agent_via_context_exec
    from orion.schemas.context_exec import ContextExecRunV1

    fake_run = ContextExecRunV1(
        run_id="run1",
        status="ok",
        mode="belief_provenance",
        text="Where did the Denver belief come from?",
        final_text="supported",
        runtime_debug={"llm_profile": "chat"},
    )

    with patch(
        "scripts.context_exec_agent_bridge.run_context_exec",
        new=AsyncMock(return_value=fake_run),
    ):
        out = await run_hub_agent_via_context_exec(
            req=CortexChatRequest(
                prompt="Where did the Denver belief come from?",
                mode="agent",
                options={"llm_route": "chat"},
            ),
            prompt="Where did the Denver belief come from?",
            correlation_id="corr-1",
            route_debug={"llm_route": "chat"},
        )
    assert out.get("llm_response") == "supported"
    assert out.get("routing_debug", {}).get("context_exec_lane") is True
    assert out.get("agent_trace", {}).get("mode") == "agent"
