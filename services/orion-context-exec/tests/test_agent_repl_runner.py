"""Tests for the agent_repl mode + runner dispatch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_agent_repl_is_a_valid_mode():
    from orion.schemas.context_exec import ContextExecRequestV1

    req = ContextExecRequestV1(text="what does orion-hub do?", mode="agent_repl")
    assert req.mode == "agent_repl"


@pytest.mark.asyncio
async def test_llm_chat_route_forwards_messages_and_stop(monkeypatch):
    from app import llm_tools
    from orion.core.bus.bus_schemas import LLMMessage

    captured = {}

    class FakeCodec:
        def decode(self, data):
            class D:
                ok = True

                class envelope:
                    payload = {"content": "ok"}

            return D()

    class FakeBus:
        codec = FakeCodec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            captured["payload"] = env.payload
            return {"data": b"x"}

    monkeypatch.setattr(llm_tools.settings, "orion_bus_enabled", True, raising=False)

    msgs = [
        LLMMessage(role="system", content="you are an agent"),
        LLMMessage(role="user", content="find the bug"),
    ]
    result = await llm_tools.llm_chat_route(
        FakeBus(),
        prompt="find the bug",
        route="agent",
        messages=msgs,
        stop=["<end_code>"],
    )
    assert result["ok"] is True
    payload = captured["payload"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["options"]["stop"] == ["<end_code>"]
