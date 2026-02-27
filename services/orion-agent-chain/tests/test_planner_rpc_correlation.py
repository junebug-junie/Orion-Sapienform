from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

from app import api as agent_api
from app import planner_rpc
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.agents.schemas import AgentChainRequest


class _FakeCodec:
    def decode(self, _data):
        env = BaseEnvelope(
            kind="agent.planner.result",
            source=ServiceRef(name="planner-react", version="1.0", node="athena"),
            correlation_id=str(uuid4()),
            payload={"status": "ok", "final_answer": {"content": "done", "structured": {}}},
        )
        return SimpleNamespace(ok=True, envelope=env, error=None)


class _FakeBus:
    def __init__(self, *_, **__):
        self.codec = _FakeCodec()
        self.captured = None

    async def connect(self):
        return None

    async def close(self):
        return None

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.captured = {
            "request_channel": request_channel,
            "envelope": envelope,
            "reply_channel": reply_channel,
            "timeout_sec": timeout_sec,
        }
        return {"data": b"ok"}


def test_planner_rpc_uses_child_corr_and_parent_metadata(monkeypatch):
    fake = _FakeBus()
    monkeypatch.setattr(planner_rpc, "OrionBusAsync", lambda *args, **kwargs: fake)

    payload = {
        "request_id": "will-be-overwritten",
        "limits": {"timeout_seconds": 5},
        "goal": {"type": "chat", "description": "x"},
        "context": {},
        "toolset": [],
        "preferences": {},
    }

    asyncio.run(planner_rpc.call_planner_react(payload, parent_correlation_id="parent-123"))

    assert fake.captured is not None
    sent_env = fake.captured["envelope"]
    child_corr = str(sent_env.correlation_id)

    # Child correlation is a UUID and drives planner reply channel.
    UUID(child_corr)
    assert fake.captured["reply_channel"].endswith(child_corr)
    assert sent_env.reply_to == fake.captured["reply_channel"]

    # Parent corr is preserved in payload for traceability.
    assert sent_env.payload.get("parent_correlation_id") == "parent-123"
    assert sent_env.payload.get("request_id") == child_corr


def test_execute_agent_chain_replies_to_exec_parent_corr_flow(monkeypatch):
    captured = {}

    async def _fake_call_planner_react(payload, *, parent_correlation_id=None, rpc_bus=None):
        captured["payload"] = payload
        captured["parent"] = parent_correlation_id
        captured["rpc_bus"] = rpc_bus
        return {"status": "ok", "final_answer": {"content": "ok", "structured": {}}}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_call_planner_react)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda _body: [])

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}], packs=[])
    result = asyncio.run(agent_api.execute_agent_chain(req, correlation_id="parent-corr-xyz"))

    assert result.text == "ok"
    assert captured["parent"] == "parent-corr-xyz"
    assert captured["payload"].get("parent_correlation_id") == "parent-corr-xyz"
    assert captured["rpc_bus"] is None
