from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from app import bus_listener
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.agents.schemas import AgentChainResult


class _FakeCodec:
    def __init__(self, envelope: BaseEnvelope):
        self._envelope = envelope

    def decode(self, _data):
        return SimpleNamespace(ok=True, envelope=self._envelope, error=None)


class _FakeBus:
    def __init__(self, envelope: BaseEnvelope):
        self.codec = _FakeCodec(envelope)
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _request_env(*, corr: str, reply_to: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="agent.chain.request",
        source=ServiceRef(name="cortex-exec", version="1.0", node="athena"),
        correlation_id=corr,
        reply_to=reply_to,
        payload={
            "text": "hello",
            "mode": "agent",
            "messages": [{"role": "user", "content": "hello"}],
            "packs": ["executive_pack"],
        },
    )


def test_agent_chain_rpc_reply_preserves_corr_and_reply_channel(monkeypatch):
    corr = str(uuid4())
    reply_to = f"orion:exec:result:AgentChainService:{corr}"
    env = _request_env(corr=corr, reply_to=reply_to)
    bus = _FakeBus(env)

    async def _fake_execute(req, *, correlation_id=None):
        assert correlation_id == corr
        return AgentChainResult(mode=req.mode, text="ok", structured={}, planner_raw={})

    monkeypatch.setattr(bus_listener, "execute_agent_chain", _fake_execute)

    asyncio.run(bus_listener._handle_request(bus, {"data": b"ignored"}))

    assert len(bus.published) == 1
    channel, result_env = bus.published[0]
    assert channel == reply_to
    assert str(result_env.correlation_id) == corr
    assert result_env.kind == "agent.chain.result"
    assert result_env.payload.get("text") == "ok"


def test_exec_style_rpc_consumer_would_receive_matching_result(monkeypatch):
    corr = str(uuid4())
    reply_to = f"orion:exec:result:AgentChainService:{corr}"
    env = _request_env(corr=corr, reply_to=reply_to)
    bus = _FakeBus(env)

    async def _fake_execute(req, *, correlation_id=None):
        return AgentChainResult(mode=req.mode, text="agent done", structured={}, planner_raw={})

    monkeypatch.setattr(bus_listener, "execute_agent_chain", _fake_execute)

    asyncio.run(bus_listener._handle_request(bus, {"data": b"ignored"}))

    # Integration-ish assertion: the publish tuple is exactly what Exec rpc_request waits on
    # (same reply channel and same correlation id flow key).
    channel, result_env = bus.published[0]
    assert channel == reply_to
    assert str(result_env.correlation_id) == corr
    assert result_env.payload.get("text") == "agent done"
