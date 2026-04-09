from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from app import api as agent_api
from app import planner_rpc
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
        self.forked: _FakePlannerBus | None = None

    async def connect(self):
        return None

    async def close(self):
        return None

    async def fork(self, *, start_rpc_worker: bool = False):
        assert start_rpc_worker is True
        if self.forked is None:
            self.forked = _FakePlannerBus()
        return self.forked

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


class _FakePlannerCodec:
    def decode(self, _data):
        return SimpleNamespace(
            ok=True,
            envelope=BaseEnvelope(
                kind="agent.planner.result",
                source=ServiceRef(name="planner-react", version="1.0", node="athena"),
                correlation_id=str(uuid4()),
                payload={"status": "ok", "final_answer": {"content": "planner done", "structured": {}}},
            ),
            error=None,
        )


class _FakePlannerBus:
    def __init__(self, *_, **__):
        self.codec = _FakePlannerCodec()
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

    async def _fake_execute(req, *, correlation_id=None, rpc_bus=None):
        assert correlation_id == corr
        assert rpc_bus is bus.forked
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

    async def _fake_execute(req, *, correlation_id=None, rpc_bus=None):
        assert rpc_bus is bus.forked
        return AgentChainResult(mode=req.mode, text="agent done", structured={}, planner_raw={})

    monkeypatch.setattr(bus_listener, "execute_agent_chain", _fake_execute)

    asyncio.run(bus_listener._handle_request(bus, {"data": b"ignored"}))

    # Integration-ish assertion: the publish tuple is exactly what Exec rpc_request waits on
    # (same reply channel and same correlation id flow key).
    channel, result_env = bus.published[0]
    assert channel == reply_to
    assert str(result_env.correlation_id) == corr
    assert result_env.payload.get("text") == "agent done"


def test_nested_planner_child_corr_still_replies_to_exec_parent(monkeypatch):
    parent_corr = str(uuid4())
    reply_to = f"orion:exec:result:AgentChainService:{parent_corr}"
    env = _request_env(corr=parent_corr, reply_to=reply_to)
    bus = _FakeBus(env)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda _body, **_kwargs: ([], []))
    monkeypatch.setattr(bus_listener, "execute_agent_chain", agent_api.execute_agent_chain)

    asyncio.run(bus_listener._handle_request(bus, {"data": b"ignored"}))

    fake_planner_bus = bus.forked
    assert fake_planner_bus is not None
    assert fake_planner_bus.captured is not None
    child_corr = str(fake_planner_bus.captured["envelope"].correlation_id)
    assert child_corr and child_corr != parent_corr
    assert fake_planner_bus.captured["reply_channel"].endswith(child_corr)
    assert fake_planner_bus.captured["envelope"].payload.get("parent_correlation_id") == parent_corr

    assert len(bus.published) == 1
    out_channel, out_env = bus.published[0]
    assert out_channel == reply_to
    assert str(out_env.correlation_id) == parent_corr
    assert out_env.payload.get("text") == "planner done"



def test_bus_listener_error_reply_uses_agent_chain_result_shape(monkeypatch):
    corr = str(uuid4())
    reply_to = f"orion:exec:result:AgentChainService:{corr}"
    env = _request_env(corr=corr, reply_to=reply_to)
    bus = _FakeBus(env)

    async def _boom(_req, *, correlation_id=None, rpc_bus=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(bus_listener, "execute_agent_chain", _boom)

    asyncio.run(bus_listener._handle_request(bus, {"data": b"ignored"}))

    assert len(bus.published) == 1
    channel, result_env = bus.published[0]
    assert channel == reply_to
    assert result_env.payload.get("mode") == "agent"
    assert "Agent-chain error:" in (result_env.payload.get("text") or "")
    assert isinstance(result_env.payload.get("planner_raw"), dict)
