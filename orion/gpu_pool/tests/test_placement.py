"""orion.gpu_pool.placement: what Orion may say about the model serving a turn."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion.gpu_pool.placement import (
    STATE_RPC_HEALTH_LABEL, discovered_role, fetch_pool_state, placement_from_lease,
    placement_from_route_default,
)
from orion.schemas.gpu_pool import GPU_POOL_STATE_REQUEST_CHANNEL

STATE = {"roles": [
    {"role": "agent", "kind": "llm", "cards": ["gpu1"], "url": "u", "status": "confirmed",
     "profile_name": "agent-flex", "model_file": "Agent-27B.gguf"},
    {"role": "agent-gpu2", "kind": "llm", "cards": ["gpu2"], "url": "u", "status": "unloaded"},
    {"role": "fast", "kind": "llm", "cards": ["gpu3"], "url": "u", "status": "mismatch",
     "profile_name": "announced", "model_file": "Other.gguf"},
]}


def test_lease_on_a_confirmed_role_names_its_discovered_model():
    p = placement_from_lease("agent", discovered_role(STATE, "agent"))
    assert (p.source, p.role, p.model, p.profile) == ("from_lease", "agent", "Agent-27B.gguf", "agent-flex")
    assert "GPU pool role agent, profile agent-flex" in p.self_line()


def test_lease_on_an_unconfirmed_role_never_names_a_model():
    for role in ("agent-gpu2", "fast", "missing"):
        p = placement_from_lease(role, discovered_role(STATE, role))
        assert p.model is None
        assert "do not name one" in p.self_line()
    # mismatch: the pool DID read a model; the reason must say it is unconfirmed, not unreadable
    assert "reports that role as mismatch" in placement_from_lease("fast", discovered_role(STATE, "fast")).self_line()
    assert "could not be read" in placement_from_lease("missing", None).self_line()


def test_route_default_is_worded_as_a_default_and_strips_the_path():
    p = placement_from_route_default("agent", "/models/gguf/Agent-27B.gguf")
    line = p.self_line()
    assert "Default backend model for route agent: Agent-27B.gguf" in line
    assert "not a confirmed one" in line
    assert placement_from_route_default("agent", None).self_line() is None


def test_discovered_role_tolerates_garbage():
    assert discovered_role(None, "agent") is None
    assert discovered_role({"roles": [{"role": "agent", "bogus": 1}]}, "agent") is None


class _Bus:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc, self.calls = payload, exc, []
        self.codec = SimpleNamespace(decode=lambda data: SimpleNamespace(
            ok=True, envelope=SimpleNamespace(payload=data)))

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec, health_label):
        self.calls.append((channel, env, reply_channel, health_label))
        if self.exc:
            raise self.exc
        return {"data": self.payload}


def test_fetch_pool_state_asks_without_leases_on_its_own_health_label():
    bus = _Bus(payload=STATE)
    assert asyncio.run(fetch_pool_state(bus, source="t")) == STATE
    channel, env, reply_channel, label = bus.calls[0]
    assert channel == GPU_POOL_STATE_REQUEST_CHANNEL
    assert env.reply_to == reply_channel
    assert env.payload == {"include_leases": False}
    assert label == STATE_RPC_HEALTH_LABEL


def test_fetch_pool_state_fails_open():
    assert asyncio.run(fetch_pool_state(_Bus(exc=TimeoutError()), source="t")) is None
