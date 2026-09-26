"""Stage 4.4: a call carrying its durable run's GPU pool hold ref ATTACHES to that hold.

The one correctness rule this file pins: the agent role has one slot and the run's hold occupies
it, so a call that took a normal lease would queue behind its own run forever. Bus calls carry the
ref in ``options.gpu_lease``, HTTP passthroughs (FCC) in ``X-Orion-Gpu-Lease``. The old durable
token (``resource_lease`` / ``X-Orion-Resource-Lease``) keeps working alongside until stage 4.6.
Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md, PR 4.4.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock

import httpx
import pytest

from orion.llm import resource_lease as wire
from orion.schemas.gpu_pool import GpuLeaseRefV1

from app import anthropic_passthrough as anthropic
from app import llm_backend as backend
from app import main as gateway
from app import openai_passthrough as openai
from app import passthrough_proxy as proxy
from app import pool_placement
from app import resource_lease as fencing
from app.models import ChatBody
from app.settings import settings

OVERFLOW = {"text": "[Error: context overflow]", "raw": {"error": "context_overflow"}}
REF = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="agent", holder="durable-runs:run-1")


def _ok(body, plan) -> Dict[str, Any]:
    return {"text": "hello", "raw": {}, "route": plan.route, "served_by": plan.route_target.served_by,
            "url": plan.route_target.url}


def _body(**options: Any) -> ChatBody:
    return ChatBody(route="agent", messages=[{"role": "user", "content": "study"}], options=options)


@pytest.fixture
def held(fake_pool, monkeypatch):
    """The agent role has ONE slot (live /props) and the run's hold is sitting in it."""
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    fake_pool.slots["agent"] = 1
    fake_pool.slots["agent-gpu2"] = 0
    fake_pool.slots["chat"] = 0
    fake_pool.add_hold(REF)
    return fake_pool


# ── bus ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_without_the_ref_the_runs_own_call_waits_behind_its_hold(held, monkeypatch):
    """The failure attach exists to prevent: a plain lease for the held role never comes."""
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    result = await gateway._dispatch_chat(_body(), correlation_id="c")
    assert result["raw"]["error"] == "gpu_pool_unavailable"
    assert result["raw"]["details"]["reason"] == "deadline"


@pytest.mark.asyncio
async def test_bus_call_with_ref_attaches_to_the_hold_and_runs_on_its_role(held, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    result = await gateway._dispatch_chat(_body(gpu_lease=REF.model_dump(mode="json")), correlation_id="c")
    assert result["text"] == "hello" and result["url"] == "http://pool-agent:8015"
    assert len(held.calls) == 1 and held.calls[0]["hold"] == REF
    assert held.calls[0]["turn_correlation_id"] == "c"
    assert held.releases == ["ok"]
    assert held.busy["agent"] == 1  # the hold still owns its slot; the child took no second one


@pytest.mark.asyncio
async def test_refused_attach_never_falls_back_to_a_plain_lease(held, monkeypatch):
    stale = REF.model_copy(update={"generation": 1})  # the hold was re-granted since
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    result = await gateway._dispatch_chat(_body(gpu_lease=stale.model_dump(mode="json")), correlation_id="c")
    assert result["raw"]["error"] == "gpu_pool_unavailable"
    assert result["raw"]["details"]["reason"] == "unknown_lease"
    assert [c.get("hold") for c in held.calls] == [stale]  # one attach, no acquire behind it


@pytest.mark.asyncio
async def test_malformed_bus_ref_is_rejected_before_any_pool_call(held, monkeypatch):
    run = AsyncMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(gateway, "run_llm_chat", run)
    result = await gateway._dispatch_chat(_body(gpu_lease={"lease_id": "hold-1"}), correlation_id="c")
    assert result["raw"]["error"] == "resource_lease_rejected"
    assert result["raw"]["details"]["reason"] == "malformed_gpu_lease"
    assert held.calls == []


@pytest.mark.asyncio
async def test_overflow_under_a_hold_is_returned_without_a_re_lease(held, monkeypatch):
    """A child can only run on its hold's role, so there is no bigger role to re-lease onto."""
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(OVERFLOW))
    result = await gateway._dispatch_chat(_body(gpu_lease=REF.model_dump(mode="json")), correlation_id="c")
    assert result["raw"]["error"] == "context_overflow"
    assert len(held.calls) == 1
    assert held.releases == ["ok"]


@pytest.mark.asyncio
async def test_old_token_and_new_ref_coexist(held, monkeypatch):
    """Until 4.6 both may ride one call: the durable token is still checked, placement attaches."""
    check = AsyncMock()
    monkeypatch.setattr(settings, "llm_gateway_lease_validation_enabled", True)
    monkeypatch.setattr(fencing.LeaseGuard, "check", check)
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    token = {"lease_id": "legacy", "lane": "agent", "backend_key": "http://agent:8015"}
    result = await gateway._dispatch_chat(
        _body(gpu_lease=REF.model_dump(mode="json"), resource_lease=token), correlation_id="c")
    assert result["text"] == "hello"
    assert check.await_count >= 2
    assert held.calls[0]["hold"] == REF


def test_lane_routing_keeps_the_route_of_a_held_call(held, monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", True)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "study"}],
                    options={"llm_lane": "background", "gpu_lease": REF.model_dump(mode="json")})
    assert backend.plan_llm_chat(body).route == "agent"


# ── HTTP passthroughs (FCC) ─────────────────────────────────────────────────────────────────


def _client(response: httpx.Response):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    return client


@pytest.mark.asyncio
async def test_anthropic_header_attaches_and_is_not_forwarded_upstream(held, monkeypatch):
    client = _client(httpx.Response(200, json={"content": "ok"}))
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: client)
    request = SimpleNamespace(headers={wire.GPU_LEASE_HEADER: wire.encode_gpu_lease_header(REF)},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": []}),
                              is_disconnected=AsyncMock(return_value=False))
    response = await anthropic.handle_messages_post(request)
    assert response.status_code == 200
    assert held.calls[0]["hold"] == REF
    assert client.post.call_args.args[0] == "http://pool-agent:8015/v1/messages"
    forwarded = {k.lower() for k in client.post.call_args.kwargs["headers"]}
    assert wire.GPU_LEASE_HEADER.lower() not in forwarded
    assert held.releases == ["ok"]
    # Interleave may make the child wait one higher-priority inference: the bus budget, not 60s.
    assert held.calls[0]["deadline_sec"] == pool_placement.wait_budget_sec("system")
    assert held.calls[0]["deadline_sec"] != pool_placement.passthrough_wait_sec()


@pytest.mark.asyncio
async def test_openai_header_attaches(held, monkeypatch):
    client = _client(httpx.Response(200, json={"choices": []}))
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: client)
    request = SimpleNamespace(headers={wire.GPU_LEASE_HEADER: wire.encode_gpu_lease_header(REF)},
                              json=AsyncMock(return_value={"model": "agent", "messages": []}),
                              is_disconnected=AsyncMock(return_value=False))
    response = await openai.handle_chat_completions_post(request)
    assert response.status_code == 200
    assert held.calls[0]["hold"] == REF


@pytest.mark.asyncio
async def test_http_overflow_under_a_hold_is_returned_without_a_re_lease(held, monkeypatch):
    overflow = httpx.Response(400, json={"error": {"message": "the request exceeds the available context size"}})
    client = _client(overflow)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: client)
    request = SimpleNamespace(headers={wire.GPU_LEASE_HEADER: wire.encode_gpu_lease_header(REF)},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": []}),
                              is_disconnected=AsyncMock(return_value=False))
    response = await anthropic.handle_messages_post(request)
    assert response.status_code == 400
    assert len(held.calls) == 1


@pytest.mark.asyncio
async def test_malformed_header_is_409_before_any_pool_call(held, monkeypatch):
    request = SimpleNamespace(headers={wire.GPU_LEASE_HEADER: "!bad!"},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": []}))
    response = await anthropic.handle_messages_post(request)
    assert response.status_code == 409
    assert b"malformed_gpu_lease" in response.body
    assert held.calls == []
