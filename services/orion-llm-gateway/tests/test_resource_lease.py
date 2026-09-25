"""Durable-run leases are still validated (admission token), but placement is a pool lease."""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import httpx
import pytest

from orion.llm import resource_lease as wire
from app import anthropic_passthrough as anthropic
from app import llm_backend as backend
from app import main as gateway
from app import passthrough_proxy as proxy
from app import resource_lease as fencing
from app.models import ChatBody
from app.settings import settings


@pytest.fixture
def token():
    now = datetime.now(timezone.utc)
    return {"run_id": "run", "demand_id": "run:turn:llm.route.agent", "lease_id": "lease",
            "resource_key": "llm.route.agent", "lane": "agent", "backend_key": "http://agent:8015",
            "generation": 3, "granted_at": now.isoformat(), "heartbeat_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=60)).isoformat(), "status": "active"}


@pytest.fixture
def configured(monkeypatch, fake_pool):
    monkeypatch.setattr(settings, "llm_gateway_lease_validation_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    return fake_pool


def test_wire_roundtrip(token):
    assert wire.decode_lease_header(wire.encode_lease_header(token)) == wire.ResourceLeaseV1.model_validate(token).model_dump(mode="json")


@pytest.mark.parametrize("value", ["", "!bad!", "a" * 8193])
def test_malformed_header_rejected(value):
    with pytest.raises(wire.ResourceLeaseRejected):
        wire.decode_lease_header(value)


@pytest.mark.asyncio
async def test_route_fencing_precedes_network(token, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(wire.httpx, "AsyncClient", client)
    with pytest.raises(wire.ResourceLeaseRejected, match="route_mismatch"):
        await wire.validate_resource_lease(token, lane="metacog", backend_key="http://metacog:8012", validation_url="http://broker/leases/validate")
    client.assert_not_called()


@pytest.mark.asyncio
async def test_authority_can_accept_renewed_original_token(token, monkeypatch):
    token["expires_at"] = "2000-01-01T00:00:00+00:00"
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(200, json={"valid": True}, request=httpx.Request("POST", "http://broker/leases/validate"))
    monkeypatch.setattr(wire.httpx, "AsyncClient", lambda **kwargs: client)
    await wire.validate_resource_lease(token, lane="agent", backend_key=token["backend_key"], validation_url="http://broker/leases/validate")
    assert client.post.call_args.kwargs["json"]["lease"]["generation"] == 3


@pytest.mark.asyncio
async def test_validation_outage_fails_closed(token, monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.side_effect = httpx.ConnectError("down")
    monkeypatch.setattr(wire.httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(wire.ResourceLeaseRejected, match="validation_unavailable"):
        await wire.validate_resource_lease(token, lane="agent", backend_key=token["backend_key"], validation_url="http://broker/leases/validate")


@pytest.mark.asyncio
async def test_legacy_call_does_not_consult_broker(monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_lease_validation_enabled", True)
    validator = AsyncMock(side_effect=AssertionError("legacy must not validate"))
    monkeypatch.setattr(fencing, "validate_resource_lease", validator)
    guard = fencing.LeaseGuard(None, lane="agent")
    await guard.check()
    assert await guard.run(AsyncMock(return_value="ordinary")()) == "ordinary"
    validator.assert_not_called()


@pytest.mark.asyncio
async def test_guard_validates_against_the_leases_own_backend_key(token, monkeypatch):
    """The pool may place the call on another role than durable admission saw in /routes, so
    the durable token is checked for lane + broker generation, not against the granted URL."""
    monkeypatch.setattr(settings, "llm_gateway_lease_validation_enabled", True)
    validator = AsyncMock()
    monkeypatch.setattr(fencing, "validate_resource_lease", validator)
    await fencing.LeaseGuard(token, lane="agent").check()
    assert validator.await_args.kwargs["lane"] == "agent"
    assert validator.await_args.kwargs["backend_key"] == token["backend_key"]


@pytest.mark.asyncio
async def test_periodic_check_cancels_idle_operation(configured, token, monkeypatch):
    cancelled = asyncio.Event()
    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async def tick(tasks, **kwargs):
        await asyncio.sleep(0)
        return set(), tasks
    monkeypatch.setattr(fencing.asyncio, "wait", tick)
    guard = fencing.LeaseGuard(token, lane="agent")
    monkeypatch.setattr(guard, "check", AsyncMock(side_effect=wire.ResourceLeaseRejected("expired")))
    with pytest.raises(wire.ResourceLeaseRejected, match="expired"):
        await guard.run(operation())
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stale_result_is_not_returned(configured, token, monkeypatch):
    guard = fencing.LeaseGuard(token, lane="agent")
    monkeypatch.setattr(guard, "check", AsyncMock(side_effect=wire.ResourceLeaseRejected("generation_changed")))
    with pytest.raises(wire.ResourceLeaseRejected, match="generation_changed"):
        await guard.run(AsyncMock(return_value="late answer")())


def test_assignment_preserves_route_over_legacy_lane_hint(configured, token, monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", True)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "study"}],
                    options={"llm_lane": "background", "resource_lease": token})
    plan = backend.plan_llm_chat(body)
    assert plan.route == "agent" and plan.work_class == "agent"


@pytest.mark.asyncio
async def test_bus_rejects_stale_lease_before_any_pool_lease(configured, token, monkeypatch):
    monkeypatch.setattr(fencing.LeaseGuard, "check", AsyncMock(side_effect=wire.ResourceLeaseRejected("expired")))
    run = MagicMock(side_effect=AssertionError("stale lease must not execute"))
    monkeypatch.setattr(gateway, "run_llm_chat", run)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "study"}], options={"resource_lease": token})
    result = await gateway._dispatch_chat(body, correlation_id="corr")
    assert result["text"] == "" and result["raw"]["error"] == "resource_lease_rejected"
    assert configured.calls == []
    run.assert_not_called()


@pytest.mark.asyncio
async def test_burst_route_with_durable_lease_is_validated_and_takes_a_pool_lease(configured, token, monkeypatch):
    token["lane"] = "agent-burst"
    check = AsyncMock()
    monkeypatch.setattr(fencing.LeaseGuard, "check", check)
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: {
        "text": "ok", "raw": {}, "route": plan.route, "served_by": plan.route_target.served_by})
    body = ChatBody(route="agent-burst", messages=[{"role": "user", "content": "study"}],
                    options={"resource_lease": token})
    result = await gateway._dispatch_chat(body, correlation_id="corr")
    assert result["text"] == "ok"
    assert check.await_count >= 2  # before the pool lease and on the result
    assert configured.calls[0]["work_class"] == "agent"
    assert result["served_by"] == "circe-worker-agent"
    assert configured.releases == ["ok"]


@pytest.mark.asyncio
async def test_bus_discards_result_after_lease_loss(configured, token, monkeypatch):
    monkeypatch.setattr(fencing.LeaseGuard, "check", AsyncMock(side_effect=[None, wire.ResourceLeaseRejected("cancelled")]))
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: {"text": "stale work", "raw": {}})
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "study"}], options={"resource_lease": token})
    result = await gateway._dispatch_chat(body, correlation_id="corr")
    assert result["text"] == "" and result["raw"]["error"] == "resource_lease_rejected"


@pytest.mark.asyncio
async def test_fcc_http_rejects_stale_before_pool_or_upstream(configured, token, monkeypatch):
    monkeypatch.setattr(fencing.LeaseGuard, "check", AsyncMock(side_effect=wire.ResourceLeaseRejected("expired")))
    client = MagicMock(side_effect=AssertionError("must not call upstream"))
    monkeypatch.setattr(proxy.httpx, "AsyncClient", client)
    request = SimpleNamespace(headers={wire.LEASE_HEADER: wire.encode_lease_header(token)},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": []}))
    response = await anthropic.handle_messages_post(request)
    assert response.status_code == 409
    client.assert_not_called()
    assert configured.calls == []


@pytest.mark.asyncio
async def test_fcc_http_discards_stale_response_and_does_not_forward_token(configured, token, monkeypatch):
    monkeypatch.setattr(fencing.LeaseGuard, "check", AsyncMock(side_effect=[None, wire.ResourceLeaseRejected("expired")]))
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(200, json={"content": "late answer"})
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: client)
    request = SimpleNamespace(headers={wire.LEASE_HEADER: wire.encode_lease_header(token)},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": []}))
    response = await anthropic.handle_messages_post(request)
    assert response.status_code == 409
    assert wire.LEASE_HEADER not in client.post.call_args.kwargs["headers"]
    assert client.post.call_args.args[0] == "http://pool-agent:8015/v1/messages"
    assert configured.releases == ["ok"] and configured.active == 0


@pytest.mark.asyncio
async def test_fcc_stream_fence_closes_upstream_emits_error_and_releases(configured, token, monkeypatch):
    monkeypatch.setattr(fencing.LeaseGuard, "check", AsyncMock())
    async def chunks(self, source):
        yield b"data: first\n\n"
        raise wire.ResourceLeaseRejected("cancelled")
    monkeypatch.setattr(fencing.LeaseGuard, "chunks", chunks)
    upstream = MagicMock(status_code=200, headers={"content-type": "text/event-stream"})
    upstream.aclose = AsyncMock()
    client = MagicMock()
    client.send = AsyncMock(return_value=upstream)
    client.aclose = AsyncMock()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kwargs: client)
    request = SimpleNamespace(headers={wire.LEASE_HEADER: wire.encode_lease_header(token)},
                              json=AsyncMock(return_value={"model": "llamacpp/agent", "messages": [], "stream": True}))
    response = await anthropic.handle_messages_post(request)
    assert configured.active == 1  # held while the stream is open
    body = b"".join([chunk async for chunk in response.body_iterator])
    assert b"event: error" in body and b"resource_lease_rejected" in body
    upstream.aclose.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert configured.active == 0 and configured.releases == ["ok"]
