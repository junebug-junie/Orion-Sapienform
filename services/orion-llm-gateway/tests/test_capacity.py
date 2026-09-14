"""Real Gateway entry points share tickets until their backend work ends."""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app import anthropic_passthrough as anthropic
from app import capacity as cap
from app import llm_backend as backend
from app import main as gateway
from app import openai_passthrough as openai
from app import resource_lease as lease_guard
from app import upstream_admission
from app.models import ChatBody
from app.settings import settings
from orion.llm.resource_lease import LEASE_HEADER, encode_lease_header


class Authority:
    def __init__(self):
        self.active = {}
        self.requests = {}
        self.events = []
        self.busy = asyncio.Event()
        self.renewed = asyncio.Event()
        self.released = asyncio.Event()
        self.valid = True
        self.ttl = 60.0
        self.lose_ack = False
        self.owner_lease_id = None
        self.waiting_ids = set()
        self.two_waiters = asyncio.Event()

    async def post(self, action, payload):
        self.events.append((action, dict(payload)))
        if action == "acquire":
            request_id = payload["request_id"]
            if request_id in self.requests:
                assert payload == self.requests[request_id], "retries must keep their original budget and identity"
            self.requests[request_id] = dict(payload)
            if self.owner_lease_id and (payload.get("lease") or {}).get("lease_id") != self.owner_lease_id:
                self.waiting_ids.add(request_id)
                if len(self.waiting_ids) >= 2:
                    self.two_waiters.set()
                return {"acquired": False, "reason": "durable_lease_active", "permit": None}
            existing = self.active.get(request_id)
            if existing:
                return {"acquired": True, "reason": "duplicate", "permit": existing}
            if any(row["backend_key"] == payload["backend_key"] for row in self.active.values()):
                self.busy.set()
                return {"acquired": False, "reason": "capacity_full", "permit": None}
            now = datetime.now(timezone.utc)
            lease = payload.get("lease") or {}
            permit = {key: payload[key] for key in ("request_id", "correlation_id", "lane", "backend_key")}
            permit.update(permit_id=f"permit-{request_id}", lease_id=lease.get("lease_id"), generation=lease.get("generation"),
                          granted_at=now.isoformat(), heartbeat_at=now.isoformat(),
                          expires_at=(now + timedelta(seconds=self.ttl)).isoformat(), status="active")
            self.active[request_id] = permit
            if self.lose_ack:
                self.lose_ack = False
                raise cap.CapacityUnavailable("lost acquire acknowledgement")
            return {"acquired": True, "reason": "granted", "permit": permit}
        permit = self.active.get(payload["request_id"])
        assert permit and permit["permit_id"] == payload["permit_id"]
        if action == "renew":
            now = datetime.now(timezone.utc)
            permit.update(heartbeat_at=now.isoformat(), expires_at=(now + timedelta(seconds=self.ttl)).isoformat())
            self.renewed.set()
            return {"valid": self.valid, "permit": permit}
        assert action == "release"
        del self.active[payload["request_id"]]
        self.released.set()
        return {"released": True}


@pytest.fixture
def authority(monkeypatch):
    authority = Authority()
    async def post(self, action, payload):
        return await authority.post(action, payload)
    monkeypatch.setattr(cap.CapacityPermit, "_post", post)
    monkeypatch.setattr(settings, "llm_gateway_capacity_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_upstream_max_inflight", 1)
    monkeypatch.setattr(settings, "llm_gateway_background_poll_interval_sec", 0.001)
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps({"agent": {"url": "http://worker:8015/", "backend": "llamacpp"}}))
    backend._load_route_targets.cache_clear()
    upstream_admission.reset_upstream_admission_for_tests()
    gateway.reset_executor_for_tests()
    yield authority
    backend._load_route_targets.cache_clear()
    upstream_admission.reset_upstream_admission_for_tests()
    gateway.reset_executor_for_tests()


def permit():
    return cap.CapacityPermit(lane="agent", backend_key="http://worker:8015/", correlation_id="corr", budget_sec=30)


def request(*, stream=False):
    return SimpleNamespace(headers={"x-request-id": "http-corr"}, json=AsyncMock(return_value={
        "model": "agent", "messages": [{"role": "user", "content": "hello"}], "stream": stream,
    }))


@pytest.mark.asyncio
async def test_lost_acquire_ack_reuses_id_and_original_budget(authority):
    authority.lose_ack = True
    ticket = await permit().acquire()
    try:
        assert len(authority.active) == 1
        calls = [payload for action, payload in authority.events if action == "acquire"]
        assert len(calls) == 2 and calls[0] == calls[1]
        assert calls[0]["backend_key"] == "http://worker:8015"
    finally:
        await ticket.close()


@pytest.mark.asyncio
async def test_capacity_wait_spends_caller_budget_without_dispatch(authority):
    first = await permit().acquire()
    second = permit()
    second.deadline = cap.time.monotonic() - 1
    try:
        with pytest.raises(cap.CapacityRejected, match="wait_budget_exhausted"):
            await second.acquire()
        assert len(authority.active) == 1
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_false_renewal_rejects_submission_and_result(authority):
    ticket = await permit().acquire()
    authority.valid = False
    submit = AsyncMock()
    try:
        with pytest.raises(cap.CapacityRejected):
            await ticket.run_blocking(submit)
        submit.assert_not_called()
    finally:
        await ticket.close()
    authority.valid = True
    ticket = await permit().acquire()
    async def result():
        authority.valid = False
        return "stale output"
    try:
        with pytest.raises(cap.CapacityRejected):
            await ticket.run(result())
    finally:
        await ticket.close()


@pytest.mark.asyncio
async def test_final_authority_check_cannot_extend_caller_budget(authority, monkeypatch):
    ticket = await permit().acquire()
    ticket.deadline = cap.time.monotonic() + 0.01
    async def slow_renew():
        await asyncio.Event().wait()
    monkeypatch.setattr(ticket, "_renew", slow_renew)
    try:
        with pytest.raises(cap.CapacityRejected, match="budget_exhausted"):
            await asyncio.wait_for(ticket.run(AsyncMock(return_value="immediate result")()), 2)
    finally:
        await ticket.close()


class UpstreamClient:
    def __init__(self, authority):
        self.authority = authority
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def post(self, *args, **kwargs):
        assert self.authority.active, "HTTP generation bypassed admission"
        self.authority.events.append(("upstream_post", {}))
        return httpx.Response(200, json={"content": "answer"})

    def build_request(self, *args, **kwargs):
        return object()

    async def send(self, *args, **kwargs):
        assert self.authority.active, "HTTP stream bypassed admission"
        self.authority.events.append(("upstream_send", {}))
        return self

    async def aiter_bytes(self):
        yield b"data: hello\n\n"
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True
        self.authority.events.append(("upstream_close", {}))


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [anthropic.handle_messages_post, openai.handle_chat_completions_post])
async def test_http_stream_holds_ticket_until_upstream_closed(authority, monkeypatch, handler):
    client = UpstreamClient(authority)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    response = await handler(request(stream=True))
    assert len(authority.active) == 1 and not client.closed
    assert await anext(response.body_iterator) == b"data: hello\n\n"
    assert authority.active
    await response.body_iterator.aclose()
    assert client.closed and not authority.active
    events = [event for event, _ in authority.events]
    assert events.index("upstream_close") < events.index("release")


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [anthropic.handle_messages_post, openai.handle_chat_completions_post])
async def test_response_send_failure_before_body_still_releases(authority, monkeypatch, handler):
    client = UpstreamClient(authority)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    response = await handler(request(stream=True))
    async def send(message):
        raise OSError("disconnected before headers")
    async def receive():
        await asyncio.Event().wait()
    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert client.closed and not authority.active


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [anthropic.handle_messages_post, openai.handle_chat_completions_post])
async def test_real_asgi_disconnect_completes_awaited_cleanup(authority, monkeypatch, handler):
    client = UpstreamClient(authority)
    async def close():
        await asyncio.sleep(0)  # AnyIO's cancelled scope used to cancel every cleanup await.
        client.closed = True
        authority.events.append(("upstream_close", {}))
    client.aclose = close
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    response = await handler(request(stream=True))
    first_chunk = asyncio.Event()
    async def send(message):
        if message["type"] == "http.response.body":
            first_chunk.set()
    async def receive():
        await first_chunk.wait()
        return {"type": "http.disconnect"}
    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    assert client.closed and not authority.active
    events = [event for event, _ in authority.events]
    assert events.count("upstream_close") == 2
    assert events.index("upstream_close") < events.index("release")


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [anthropic.handle_messages_post, openai.handle_chat_completions_post])
async def test_stream_lease_validation_cannot_exceed_capacity_budget(authority, monkeypatch, handler):
    client = UpstreamClient(authority)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(anthropic, "_passthrough_read_timeout_sec", lambda: 0.03)
    monkeypatch.setattr(openai, "_passthrough_read_timeout_sec", lambda: 0.03)
    async def slow_validation(self, source):
        yield await anext(source)
        await asyncio.Event().wait()  # models an unresponsive final lease validation
    monkeypatch.setattr(lease_guard.LeaseGuard, "chunks", slow_validation)
    response = await handler(request(stream=True))
    async def consume():
        return b"".join([chunk async for chunk in response.body_iterator])
    body = await asyncio.wait_for(consume(), 2)
    assert b"capacity_budget_exhausted" in body
    assert client.closed and not authority.active


@pytest.mark.asyncio
async def test_cancelled_bus_handler_renews_until_real_thread_then_http_admits(authority, monkeypatch):
    authority.ttl = 0.12
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    done = threading.Event()
    def backend_work(body, plan):
        loop.call_soon_threadsafe(started.set)
        assert done.wait(timeout=5), "test did not release backend"
        return {"text": "answer"}
    monkeypatch.setattr(gateway, "run_llm_chat", backend_work)
    client = UpstreamClient(authority)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "study"}])
    bus = asyncio.create_task(gateway._dispatch_chat(body, correlation_id="bus-corr"))
    http = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        bus.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bus
        authority.renewed.clear()
        await asyncio.wait_for(authority.renewed.wait(), 2)
        assert len(authority.active) == 1 and not authority.released.is_set()
        http = asyncio.create_task(anthropic.handle_messages_post(request()))
        await asyncio.wait_for(authority.busy.wait(), 2)
        assert not http.done(), "HTTP stole a permit from a cancelled but live thread"
        done.set()
        response = await asyncio.wait_for(http, 2)
        assert response.status_code == 200 and not authority.active
    finally:
        done.set()
        await asyncio.gather(bus, return_exceptions=True)
        if http is not None:
            await asyncio.gather(http, return_exceptions=True)
        await asyncio.gather(*list(cap._supervisors), return_exceptions=True)


@pytest.mark.asyncio
async def test_disabled_capacity_preserves_direct_http(authority, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_capacity_enabled", False)
    client = UpstreamClient(authority)
    client.post = AsyncMock(return_value=httpx.Response(200, json={"content": "legacy"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    for handler in (anthropic.handle_messages_post, openai.handle_chat_completions_post):
        response = await handler(request())
        assert response.status_code == 200
    assert not authority.events or all(name == "upstream_close" for name, _ in authority.events)


@pytest.mark.asyncio
async def test_supervisor_shutdown_still_waits_for_actual_future(authority):
    ticket = await permit().acquire()
    future = Future()
    future.set_running_or_notify_cancel()
    submitted = asyncio.Event()
    def submit():
        submitted.set()
        return future
    work = asyncio.create_task(ticket.run_blocking(submit))
    await submitted.wait()
    supervisor = next(task for task in cap._supervisors if task.get_name() == f"capacity-backend-{ticket.request_id}")
    supervisor.cancel()
    await asyncio.gather(work, return_exceptions=True)
    assert future.running() and authority.active
    assert not authority.released.is_set()
    future.set_result("thread ended")
    await asyncio.wait_for(authority.released.wait(), 2)
    assert not authority.active


@pytest.mark.asyncio
async def test_ordinary_bus_waiters_do_not_starve_durable_owner_local_gate(authority, monkeypatch):
    authority.owner_lease_id = "owner"
    monkeypatch.setattr(settings, "llm_gateway_upstream_max_inflight", 2)
    monkeypatch.setattr(lease_guard.LeaseGuard, "check", AsyncMock())
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: {"text": "owner dispatched"})
    ordinary_body = ChatBody(route="agent", messages=[{"role": "user", "content": "ordinary"}])
    ordinary = [asyncio.create_task(gateway._dispatch_chat(ordinary_body, correlation_id=f"ordinary-{n}")) for n in range(2)]
    now = datetime.now(timezone.utc)
    owner = {"run_id": "run", "demand_id": "demand", "lease_id": "owner", "resource_key": "llm.route.agent",
             "lane": "agent", "backend_key": "http://worker:8015", "generation": 1, "status": "active",
             "granted_at": now.isoformat(), "heartbeat_at": now.isoformat(),
             "expires_at": (now + timedelta(seconds=60)).isoformat()}
    try:
        await asyncio.wait_for(authority.two_waiters.wait(), 2)
        owner_body = ordinary_body.model_copy(update={"options": {"resource_lease": owner}})
        result = await asyncio.wait_for(gateway._dispatch_chat(owner_body, correlation_id="owner"), 2)
        assert result["text"] == "owner dispatched"
        assert all(not task.done() for task in ordinary)
    finally:
        for task in ordinary:
            task.cancel()
        await asyncio.gather(*ordinary, return_exceptions=True)
        await asyncio.gather(*list(cap._supervisors), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["bus", "openai"])
async def test_admitted_background_owner_bypasses_legacy_background_gate(authority, monkeypatch, path):
    authority.owner_lease_id = "owner"
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps({"quick_background": {
        "url": "http://worker:8015", "backend": "llamacpp", "priority": "background",
    }}))
    backend._load_route_targets.cache_clear()
    monkeypatch.setattr(lease_guard.LeaseGuard, "check", AsyncMock())
    gate = MagicMock(side_effect=AssertionError("an admitted owner cannot wait behind its own reservation"))
    monkeypatch.setattr(gateway, "background_admission", gate)
    monkeypatch.setattr(openai, "background_admission", gate)
    now = datetime.now(timezone.utc)
    owner = {"run_id": "run", "demand_id": "demand", "lease_id": "owner", "resource_key": "llm.route.quick_background",
             "lane": "quick_background", "backend_key": "http://worker:8015", "generation": 1, "status": "active",
             "granted_at": now.isoformat(), "heartbeat_at": now.isoformat(),
             "expires_at": (now + timedelta(seconds=60)).isoformat()}
    if path == "bus":
        monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: {"text": "owner"})
        body = ChatBody(route="quick_background", messages=[{"role": "user", "content": "study"}],
                        options={"resource_lease": owner})
        result = await gateway._dispatch_chat(body, correlation_id="owner")
        assert result["text"] == "owner"
    else:
        client = UpstreamClient(authority)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
        req = SimpleNamespace(headers={LEASE_HEADER: encode_lease_header(owner)},
                              json=AsyncMock(return_value={"model": "quick_background", "messages": []}))
        result = await openai.handle_chat_completions_post(req)
        assert result.status_code == 200
    gate.assert_not_called()
    await asyncio.gather(*list(cap._supervisors), return_exceptions=True)


def test_capacity_mode_disables_context_overflow_lane_migration(authority, monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = httpx.Response(400, json={"error": "context overflow"},
                                            request=httpx.Request("POST", "http://worker:8015/v1/chat/completions"))
    monkeypatch.setattr(backend, "_common_http_client", lambda body: client)
    ladder = MagicMock(side_effect=AssertionError("must not acquire an unowned backend"))
    monkeypatch.setattr(backend, "_ctx_ladder", ladder)
    monkeypatch.setattr(backend.ctx_overflow, "is_context_overflow", lambda *args: True)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "long study"}])
    backend._execute_openai_chat(body=body, model="test", base_url="http://worker:8015",
                                 backend_name="llamacpp", route="agent")
    assert client.post.call_count == 1
    ladder.assert_not_called()


@pytest.mark.asyncio
async def test_agent_burst_refuses_unleased_calls_even_with_capacity_disabled(monkeypatch):
    from app.capacity import CapacityPermit,CapacityRejected,settings
    for enabled in (False,True):
        monkeypatch.setattr(settings,"llm_gateway_capacity_enabled",enabled)
        ticket=CapacityPermit(lane="agent-burst",backend_key="http://burst",correlation_id="test",budget_sec=1)
        with pytest.raises(CapacityRejected,match="agent_burst_requires"):
            await ticket.acquire()
