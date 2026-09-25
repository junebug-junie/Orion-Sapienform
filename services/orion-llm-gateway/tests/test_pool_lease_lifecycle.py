"""Lease lifecycle on the gateway: oversized prompts, recall/loss mid-call, bounded waits, and an
unreachable pool. The FakePool (conftest) enforces min_ctx vs the LIVE per-slot contexts and answers
``min_ctx_exceeds_class:<max>`` like the real scheduler."""
from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from orion.gpu_pool.client import Lease, LeaseUnavailable

from app import main as gateway
from app import passthrough_proxy, pool_placement, upstream_cancel
from app.models import ChatBody
from app.settings import settings

OVERFLOW = {"text": "[Error: context overflow]", "raw": {"error": "context_overflow"}}
OVERFLOW_HTTP = b'{"error":{"message":"the request exceeds the available context size"}}'


@pytest.fixture(autouse=True)
def _no_lane_routing(monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)


def _body(route: str, chars: int = 40, **options: Any) -> ChatBody:
    return ChatBody(route=route, messages=[{"role": "user", "content": "x" * chars}], options=options)


def _ok(body, plan) -> Dict[str, Any]:
    return {"text": "hello", "raw": {}, "route": plan.route, "url": plan.route_target.url}


# ── a real upstream that never answers ─────────────────────────────────────────────────────


class SilentUpstream:
    """Accepts connections, reads the request, never answers. Records when the client hung up."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.accepted = threading.Event()
        self.hung_up = threading.Event()
        self._conns: List[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self._conns.append(conn)
            self.accepted.set()
            threading.Thread(target=self._drain, args=(conn,), daemon=True).start()

    def _drain(self, conn: socket.socket) -> None:
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass
        self.hung_up.set()

    def close(self) -> None:
        for conn in self._conns:
            conn.close()
        self.sock.close()


@pytest.fixture
def silent_upstream():
    server = SilentUpstream()
    yield server
    server.close()


def _blocking_run(body, plan) -> Dict[str, Any]:
    """Stands in for run_llm_chat's executors: a sync POST through llm_backend's own client factory
    (the cancel hook lives there), errors folded into a result like the real executors do."""
    from app import llm_backend

    try:
        with llm_backend._common_http_client(body) as client:
            client.post(plan.route_target.url + "/v1/chat/completions", json={})
        return {"text": "done", "raw": {}}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"[Error: llamacpp failed: {exc}]", "raw": {}}


def test_cancel_wakes_a_thread_blocked_on_a_real_socket(silent_upstream):
    """httpx.Client.close() from another thread does not unblock recv(); shutdown() does. This is
    the load-bearing assumption of the bus-path cancel, checked on a real socket."""
    handle = upstream_cancel.UpstreamCancel()
    out: Dict[str, Any] = {}

    class _Plan:
        class route_target:
            url = silent_upstream.url

    def work():
        out["result"] = upstream_cancel.run_cancellable(handle, _blocking_run, None, _Plan)

    thread = threading.Thread(target=work)
    started = time.monotonic()
    thread.start()
    assert silent_upstream.accepted.wait(2)
    handle.cancel("lease_lost")
    thread.join(3)
    assert not thread.is_alive(), "the blocked upstream read was not interrupted"
    assert time.monotonic() - started < 3
    assert out["result"]["text"].startswith("[Error:")
    assert upstream_cancel.current() is None  # the handle does not leak to the next call on this thread


def test_no_handle_means_a_plain_client():
    from app import llm_backend

    assert upstream_cancel.cancellable_transport() is None
    with llm_backend._common_http_client(None) as client:
        assert isinstance(client, httpx.Client)


# ── 1. prompts bigger than every role of the class ───────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_bigger_than_the_class_is_clamped_to_its_largest_role(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    result = await gateway._dispatch_chat(_body("chat", chars=70000 * 4), correlation_id="c")
    assert result["text"] == "hello" and result["url"] == "http://pool-chat:8011"
    assert [c["min_ctx_tokens"] for c in fake_pool.calls] == [70000, 65536]
    assert fake_pool.releases == ["ok"]


@pytest.mark.asyncio
async def test_clamped_then_overflowing_returns_the_overflow_without_another_lease(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: dict(OVERFLOW))
    result = await gateway._dispatch_chat(_body("quick", chars=200000 * 4), correlation_id="c")
    assert result["raw"]["error"] == "context_overflow"
    # clamp to the fast class's largest role (agent, 131072); its overflow is the answer
    assert [c["min_ctx_tokens"] for c in fake_pool.calls] == [200000, 131072]
    assert fake_pool.releases == ["upstream_error"]


@pytest.mark.asyncio
async def test_other_unavailable_reasons_are_not_clamped(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    fake_pool.unavailable = "no_serviceable_role"
    result = await gateway._dispatch_chat(_body("chat"), correlation_id="c")
    assert result["raw"]["details"]["reason"] == "no_serviceable_role"
    assert len(fake_pool.calls) == 1


def test_class_max_ctx_parsing():
    assert pool_placement.class_max_ctx("min_ctx_exceeds_class:65536") == 65536
    assert pool_placement.class_max_ctx("min_ctx_exceeds_class:0") is None
    assert pool_placement.class_max_ctx("min_ctx_exceeds_class:x") is None
    assert pool_placement.class_max_ctx("deadline") is None
    assert pool_placement.class_max_ctx(None) is None


@pytest.fixture
def openai_on(monkeypatch, fake_pool):
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    return fake_pool


def test_passthrough_clamps_then_returns_the_real_overflow(openai_on, monkeypatch):
    posted: List[str] = []

    async def _post(self, url, **kw):
        posted.append(url)
        return httpx.Response(400, content=OVERFLOW_HTTP, headers={"content-type": "application/json"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    response = TestClient(gateway.app).post("/v1/chat/completions", json={
        "model": "chat", "messages": [{"role": "user", "content": "x" * (70000 * 4)}]})
    assert response.status_code == 400
    assert "exceeds the available context" in response.json()["error"]["message"]
    assert [c["min_ctx_tokens"] for c in openai_on.calls] == [70000, 65536]
    assert posted == ["http://pool-chat:8011/v1/chat/completions"]
    assert openai_on.releases == ["upstream_error"]


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_passthrough_stream_overflow_with_nothing_bigger_returns_the_overflow(mock_client_cls, openai_on):
    class _Upstream:
        status_code = 400
        headers = {"content-type": "application/json"}

        async def aread(self):
            return OVERFLOW_HTTP

        async def aclose(self):
            return None

    client = MagicMock()
    client.build_request = MagicMock(return_value=MagicMock())
    client.send = AsyncMock(return_value=_Upstream())
    client.aclose = AsyncMock()
    mock_client_cls.return_value = client
    response = TestClient(gateway.app).post("/v1/messages", json={
        "model": "chat", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    # chat overflowed; the re-lease at 65537 is refused min_ctx_exceeds_class -> the overflow itself
    assert response.status_code == 400
    assert b"exceeds the available context" in response.content
    assert [c["min_ctx_tokens"] for c in openai_on.calls][1] == 65536 + 1
    assert len(openai_on.calls) == 2


# ── 2. recall / loss stops the upstream ──────────────────────────────────────────────────


def _lease() -> Lease:
    from orion.schemas.gpu_pool import GpuLeaseGrantV1

    grant = GpuLeaseGrantV1(lease_id="l1", generation=1, role="fast", cards=["gpu3"], url="http://x:1",
                            profile_name="p", model_file="m.gguf", ctx_per_slot=4096, served_by="circe-worker-fast")
    return Lease("l1", grant)


@pytest.mark.asyncio
async def test_wait_lease_revoked_lost_is_immediate():
    lease = _lease()
    task = asyncio.ensure_future(pool_placement.wait_lease_revoked(lease))
    await asyncio.sleep(0.01)
    assert not task.done()
    lease.lost.set()
    lease.recalled.set()
    assert await asyncio.wait_for(task, 1) == "lost"


@pytest.mark.asyncio
async def test_wait_lease_revoked_recall_waits_out_its_grace():
    lease = _lease()
    lease.recall_by = datetime.now(timezone.utc) + timedelta(seconds=0.15)
    lease.recalled.set()
    started = time.monotonic()
    assert await asyncio.wait_for(pool_placement.wait_lease_revoked(lease), 2) == "recalled"
    assert time.monotonic() - started >= 0.1  # the borrower got its grace


@pytest.mark.asyncio
async def test_wait_lease_revoked_loss_during_grace_ends_it_early():
    lease = _lease()
    lease.recall_by = datetime.now(timezone.utc) + timedelta(seconds=30)
    lease.recalled.set()
    task = asyncio.ensure_future(pool_placement.wait_lease_revoked(lease))
    await asyncio.sleep(0.01)
    lease.lost.set()
    assert await asyncio.wait_for(task, 1) == "lost"


@pytest.mark.asyncio
async def test_bus_call_is_stopped_when_the_lease_is_lost(fake_pool, monkeypatch, silent_upstream):
    fake_pool.urls["fast"] = silent_upstream.url
    monkeypatch.setattr(gateway, "run_llm_chat", _blocking_run)
    task = asyncio.ensure_future(gateway._dispatch_chat(_body("quick"), correlation_id="c"))
    assert await asyncio.to_thread(silent_upstream.accepted.wait, 2)
    started = time.monotonic()
    fake_pool.leases[-1].lost.set()
    result = await asyncio.wait_for(task, 3)
    assert time.monotonic() - started < 2
    assert result["content"] == "" and result["text"] == ""
    assert result["raw"]["error"] == "gpu_pool_recalled"
    assert result["raw"]["details"]["reason"] == "lease_lost"
    assert fake_pool.releases == ["cancelled"]
    assert await asyncio.to_thread(silent_upstream.hung_up.wait, 2)  # the upstream connection is gone


@pytest.mark.asyncio
async def test_bus_call_is_stopped_after_the_recall_grace(fake_pool, monkeypatch, silent_upstream):
    fake_pool.urls["fast"] = silent_upstream.url
    monkeypatch.setattr(gateway, "run_llm_chat", _blocking_run)
    task = asyncio.ensure_future(gateway._dispatch_chat(_body("quick"), correlation_id="c"))
    assert await asyncio.to_thread(silent_upstream.accepted.wait, 2)
    lease = fake_pool.leases[-1]
    lease.recall_by = datetime.now(timezone.utc) + timedelta(seconds=0.2)
    lease.recalled.set()
    await asyncio.sleep(0.1)
    assert not task.done()  # still inside the grace
    result = await asyncio.wait_for(task, 3)
    assert result["raw"]["details"]["reason"] == "lease_recalled"
    assert fake_pool.releases == ["cancelled"]


@pytest.mark.asyncio
async def test_a_recalled_call_that_finishes_in_its_grace_is_kept(fake_pool, monkeypatch):
    def run(body, plan):
        time.sleep(0.05)
        return _ok(body, plan)

    fake_pool.on_grant = lambda lease: (setattr(lease, "recall_by", datetime.now(timezone.utc) + timedelta(seconds=5)),
                                        lease.recalled.set())
    monkeypatch.setattr(gateway, "run_llm_chat", run)
    result = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
    assert result["text"] == "hello"
    assert fake_pool.releases == ["ok"]


@pytest.mark.asyncio
async def test_bus_call_never_outlives_the_callers_budget(fake_pool, monkeypatch, silent_upstream):
    """The upstream client floors its read timeout at 30s; a 1s caller must still get its answer
    (a timeout) at ~1s, with the upstream stopped and the lease released as a timeout."""
    fake_pool.urls["fast"] = silent_upstream.url
    monkeypatch.setattr(gateway, "run_llm_chat", _blocking_run)
    started = time.monotonic()
    result = await asyncio.wait_for(
        gateway._dispatch_chat(_body("quick", gateway_read_timeout_sec=1), correlation_id="c"), 5)
    assert time.monotonic() - started < 2.5
    assert result["raw"]["error"] == "timeout"
    assert result["raw"]["details"]["reason"] == "caller_budget_exhausted"
    assert fake_pool.releases == ["timeout"]


def test_passthrough_request_is_stopped_when_the_lease_is_lost(openai_on, monkeypatch):
    pool = openai_on

    async def _post(self, url, **kw):
        pool.leases[-1].lost.set()  # the pool expires the lease while the upstream is generating
        await asyncio.Event().wait()

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    response = TestClient(gateway.app).post("/v1/chat/completions", json={
        "model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    err = response.json()["error"]
    assert err["type"] == "gpu_pool_recalled" and err["reason"] == "lease_lost"
    assert pool.releases == ["cancelled"]


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_passthrough_stream_is_closed_with_an_error_event_when_the_lease_is_lost(mock_client_cls, openai_on):
    pool = openai_on
    closed: List[bool] = []

    class _Upstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b"event: message_start\ndata: {}\n\n"
            pool.leases[-1].lost.set()
            await asyncio.Event().wait()  # upstream stalls; only the lease watch can end this
            yield b"never"

        async def aclose(self):
            closed.append(True)

    client = MagicMock()
    client.build_request = MagicMock(return_value=MagicMock())
    client.send = AsyncMock(return_value=_Upstream())
    client.aclose = AsyncMock()
    mock_client_cls.return_value = client
    with TestClient(gateway.app).stream("POST", "/v1/messages", json={
            "model": "agent", "max_tokens": 8, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}) as response:
        body = b"".join(response.iter_bytes())
    assert b"message_start" in body
    assert b"event: error" in body and b"gpu_pool_recalled" in body and b"lease_lost" in body
    assert b"never" not in body
    assert closed and pool.active == 0
    assert pool.releases == ["cancelled"]


# ── 3. bounded waits ─────────────────────────────────────────────────────────────────────


class _Request:
    def __init__(self, disconnect_after: int) -> None:
        self.headers: Dict[str, str] = {}
        self.polls = 0
        self._after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return self.polls > self._after


def _no_free_slots(pool) -> None:
    for role in pool.slots:
        pool.slots[role] = 0


@pytest.mark.asyncio
async def test_queued_passthrough_is_withdrawn_when_the_client_leaves(fake_pool, monkeypatch):
    _no_free_slots(fake_pool)
    fake_pool.max_wait_sec = 30.0
    monkeypatch.setattr(passthrough_proxy, "_DISCONNECT_POLL_SEC", 0.01)
    request = _Request(disconnect_after=2)
    started = time.monotonic()
    response = await passthrough_proxy.proxy_on_pool(
        request=request, route_key="quick", forward_body={"model": "quick"}, path="/v1/chat/completions",
        holder=pool_placement.HOLDER_OPENAI, guard=None, correlation_id=None, min_ctx_tokens=10, anthropic=False)
    assert response.status_code == passthrough_proxy.CLIENT_CLOSED_STATUS
    assert time.monotonic() - started < 1
    assert fake_pool.withdrawn == 1 and fake_pool.releases == []


@pytest.mark.asyncio
async def test_queued_passthrough_times_out_on_its_own_wait(fake_pool, monkeypatch):
    _no_free_slots(fake_pool)
    fake_pool.max_wait_sec = 0.05
    response = await passthrough_proxy.proxy_on_pool(
        request=_Request(disconnect_after=10**6), route_key="quick", forward_body={"model": "quick"},
        path="/v1/chat/completions", holder=pool_placement.HOLDER_OPENAI, guard=None, correlation_id=None,
        min_ctx_tokens=10, anthropic=False)
    assert response.status_code == 503
    assert b'"reason":"deadline"' in response.body
    assert fake_pool.calls[0]["deadline_sec"] == settings.llm_gateway_pool_passthrough_wait_sec


@pytest.mark.asyncio
async def test_bus_wait_is_the_callers_budget_when_smaller(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", _ok)
    await gateway._dispatch_chat(_body("quick_background", gateway_read_timeout_sec=20), correlation_id="c")
    assert 19.0 < fake_pool.calls[0]["deadline_sec"] <= 20.0


# ── 4. unreachable pool ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_rpc_timeout_fails_following_calls_fast(fake_pool, monkeypatch):
    attempts: List[int] = []

    @contextlib.asynccontextmanager
    async def timing_out(bus, **kw):
        attempts.append(1)
        raise asyncio.TimeoutError()  # the acquire RPC got no reply (rpc_request's own timeout)
        yield

    monkeypatch.setattr(pool_placement, "gpu_lease", timing_out)
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    try:
        first = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
        second = await gateway._dispatch_chat(_body("quick"), correlation_id="c")
        assert first["raw"]["details"]["reason"] == "pool_unreachable"
        assert second["raw"]["details"]["reason"] == "pool_unreachable"
        assert len(attempts) == 1  # the second call did not wait on the pool again
        pool_placement._unreachable_until[0] = time.monotonic() - 1  # cache expired
        await gateway._dispatch_chat(_body("quick"), correlation_id="c")
        assert len(attempts) == 2
    finally:
        pool_placement.reset_pool_unreachable()


class _ReadyBus:
    enabled = True
    redis = object()


def test_ready_reports_the_pool_bus_fork(monkeypatch):
    from orion.bus.consumer_readiness import BusConsumerReadinessResult

    async def ready_result(*a, **k):
        return BusConsumerReadinessResult(ok=True, bus_consumer_ready=True, intake_channel="x",
                                          subscriber_count=1, dependency_status="available")

    monkeypatch.setattr(gateway, "bus_handle", _ReadyBus())
    monkeypatch.setattr(gateway, "check_bus_consumer_readiness", ready_result)
    client = TestClient(gateway.app)

    monkeypatch.setattr(pool_placement, "_bus", None)
    down = client.get("/ready")
    assert down.status_code == 503 and down.headers["x-gpu-pool-bus"] == "down"
    assert "gpu_pool_bus_unavailable" in down.json()["error"]

    monkeypatch.setattr(pool_placement, "_bus", object())
    up = client.get("/ready")
    assert up.status_code == 200 and up.headers["x-gpu-pool-bus"] == "up"


# ── 6. /routes model path ────────────────────────────────────────────────────────────────


def test_routes_compat_reports_the_full_model_path_when_known():
    role = {"role": "fast", "status": "confirmed", "url": "http://h:8013", "model_file": "Qwen.gguf",
            "model_path": "/models/gguf/Qwen.gguf", "ctx_per_slot": 4096, "vision": False}
    state = {"generated_at": "2026-09-25T00:00:00Z", "cards": [], "roles": [role]}
    quick = {r["id"]: r for r in pool_placement.build_routes_compat(state)["routes"]}["quick"]
    assert quick["model"] == "/models/gguf/Qwen.gguf"
    role.pop("model_path")
    quick = {r["id"]: r for r in pool_placement.build_routes_compat(state)["routes"]}["quick"]
    assert quick["model"] == "Qwen.gguf"
