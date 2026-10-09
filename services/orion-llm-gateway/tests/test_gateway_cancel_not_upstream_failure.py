"""When the gateway itself hangs up on an upstream call, the worker is not blamed for it.

2026-10-09 (/tmp/gw-disconnect-2026-10-09/): 16/16 ``upstream_failed ... error=upstream_error
reason=RemoteProtocolError`` ERROR lines landed 5-9 ms after ``gateway_caller_budget_exhausted``:
the gateway shut the sockets down (upstream_cancel.py), the worker thread's httpx call raised, and
llm_backend classified the gateway's own hang-up as the worker's failure.

These drive the REAL ``run_llm_chat`` (not a stand-in) against a real socket, so the exception
path under test is the one production runs.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import threading
import uuid
from typing import Dict, List

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app import grammar_emit, llm_backend, upstream_cancel
from app import main as gateway
from app.models import ChatBody
from app.settings import settings


class _Upstream:
    """Accepts connections and reads the request. ``hang_up=False``: never answers (the gateway has
    to cancel). ``hang_up=True``: closes the connection without a response (a genuine worker-side
    RemoteProtocolError)."""

    def __init__(self, *, hang_up: bool) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.hang_up = hang_up
        self.accepted = threading.Event()
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
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            if self.hang_up:
                conn.recv(65536)
                conn.close()
                return
            while conn.recv(65536):
                pass
        except OSError:
            pass

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self.sock.close()


@pytest.fixture
def silent():
    server = _Upstream(hang_up=False)
    yield server
    server.close()


@pytest.fixture
def hangs_up():
    server = _Upstream(hang_up=True)
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)
    grammar_emit.reset_recorder_for_tests()
    monkeypatch.setattr(settings, "llm_gateway_grammar_enabled", True)


def _body(route: str = "quick", **options) -> ChatBody:
    return ChatBody(route=route, messages=[{"role": "user", "content": "ping"}], options=options,
                    trace_id="corr-1")


def _lines(caplog, needle: str) -> List[logging.LogRecord]:
    return [r for r in caplog.records if needle in r.getMessage()]


# ── the gateway hung up: upstream_cancelled, not upstream_failed ───────────────────────────


@pytest.mark.asyncio
async def test_caller_budget_cancel_is_logged_as_upstream_cancelled_not_upstream_failed(
        fake_pool, silent, caplog):
    fake_pool.urls["fast"] = silent.url
    caplog.set_level(logging.DEBUG)
    result = await asyncio.wait_for(
        gateway._dispatch_chat(_body(gateway_read_timeout_sec=1), correlation_id="corr-1"), 8)

    # the caller's reply and the lease release are exactly what they were before this fix
    assert result["text"] == "" and result["content"] == ""
    assert result["raw"]["error"] == "timeout"
    assert result["raw"]["details"]["reason"] == "caller_budget_exhausted"
    assert fake_pool.releases == ["timeout"]
    assert grammar_emit.classify_outcome(result) == "upstream_timeout"

    assert _lines(caplog, "upstream_failed") == []
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    (line,) = _lines(caplog, "upstream_cancelled")
    assert line.levelno == logging.WARNING
    msg = line.getMessage()
    assert "reason=caller_budget_exhausted" in msg
    assert f"url={silent.url}" in msg and "corr=corr-1" in msg and "served_by=" in msg
    assert "elapsed_ms=" in msg and "after_cancel_ms=" in msg


@pytest.mark.asyncio
async def test_lease_lost_cancel_is_logged_as_upstream_cancelled_not_upstream_failed(fake_pool, silent, caplog):
    fake_pool.urls["fast"] = silent.url
    caplog.set_level(logging.DEBUG)
    task = asyncio.ensure_future(gateway._dispatch_chat(_body(), correlation_id="corr-1"))
    assert await asyncio.to_thread(silent.accepted.wait, 3)
    fake_pool.leases[-1].lost.set()
    result = await asyncio.wait_for(task, 5)

    assert result["raw"]["error"] == "gpu_pool_recalled"
    assert result["raw"]["details"]["reason"] == "lease_lost"
    assert fake_pool.releases == ["cancelled"]
    assert _lines(caplog, "upstream_failed") == []
    (line,) = _lines(caplog, "upstream_cancelled")
    assert line.levelno == logging.WARNING and "reason=lease_lost" in line.getMessage()


@pytest.mark.asyncio
async def test_bus_call_cancelled_on_budget_is_counted_once_as_upstream_timeout(fake_pool, silent):
    """Stage 6.2 telemetry sees one call, classed as the cancel (timeout -> upstream_timeout),
    never also as an upstream_error from the worker thread's own failure result."""
    fake_pool.urls["fast"] = silent.url
    env = BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="quick",
                                   options={"gateway_read_timeout_sec": 1}).model_dump(mode="json"),
    )
    await asyncio.wait_for(gateway.handle_chat(env), 8)
    _, _, buckets = grammar_emit.get_recorder().drain()
    totals: Dict[str, int] = {}
    for bucket in buckets.values():
        for cls, n in bucket.classes.items():
            totals[cls] = totals.get(cls, 0) + n
    assert totals == {"upstream_timeout": 1}


def test_cancelled_handle_is_read_from_the_worker_thread_only():
    """Outside a cancellable call, or with a handle nobody cancelled, the check is a no-op."""
    exc = RuntimeError("x")
    kw = dict(backend_name="llamacpp", url="u", route="r", served_by="s", spark_meta={}, trace_id="t")
    assert llm_backend._gateway_cancelled_result(exc, **kw) is None
    handle = upstream_cancel.UpstreamCancel()
    assert upstream_cancel.run_cancellable(handle, lambda: llm_backend._gateway_cancelled_result(exc, **kw)) is None
    handle.cancel("caller_cancelled")
    out = upstream_cancel.run_cancellable(handle, lambda: llm_backend._gateway_cancelled_result(exc, **kw))
    assert out["raw"]["error"] == "timeout"
    assert out["raw"]["details"]["reason"] == "caller_cancelled"
    assert out["raw"]["details"]["cancelled_by"] == "gateway"
    assert out["text"] == ""


@pytest.mark.parametrize("reason, cls", [
    ("caller_budget_exhausted", "upstream_timeout"),
    ("caller_cancelled", "upstream_timeout"),
    ("lease_lost", "gpu_pool_recalled"),
    ("lease_recalled", "gpu_pool_recalled"),
])
def test_cancelled_result_classes_as_the_cancel_never_as_upstream_error(reason, cls):
    """main.py replaces it, but if it ever leaked it must still count as the cancel."""
    kw = dict(backend_name="llamacpp", url="u", route="r", served_by="s", spark_meta={}, trace_id="t")
    handle = upstream_cancel.UpstreamCancel()
    handle.cancel(reason)
    out = upstream_cancel.run_cancellable(
        handle, lambda: llm_backend._gateway_cancelled_result(RuntimeError("x"), **kw))
    assert grammar_emit.classify_outcome(out) == cls


# ── a genuine upstream hang-up is still the worker's failure (#2536 unchanged) ─────────────


@pytest.mark.asyncio
async def test_genuine_remote_protocol_error_is_still_upstream_error(fake_pool, hangs_up, caplog):
    fake_pool.urls["fast"] = hangs_up.url
    caplog.set_level(logging.DEBUG)
    result = await asyncio.wait_for(
        gateway._dispatch_chat(_body(gateway_read_timeout_sec=5), correlation_id="corr-1"), 8)

    assert result["raw"]["error"] == "upstream_error"
    assert result["raw"]["details"]["reason"] == "RemoteProtocolError"
    assert fake_pool.releases == ["upstream_error"]
    assert _lines(caplog, "upstream_cancelled") == []
    (line,) = _lines(caplog, "upstream_failed")
    assert line.levelno == logging.ERROR
    assert "error=upstream_error" in line.getMessage() and "reason=RemoteProtocolError" in line.getMessage()


@pytest.mark.parametrize("executor", ["ollama", "native_completion", "openai_chat"])
def test_every_executor_logs_a_gateway_cancel_as_upstream_cancelled(executor, silent, caplog):
    """All three ``_common_http_client`` call sites, on a real socket the gateway shuts down."""
    caplog.set_level(logging.DEBUG)
    body = _body()
    calls = {
        "ollama": lambda: llm_backend._execute_ollama_chat(body, "m", silent.url, route="quick", served_by="w"),
        "native_completion": lambda: llm_backend._execute_llamacpp_native_completion(
            body, "m", silent.url, "llamacpp", route="quick", served_by="w"),
        "openai_chat": lambda: llm_backend._execute_openai_chat(
            body, "m", silent.url, "llamacpp", route="quick", served_by="w"),
    }
    handle = upstream_cancel.UpstreamCancel()
    threading.Thread(target=lambda: (silent.accepted.wait(3), handle.cancel("caller_budget_exhausted")),
                     daemon=True).start()
    out = upstream_cancel.run_cancellable(handle, calls[executor])

    assert out["raw"]["error"] == "timeout" and out["raw"]["details"]["cancelled_by"] == "gateway"
    assert _lines(caplog, "upstream_failed") == []
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    (line,) = _lines(caplog, "upstream_cancelled")
    assert line.levelno == logging.WARNING and "reason=caller_budget_exhausted" in line.getMessage()
