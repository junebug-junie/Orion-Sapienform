"""Per-call token ledger, gateway producer: one llm.call.usage.v1 per call that reached a model,
with its correlation id, from the bus path and both HTTP passthroughs (streamed and not).

Reply bodies are the real llama.cpp shapes captured 2026-10-29 (orion/llm/tests/fixtures/call_usage).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.schemas.llm_call_usage import LlmCallUsageV1

from app import call_usage_emit
from app import main as gateway
from app.settings import settings

FIX = Path(__file__).resolve().parents[3] / "orion/llm/tests/fixtures/call_usage"


@pytest.fixture(autouse=True)
def _fresh_sink(monkeypatch):
    sink = call_usage_emit.reset_sink_for_tests()
    monkeypatch.setattr(settings, "llm_gateway_call_usage_enabled", True)
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)
    yield sink
    call_usage_emit.reset_sink_for_tests()


def _events() -> list[LlmCallUsageV1]:
    return call_usage_emit.get_sink().drain()


def _env(corr: str, route: str = "chat") -> BaseEnvelope:
    return BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=corr,
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route=route,
                                   options={"verb": "chat_general"}).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_bus_call_records_tokens_and_correlation(fake_pool, monkeypatch):
    raw = json.loads((FIX / "openai_nonstream.json").read_text())

    def run(body, plan):
        return {"text": "Hi", "raw": raw, "route": plan.route, "served_by": plan.route_target.served_by,
                "model": raw["model"]}

    monkeypatch.setattr(gateway, "run_llm_chat", run)
    corr = str(uuid.uuid4())
    await gateway.handle_chat(_env(corr))
    (ev,) = _events()
    assert ev.recorder == "gateway" and ev.path == "bus_rpc" and ev.correlation_id == corr
    assert (ev.prompt_tokens, ev.cached_prompt_tokens, ev.completion_tokens) == (14, 0, 8)
    assert ev.token_source == "usage" and ev.outcome == "served" and ev.route == "chat"
    assert ev.holder == "cortex-exec" and ev.verb == "chat_general" and ev.pool_role
    assert ev.duration_ms is not None and ev.wait_ms is not None and ev.decode_tps > 0
    assert ev.model.endswith(".gguf")


@pytest.mark.asyncio
async def test_pool_refusal_is_not_a_model_call(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    fake_pool.unavailable = "no_serviceable_role"
    await gateway.handle_chat(_env(str(uuid.uuid4()), route="quick"))
    assert _events() == []


@pytest.mark.asyncio
async def test_flag_off_records_nothing(fake_pool, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_call_usage_enabled", False)
    monkeypatch.setattr(gateway, "run_llm_chat", lambda body, plan: {"text": "x", "raw": {"usage": {"prompt_tokens": 1}}})
    await gateway.handle_chat(_env(str(uuid.uuid4())))
    assert _events() == []


@pytest.fixture
def client(monkeypatch, fake_pool):
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    from app.main import app

    return TestClient(app)


def test_openai_passthrough_nonstream_reads_x_orion_correlation_id(client, monkeypatch):
    body = json.loads((FIX / "openai_nonstream.json").read_text())

    async def _post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    r = client.post("/v1/chat/completions", headers={"X-Orion-Correlation-Id": "turn-123"},
                    json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    (ev,) = _events()
    assert ev.path == "http_openai" and ev.correlation_id == "turn-123" and ev.holder == "http:openai"
    assert (ev.prompt_tokens, ev.completion_tokens, ev.stream) == (14, 8, False)


def _stream_client(chunks: list[bytes]) -> MagicMock:
    class _Upstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            for c in chunks:
                yield c

        async def aclose(self) -> None:
            return None

    mock = MagicMock()
    mock.build_request = MagicMock(return_value=MagicMock())
    mock.send = AsyncMock(return_value=_Upstream())
    mock.aclose = AsyncMock()
    return mock


def _split(raw: bytes, n: int = 64) -> list[bytes]:
    return [raw[i:i + n] for i in range(0, len(raw), n)]


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_anthropic_stream_reads_prompt_from_head_and_completion_from_tail(mock_cls, client):
    """The harness's own call shape: Claude Code streams /v1/messages. Before this patch the stream's
    head was never kept, so input_tokens (message_start) could not be read at all."""
    raw = (FIX / "anthropic_stream.sse").read_bytes()
    # pad the middle so head and tail are disjoint windows, as on a real long turn
    pad = b"".join(b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
                   b'"delta":{"type":"text_delta","text":"xxxxxxxxxxxxxxxx"}}\n\n' for _ in range(200))
    start = raw.index(b"event: content_block_start")
    mock_cls.return_value = _stream_client(_split(raw[:start] + pad + raw[start:]))
    with client.stream("POST", "/v1/messages", headers={"X-Orion-Correlation-Id": "turn-9"},
                       json={"model": "llamacpp/agent", "max_tokens": 8, "stream": True,
                             "messages": [{"role": "user", "content": "Say hi"}]}) as resp:
        b"".join(resp.iter_bytes())
    (ev,) = _events()
    assert ev.path == "http_anthropic" and ev.stream and ev.correlation_id == "turn-9"
    assert (ev.prompt_tokens, ev.cached_prompt_tokens, ev.completion_tokens) == (14, 10, 8)
    assert ev.outcome == "served" and ev.holder == "http:anthropic"


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_openai_stream_without_usage_uses_llama_timings(mock_cls, client):
    mock_cls.return_value = _stream_client(_split((FIX / "openai_stream.sse").read_bytes()))
    with client.stream("POST", "/v1/chat/completions",
                       json={"model": "quick", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as resp:
        b"".join(resp.iter_bytes())
    (ev,) = _events()
    assert (ev.prompt_tokens, ev.cached_prompt_tokens, ev.completion_tokens, ev.token_source) == (14, 10, 8, "timings")
    assert ev.correlation_id is None


def test_passthrough_pool_refusal_records_nothing(client, fake_pool):
    fake_pool.unavailable = "no_serviceable_role"
    r = client.post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert _events() == []


def test_upstream_5xx_after_grant_is_recorded_with_no_tokens(client, monkeypatch):
    async def _post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client.post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    (ev,) = _events()
    assert ev.outcome == "upstream_http_5xx" and ev.prompt_tokens is None and ev.token_source == "none"
    assert ev.duration_ms is not None  # the GPU was held: a model call, even if it failed


def test_an_overlong_correlation_header_is_clipped_not_refused(client, monkeypatch):
    """The header now feeds the pool lease's turn_correlation_id (max 128): a longer id must not
    make the lease request fail validation and refuse the call."""
    async def _post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    r = client.post("/v1/chat/completions", headers={"X-Orion-Correlation-Id": "t" * 300},
                    json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    (ev,) = _events()
    assert ev.correlation_id == "t" * 128
