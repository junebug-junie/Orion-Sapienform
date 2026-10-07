"""An upstream worker that answers with an error must reach the caller as a failure.

Live 2026-10-02..06: the gpu2 Bonsai worker (circe:8016) answered 44/44 agent-lane
/v1/chat/completions calls with HTTP 500 ``No user query found in messages`` (a Jinja chat
template error). The gateway logged the 500 but replied ``llm.chat.result`` with ``raw={}`` and
the generic ``[Error: llamacpp failed: Server error '500 ...']`` text -- no ``raw.error``, and the
worker's own message dropped -- so callers keyed on ``raw.error`` recorded a success.

These tests drive the real request path (run_llm_chat -> _execute_openai_chat /
_execute_llamacpp_native_completion) against a fake upstream served by httpx.MockTransport.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List

import httpx
import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from orion.schemas.gpu_pool import GpuLeaseRefV1

from app import grammar_emit, upstream_cancel
from app import main as gateway
from app.llm_backend import (
    UPSTREAM_ERROR_MESSAGE_MAX_CHARS,
    _error_body_message,
    _execute_llamacpp_native_completion,
    _execute_openai_chat,
)
from app.models import ChatBody
from app.settings import settings

JINJA_500 = {
    "error": {
        "code": 500,
        "message": "No user query found in messages.",
        "type": "server_error",
    }
}


def _transport(monkeypatch, handler: Callable[[httpx.Request], httpx.Response]) -> List[str]:
    seen: List[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return handler(request)

    monkeypatch.setattr(upstream_cancel, "cancellable_transport", lambda: httpx.MockTransport(wrapped))
    return seen


def _body(**options: Any) -> ChatBody:
    return ChatBody(route="agent", messages=[{"role": "user", "content": "hi"}], options=options,
                    trace_id="corr-test")


@pytest.fixture(autouse=True)
def _no_lane_routing(monkeypatch):
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)


# ── the backend call itself ────────────────────────────────────────────────────────────────


def test_a_500_with_the_jinja_body_is_a_typed_failure_carrying_the_workers_message(monkeypatch):
    _transport(monkeypatch, lambda req: httpx.Response(500, json=JINJA_500))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp",
                                  route="agent", served_by="circe-worker-agent-gpu2")
    assert result["raw"]["error"] == "upstream_http_5xx"
    details = result["raw"]["details"]
    assert details["status_code"] == 500
    assert details["message"] == "No user query found in messages."
    assert details["served_by"] == "circe-worker-agent-gpu2"
    assert details["reason"] == "http_500"
    # Empty text: the gateway's standard failure shape. cortex-exec's gateway_error_step_failure
    # only fails a step whose content is empty -- error text in content passed as a success.
    assert result["text"] == ""


def test_a_non_json_5xx_body_keeps_its_text(monkeypatch):
    _transport(monkeypatch, lambda req: httpx.Response(502, text="bad gateway from proxy"))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert result["raw"]["error"] == "upstream_http_5xx"
    assert result["raw"]["details"]["message"] == "bad gateway from proxy"


def test_the_upstream_message_is_truncated(monkeypatch):
    huge = {"error": {"message": "x" * 10_000}}
    _transport(monkeypatch, lambda req: httpx.Response(500, json=huge))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert len(result["raw"]["details"]["message"]) == UPSTREAM_ERROR_MESSAGE_MAX_CHARS


@pytest.mark.parametrize("status,klass", [(400, "upstream_http_4xx"), (404, "upstream_not_found"),
                                          (503, "upstream_http_5xx")])
def test_status_classes_match_the_telemetry_contract(monkeypatch, status, klass):
    _transport(monkeypatch, lambda req: httpx.Response(status, json={"error": {"message": "nope"}}))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert result["raw"]["error"] == klass


def test_a_200_whose_body_is_an_error_is_a_failure_not_a_blank_answer(monkeypatch):
    _transport(monkeypatch, lambda req: httpx.Response(200, json=JINJA_500))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert result["raw"]["error"] == "upstream_error"
    assert result["raw"]["details"]["message"] == "No user query found in messages."


def test_a_real_completion_is_untouched(monkeypatch):
    ok = {"model": "bonsai.gguf", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    _transport(monkeypatch, lambda req: httpx.Response(200, json=ok))
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert result["text"] == "hello"
    assert "error" not in result["raw"]


def test_error_body_detection_is_narrow():
    assert _error_body_message({"choices": [{"message": {"content": "x"}}], "error": "x"}) is None
    assert _error_body_message({"content": "x", "error": "warn"}) is None
    assert _error_body_message({}) is None
    assert _error_body_message({"error": "boom"}) == "boom"


@pytest.mark.parametrize("fail_path", ["/apply-template", "/completion"])
def test_native_completion_path_reports_the_same_typed_failure(monkeypatch, fail_path):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == fail_path:
            return httpx.Response(500, json=JINJA_500)
        if req.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<|user|>hi"})
        return httpx.Response(200, json={"content": "hello"})

    _transport(monkeypatch, handler)
    result = _execute_llamacpp_native_completion(_body(), "m", "http://w:8016", "llamacpp",
                                                 route="agent", served_by="w")
    assert result["raw"]["error"] == "upstream_http_5xx"
    assert result["raw"]["details"]["message"] == "No user query found in messages."
    assert result["raw"]["details"]["url"].endswith(fail_path)


def test_native_completion_200_error_body_is_a_failure(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<|user|>hi"})
        return httpx.Response(200, json={"error": {"message": "slot unavailable"}})

    _transport(monkeypatch, handler)
    result = _execute_llamacpp_native_completion(_body(), "m", "http://w:8016", "llamacpp",
                                                 route="agent", served_by="w")
    assert result["raw"]["error"] == "upstream_error"


# ── the whole bus reply: lease, telemetry, payload ─────────────────────────────────────────


def _env(route: str = "agent", gpu_lease: Dict[str, Any] | None = None) -> BaseEnvelope:
    options: Dict[str, Any] = {}
    return BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route=route,
                                   options=options).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_bus_reply_for_a_worker_500_is_a_failure_the_caller_can_see(fake_pool, monkeypatch):
    """THE REGRESSION. Before: payload.raw == {} and the worker's message was gone."""
    seen = _transport(monkeypatch, lambda req: httpx.Response(500, json=JINJA_500))
    recorded: List[str] = []
    monkeypatch.setattr(settings, "llm_gateway_grammar_enabled", True)

    class _Rec:
        def record(self, result, *, served_by=None, timing=None):
            recorded.append(grammar_emit.classify_outcome(result))

    monkeypatch.setattr(grammar_emit, "get_recorder", lambda: _Rec())

    out = await gateway.handle_chat(_env("agent"))
    payload = out.payload
    assert payload.raw["error"] == "upstream_http_5xx"
    assert payload.raw["details"]["message"] == "No user query found in messages."
    assert payload.content == ""
    # Pool: released as an upstream failure, with the worker's words as the reason.
    assert fake_pool.releases == ["upstream_error"]
    assert fake_pool.release_details == [
        "_UpstreamFailed: upstream_http_5xx: No user query found in messages."
    ]
    # Telemetry (stage 6.2): an upstream failure class, never "served".
    assert recorded == ["upstream_http_5xx"]
    # No silent retry: one lease, one upstream call.
    assert len(fake_pool.calls) == 1
    assert seen == ["/v1/chat/completions"]


@pytest.mark.asyncio
async def test_held_lease_child_call_fails_the_same_way(fake_pool, monkeypatch):
    """Durable runs attach their child calls to a hold; the failure shape is the same and the
    child's release (not the hold) records the upstream failure."""
    _transport(monkeypatch, lambda req: httpx.Response(500, json=JINJA_500))
    ref = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="agent", holder="durable-runs:run-1")
    fake_pool.add_hold(ref)
    body = ChatBody(route="agent", messages=[{"role": "user", "content": "hi"}],
                    options={"gpu_lease": ref.model_dump(mode="json")}, trace_id="c")
    result = await gateway._dispatch_chat(body, correlation_id="c")
    assert result["raw"]["error"] == "upstream_http_5xx"
    assert fake_pool.releases == ["upstream_error"]
    assert len(fake_pool.calls) == 1


def test_classify_outcome_passes_typed_upstream_classes_through():
    for klass in ("upstream_http_5xx", "upstream_not_found", "upstream_timeout"):
        assert grammar_emit.classify_outcome({"text": "[Error: x]", "raw": {"error": klass}}) in (
            klass, "upstream_timeout")
    assert grammar_emit.classify_outcome({"text": "x", "raw": {"error": "upstream_http_4xx"}}) == "upstream_http_4xx"


@pytest.mark.parametrize("exc,klass", [
    (httpx.ConnectError("All connection attempts failed"), "upstream_connect"),
    (httpx.ReadTimeout("timed out"), "upstream_timeout"),
])
def test_no_answer_at_all_is_the_same_failure_shape(monkeypatch, exc, klass):
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    _transport(monkeypatch, handler)
    result = _execute_openai_chat(_body(), "m", "http://w:8016", "llamacpp", route="agent", served_by="w")
    assert result["text"] == ""
    assert result["raw"]["error"] == klass
    assert result["raw"]["details"]["reason"] == type(exc).__name__
    assert grammar_emit.classify_outcome(result) == klass


def test_apply_template_empty_prompt_is_a_failure(monkeypatch):
    _transport(monkeypatch, lambda req: httpx.Response(200, json={"prompt": ""}))
    result = _execute_llamacpp_native_completion(_body(), "m", "http://w:8016", "llamacpp",
                                                 route="agent", served_by="w")
    assert result["text"] == ""
    assert result["raw"]["error"] == "upstream_error"
    assert result["raw"]["details"]["reason"] == "empty_prompt"
