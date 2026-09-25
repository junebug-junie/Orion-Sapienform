"""Gateway self-report: outcome classification, window aggregation, wire format."""

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.schemas.llm_inference_projection import (
    LLM_INFERENCE_SOURCE_SERVICE,
    LLM_INFERENCE_TRACE_PREFIX,
    ROLE_NODE_WINDOW,
    ROLE_WINDOW_COMPLETED,
)

from app import grammar_emit
from app.grammar_emit import (
    InferenceWindowRecorder,
    build_window_events,
    classify_outcome,
    node_hint,
    run_window_publisher,
)


# Each text below is the exact shape one of llm_backend.py's `[Error: ...]` returns
# produces (httpx exception text included where the backend interpolates str(e)).
@pytest.mark.parametrize(
    "result, expected",
    [
        ({"text": "Hello there.", "raw": {}}, "served"),
        ({"text": "the codebase is throwing errors I can't map yet", "raw": {}}, "served"),
        # model prose that the canonical detector alone would flag (review finding)
        ({"text": "Connection refused usually means nothing is listening on that port.", "raw": {}}, "served"),
        ({"text": "Error: that value must be positive -- here's why.", "raw": {}}, "served"),
        ({"text": "A read timeout happens when the server accepts but never answers.", "raw": {}}, "served"),
        ({"text": "[Error: route 'chat' cannot accept images (vision=false)]", "raw": {}}, "request_invalid"),
        ({"text": "[Error: /v1/chat cannot accept image attachments]", "raw": {}}, "request_invalid"),
        ({"text": "", "raw": {"error": "gateway_exception"}}, "gateway_exception"),
        ({"text": "", "reasoning_content": "thinking only", "raw": {}}, "served"),
        ({"text": "", "raw": {}}, "upstream_empty"),
        ({"text": "[Error: llamacpp timed out after waiting]", "raw": {}}, "upstream_timeout"),
        ({"text": "[Error: llamacpp timed out]", "raw": {}}, "upstream_timeout"),
        (
            {"text": "[Error: llamacpp failed: Server error '503 Service Unavailable' for url 'http://x/v1/chat/completions'", "raw": {}},
            "upstream_http_5xx",
        ),
        (
            {"text": "[Error: vllm failed: Client error '400 Bad Request' for url 'http://x/v1/chat/completions'", "raw": {}},
            "upstream_http_4xx",
        ),
        ({"text": "[Error: llamacpp failed: [Errno 111] Connection refused]", "raw": {}}, "upstream_connect"),
        ({"text": "[Error: llamacpp failed: All connection attempts failed]", "raw": {}}, "upstream_connect"),
        ({"text": "[Error: llamacpp 404 Not Found at http://x/v1/chat/completions]", "raw": {}}, "upstream_not_found"),
        ({"text": "[Error: llamacpp /completion 404 at http://x/completion]", "raw": {}}, "upstream_not_found"),
        ({"text": "[Error: llamacpp failed: boom]", "raw": {}}, "upstream_error"),
        ({"text": "[Error: attachments could not be read: bad]", "raw": {}}, "request_invalid"),
        ({"text": "[Error: route 'nope' not configured]", "raw": {"error": "route_not_configured"}}, "route_not_configured"),
        ({"text": "", "raw": {"error": "gateway_overloaded"}}, "gateway_overloaded"),
        ({"text": "", "raw": {"error": "route_operator_closed"}}, "route_operator_closed"),
        ({"text": "", "raw": {"error": "gateway_capacity_rejected"}}, "gateway_capacity_rejected"),
        ({"text": "", "raw": {"error": "llm_route_unavailable"}}, "llm_route_unavailable"),
        ({"text": "", "raw": {"error": "something_new"}}, "upstream_error"),
        ("not a dict", "upstream_error"),
    ],
)
def test_classify_outcome(result, expected):
    assert classify_outcome(result) == expected


def test_source_service_literal_matches_the_contract():
    assert grammar_emit.SOURCE_SERVICE == LLM_INFERENCE_SOURCE_SERVICE


def test_node_hint_follows_worker_label_convention():
    assert node_hint("circe-worker-2") == "circe"
    assert node_hint("circe-worker-fast-1") == "circe"
    assert node_hint("Circe") == "circe"
    assert node_hint(None) == "unrouted"
    assert node_hint("  ") == "unrouted"


def _kv(summary: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in summary.split() if "=" in part)


def test_window_groups_by_node_and_counts_only_upstream_failures():
    t = [1000.0]
    rec = InferenceWindowRecorder(clock=lambda: t[0])
    ok = {"text": "fine", "raw": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}
    rec.record(ok, served_by="circe-worker-2", elapsed_s=0.5)
    rec.record(ok, served_by="circe-worker-fast-1", elapsed_s=1.5)
    rec.record({"text": "[Error: llamacpp timed out after waiting]", "raw": {}}, served_by="circe-worker-2", elapsed_s=30)
    rec.record({"text": "", "raw": {"error": "gateway_overloaded"}}, served_by="circe-worker-2", elapsed_s=3)
    rec.record({"text": "", "raw": {"error": "route_operator_closed"}}, served_by=None, elapsed_s=0)
    t[0] = 1060.0
    start, end, buckets = rec.drain()
    assert (start, end) == (1000.0, 1060.0)
    events = build_window_events(gateway_node="athena", window_start=start, window_end=end, buckets=buckets)

    assert all(e.trace_id == f"{LLM_INFERENCE_TRACE_PREFIX}athena:19700101T001640Z" for e in events)
    assert all(e.provenance.source_service == LLM_INFERENCE_SOURCE_SERVICE for e in events)
    roles = [e.atom.semantic_role for e in events]
    assert roles == [ROLE_NODE_WINDOW, ROLE_NODE_WINDOW, ROLE_WINDOW_COMPLETED]

    circe = _kv(events[0].atom.summary)
    assert circe["node"] == "circe"
    assert circe["calls"] == "4"
    assert circe["served"] == "2"
    assert circe["upstream_failed"] == "1"  # the timeout
    assert circe["refused"] == "1"  # overloaded is the gateway's doing, not the backend's
    assert circe["p50_ms"] in {"500", "1500"}
    assert circe["prompt_tokens"] == "200"
    assert circe["completion_tokens"] == "40"
    assert circe["workers"] == "circe-worker-2|circe-worker-fast-1"
    assert "upstream_timeout:1" in circe["classes"]

    unrouted = _kv(events[1].atom.summary)
    assert unrouted["node"] == "unrouted" and unrouted["refused"] == "1"

    done = _kv(events[2].atom.summary)
    assert done == {"gateway": "athena", "calls": "5", "nodes": "2", "window_sec": "60.0"}

    # drained: the next window starts empty
    t[0] = 1120.0
    _, _, again = rec.drain()
    assert again == {}


def test_empty_window_still_emits_completed_atom():
    events = build_window_events(gateway_node="", window_start=0.0, window_end=60.0, buckets={})
    assert len(events) == 1
    assert events[0].atom.semantic_role == ROLE_WINDOW_COMPLETED
    assert _kv(events[0].atom.summary)["calls"] == "0"
    assert events[0].trace_id.startswith(f"{LLM_INFERENCE_TRACE_PREFIX}gateway:")


def test_no_prompt_or_reply_text_leaves_the_gateway():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    secret_reply = "SECRET-REPLY-TEXT"
    rec.record({"text": secret_reply, "raw": {}}, served_by="circe-worker-2", elapsed_s=1)
    rec.record({"text": "[Error: llamacpp failed: SECRET-ERR-DETAIL]", "raw": {}}, served_by="circe-worker-2", elapsed_s=1)
    _, _, buckets = rec.drain()
    dumped = "".join(
        e.model_dump_json() for e in build_window_events(gateway_node="athena", window_start=0, window_end=1, buckets=buckets)
    )
    assert "SECRET" not in dumped


def test_latency_samples_are_bounded():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    for i in range(2000):
        rec.record({"text": "ok", "raw": {}}, served_by="circe-worker-2", elapsed_s=i / 1000)
    _, _, buckets = rec.drain()
    assert len(buckets["circe"].served_latency_ms) == 512
    assert buckets["circe"].calls == 2000


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


@pytest.mark.asyncio
async def test_publisher_flushes_a_window_onto_the_grammar_channel():
    grammar_emit.reset_recorder_for_tests()
    grammar_emit.get_recorder().record({"text": "ok", "raw": {}}, served_by="circe-worker-2", elapsed_s=0.2)
    bus = _FakeBus()
    stop = asyncio.Event()

    async def _stop_soon():
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(
        run_window_publisher(bus, gateway_node="athena", window_sec=60, stop=stop),
        _stop_soon(),
    )
    channels = {c for c, _ in bus.published}
    assert channels == {"orion:grammar:event"}
    kinds = [env.payload["atom"]["semantic_role"] for _, env in bus.published]
    assert kinds == [ROLE_NODE_WINDOW, ROLE_WINDOW_COMPLETED]
    grammar_emit.reset_recorder_for_tests()


@pytest.mark.asyncio
async def test_publisher_survives_a_bus_failure():
    grammar_emit.reset_recorder_for_tests()

    class _Broken:
        async def publish(self, channel, envelope):
            raise RuntimeError("bus down")

    stop = asyncio.Event()

    async def _stop_soon():
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(run_window_publisher(_Broken(), gateway_node="athena", window_sec=60, stop=stop), _stop_soon())
    grammar_emit.reset_recorder_for_tests()


def _req():
    return BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name="test", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(
            messages=[LLMMessage(role="user", content="ping")],
            route="quick",
        ).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_handle_chat_records_outcome_only_when_enabled():
    from app import main

    grammar_emit.reset_recorder_for_tests()
    fake = {"text": "[Error: llamacpp timed out after waiting]", "raw": {}, "served_by": "circe-worker-fast-1"}

    async def _dispatch(body, *, correlation_id):
        return dict(fake)

    with patch.object(main, "_dispatch_chat", _dispatch), patch.object(main.settings, "llm_gateway_grammar_enabled", False):
        await main.handle_chat(_req())
    assert grammar_emit.get_recorder().drain()[2] == {}

    with patch.object(main, "_dispatch_chat", _dispatch), patch.object(main.settings, "llm_gateway_grammar_enabled", True):
        out = await main.handle_chat(_req())
    buckets = grammar_emit.get_recorder().drain()[2]
    assert buckets["circe"].classes == {"upstream_timeout": 1}
    # the reply itself is untouched
    assert out.payload.content == fake["text"]
    grammar_emit.reset_recorder_for_tests()


@pytest.mark.asyncio
async def test_handle_chat_reply_survives_a_recorder_crash():
    from app import main

    async def _dispatch(body, *, correlation_id):
        return {"text": "hi", "raw": {}, "served_by": "circe-worker-2"}

    class _Boom:
        def record(self, *a, **k):
            raise RuntimeError("boom")

    with patch.object(main, "_dispatch_chat", _dispatch), patch.object(
        main.settings, "llm_gateway_grammar_enabled", True
    ), patch.object(grammar_emit, "get_recorder", lambda: _Boom()):
        out = await main.handle_chat(_req())
    assert out.payload.content == "hi"
