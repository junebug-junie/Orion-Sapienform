"""The vision gate inside _execute_openai_chat, not just its helpers.

Acceptance check 4 of the design spec: an image sent at a blind route must
produce an explicit refusal and must NEVER be silently answered as text. That
guarantee lives in the branch inside _execute_openai_chat, so it has to be
tested there -- helper-level tests cannot prove the caller consults them.
"""

from __future__ import annotations

import json

import pytest

from app import llm_backend as B
from app import vision as V
from app.models import ChatBody, ChatMessage
from orion.core.bus.bus_schemas import AttachmentRefV1


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def ref() -> AttachmentRefV1:
    return AttachmentRefV1(
        sha256="f" * 64,
        mime="image/png",
        bytes=len(PNG_BYTES),
        width=8,
        height=8,
        source_url="http://orion-hub.test/api/chat/attachments/" + "f" * 64,
    )


def body(with_image: bool) -> ChatBody:
    return ChatBody(
        messages=[ChatMessage(role="user", content="what is this?")],
        attachments=[ref()] if with_image else [],
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    V.clear_capability_cache()
    monkeypatch.setattr(V.settings, "llm_gateway_vision_enabled", True, raising=False)
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_allowed_hosts", "orion-hub.test", raising=False
    )
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 1_000_000, raising=False)
    # Keep Spark out of the way; it is orthogonal to the gate.
    monkeypatch.setattr(B, "_spark_ingest_for_body", lambda *a, **k: {})
    monkeypatch.setattr(B, "_spark_post_ingest_for_reply", lambda *a, **k: None)
    monkeypatch.setattr(B, "_maybe_publish_spark_introspect", lambda *a, **k: None)
    yield
    V.clear_capability_cache()


class _Resp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "red"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }


class _Client:
    """Captures the outbound payload instead of talking to a worker."""

    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, **kw):
        self.sink["url"] = url
        self.sink["payload"] = json
        return _Resp()


def run(monkeypatch, *, vision: bool, with_image: bool):
    sink: dict = {}
    monkeypatch.setattr(B, "_common_http_client", lambda b: _Client(sink))
    monkeypatch.setattr(
        B, "resolve_vision_capability",
        lambda base_url: V.VisionCapability(
            vision=vision, source="props",
            detail=None if vision else "worker reports modalities.vision=false",
        ),
    )
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: type(
        "R", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "iter_content": lambda self, chunk_size=65536: iter([PNG_BYTES]),
        },
    )())
    result = B._execute_openai_chat(
        body(with_image), "test-model", "http://worker:8080", "llamacpp", route="chat",
    )
    return result, sink


# ── The guarantee ────────────────────────────────────────────────────────

def test_image_at_a_blind_route_is_refused_not_silently_answered(monkeypatch):
    result, sink = run(monkeypatch, vision=False, with_image=True)
    assert "cannot accept images" in result["text"]
    assert result["vision"]["status"] == "refused"
    assert result["vision"]["vision"] is False
    # The decisive assertion: no request was made at all. A silent text-only
    # answer would show up here as a populated payload.
    assert "payload" not in sink, "sent a text-only request after refusing an image"


def test_image_at_a_sighted_route_becomes_content_parts(monkeypatch):
    result, sink = run(monkeypatch, vision=True, with_image=True)
    assert result["vision"]["status"] == "attached"
    content = sink["payload"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unreadable_attachment_fails_loudly_rather_than_dropping_the_image(monkeypatch):
    sink: dict = {}
    monkeypatch.setattr(B, "_common_http_client", lambda b: _Client(sink))
    monkeypatch.setattr(
        B, "resolve_vision_capability",
        lambda base_url: V.VisionCapability(vision=True, source="props"),
    )

    def boom(*a, **k):
        raise OSError("store unreachable")

    monkeypatch.setattr(V.requests, "get", boom)
    result = B._execute_openai_chat(
        body(True), "test-model", "http://worker:8080", "llamacpp", route="chat",
    )
    assert "attachments could not be read" in result["text"]
    assert result["vision"]["status"] == "fetch_failed"
    assert "payload" not in sink


# ── The no-regression guarantee ──────────────────────────────────────────

def test_text_only_turn_is_untouched_by_the_gate(monkeypatch):
    """No attachments -> no probe, no vision key, plain string content."""
    def should_not_probe(*a, **k):
        raise AssertionError("probed vision capability on a text-only turn")

    monkeypatch.setattr(B, "resolve_vision_capability", should_not_probe)
    result, sink = run(monkeypatch, vision=True, with_image=False)

    assert result["vision"] is None
    assert "vision" not in (result.get("raw") or {})
    assert sink["payload"]["messages"] == [{"role": "user", "content": "what is this?"}]


def test_text_only_payload_matches_the_pre_attachment_shape(monkeypatch):
    """Golden check on the outbound payload keys for a text-only turn."""
    _, sink = run(monkeypatch, vision=True, with_image=False)
    payload = sink["payload"]
    assert json.dumps(payload["messages"]) == json.dumps(
        [{"role": "user", "content": "what is this?"}]
    )
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
