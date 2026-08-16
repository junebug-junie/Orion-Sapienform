"""The vision gate inside the backend executors, not just its helpers.

Acceptance check 4 of the design spec: an image sent at a blind route must
produce an explicit refusal and must NEVER be silently answered as text. That
guarantee lives in a branch inside `_execute_openai_chat`, so it has to be
tested there -- helper-level tests cannot prove the caller consults them.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

import pytest

from app import llm_backend as B
from app import vision as V
from app.models import ChatBody, ChatMessage
from orion.core.bus.bus_schemas import AttachmentRefV1


def make_png(width: int = 8, height: int = 8) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xdc\x14\x14" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


PNG_BYTES = make_png()
PNG_SHA = hashlib.sha256(PNG_BYTES).hexdigest()
BASE_URL = "http://orion-hub.test:8080/api/chat/attachments"


def ref() -> AttachmentRefV1:
    return AttachmentRefV1(
        sha256=PNG_SHA,
        mime="image/png",
        bytes=len(PNG_BYTES),
        width=8,
        height=8,
        source_url=f"{BASE_URL}/{PNG_SHA}",
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
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_base_url", BASE_URL, raising=False)
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_allowed_hosts", "orion-hub.test", raising=False
    )
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 1_000_000, raising=False)
    # Spark is orthogonal to the gate; keep it out of the way.
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
    """Captures the outbound model payload instead of talking to a worker."""

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


class _FetchStream:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size=65536):
        yield self._data


def install_attachment_fetch(monkeypatch, *, status: int = 200):
    class _HttpxClient:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _FetchStream(PNG_BYTES, status)

    monkeypatch.setattr(V.httpx, "Client", _HttpxClient)


def run(monkeypatch, *, vision: bool, with_image: bool, fetch_status: int = 200):
    sink: dict = {}
    monkeypatch.setattr(B, "_common_http_client", lambda b: _Client(sink))
    monkeypatch.setattr(
        B, "resolve_vision_capability",
        lambda base_url: V.VisionCapability(
            vision=vision, source="props",
            detail=None if vision else "worker reports modalities.vision=false",
        ),
    )
    install_attachment_fetch(monkeypatch, status=fetch_status)
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


def test_refusal_diagnostic_reaches_the_hub_via_raw(monkeypatch):
    """handle_chat forwards result['raw'] plus an explicit key list.

    'vision' is not on that list, so a top-level-only diagnostic is dropped
    before it reaches the Hub -- on exactly the path where it matters most.
    """
    result, _ = run(monkeypatch, vision=False, with_image=True)
    assert result["raw"]["vision"]["status"] == "refused"


def test_image_at_a_sighted_route_becomes_content_parts(monkeypatch):
    result, sink = run(monkeypatch, vision=True, with_image=True)
    assert result["vision"]["status"] == "attached"
    content = sink["payload"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unreadable_attachment_fails_loudly_rather_than_dropping_the_image(monkeypatch):
    result, sink = run(monkeypatch, vision=True, with_image=True, fetch_status=500)
    assert "attachments could not be read" in result["text"]
    assert result["vision"]["status"] == "fetch_failed"
    assert result["raw"]["vision"]["status"] == "fetch_failed"
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
    """Golden check on the outbound payload for a text-only turn."""
    _, sink = run(monkeypatch, vision=True, with_image=False)
    payload = sink["payload"]
    assert json.dumps(payload["messages"]) == json.dumps(
        [{"role": "user", "content": "what is this?"}]
    )
    assert payload["model"] == "test-model"
    assert payload["stream"] is False


# ── Backend paths that cannot carry images must refuse, not drop ─────────

@pytest.mark.parametrize(
    "executor,label",
    [
        (lambda b: B._execute_ollama_chat(b, "m", "http://o:11434"), "ollama"),
        (
            lambda b: B._execute_llamacpp_native_completion(b, "m", "http://w:8080", "llamacpp"),
            "native completion",
        ),
    ],
)
def test_non_openai_paths_refuse_attachments(monkeypatch, executor, label):
    """Only the OpenAI path builds content parts.

    Ollama uses its own images:[b64] field and llama.cpp's /completion takes a
    flat prompt string. Neither is wired, so both must refuse rather than answer
    from the text alone -- dropping the image and replying anyway is the exact
    'Orion appears to have looked' failure this feature exists to prevent.
    """
    def should_not_be_called(*a, **k):
        raise AssertionError(f"{label} sent a request despite carrying attachments")

    monkeypatch.setattr(B, "_common_http_client", should_not_be_called)
    result = executor(body(True))
    assert "cannot accept image attachments" in result["text"]
    assert result["vision"]["status"] == "refused"
    assert result["vision"]["source"] == "unsupported-backend-path"
    assert result["raw"]["vision"]["status"] == "refused"


def test_non_openai_paths_are_unaffected_without_attachments():
    """The guard must be inert on every text-only turn."""
    assert B._refuse_attachments_on_unsupported_path(body(False), "ollama") is None
