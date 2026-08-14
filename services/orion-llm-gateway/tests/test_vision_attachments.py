"""Vision capability gate + multimodal assembly.

The load-bearing test here is `test_text_only_turn_is_byte_identical`: the whole
safety argument for this feature is that a turn with no attachments serializes
exactly as it did before attachments existed. Everything else can be iterated
on; that one regressing means chat turns broke.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from app import vision as V
from app.models import ChatBody, ChatMessage
from orion.core.bus.bus_schemas import AttachmentRefV1


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def make_ref(**overrides: Any) -> AttachmentRefV1:
    base = {
        "sha256": "a" * 64,
        "mime": "image/png",
        "bytes": len(PNG_BYTES),
        "width": 64,
        "height": 64,
        "source_url": "http://orion-hub.test/api/chat/attachments/" + "a" * 64,
    }
    base.update(overrides)
    return AttachmentRefV1(**base)


@pytest.fixture(autouse=True)
def _reset_and_allow(monkeypatch):
    V.clear_capability_cache()
    monkeypatch.setattr(V.settings, "llm_gateway_vision_enabled", True, raising=False)
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_allowed_hosts", "orion-hub.test", raising=False
    )
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 1_000_000, raising=False)
    monkeypatch.setattr(V.settings, "llm_gateway_vision_props_cache_ttl_sec", 300.0, raising=False)
    yield
    V.clear_capability_cache()


class FakeResponse:
    def __init__(self, payload=None, body: bytes = b"", status: int = 200):
        self._payload = payload
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


# ───────────────────────────────────────────────────────────────
# The guarantee
# ───────────────────────────────────────────────────────────────

def test_text_only_turn_is_byte_identical():
    """No attachments -> assembly is a pure passthrough of the text serializer."""
    serialized = [
        {"role": "system", "content": "you are orion"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    before = json.dumps(serialized, sort_keys=True)
    result = V.build_multimodal_messages(serialized, [])
    assert json.dumps(result, sort_keys=True) == before


def test_chat_body_defaults_to_no_attachments():
    body = ChatBody(messages=[ChatMessage(role="user", content="hi")])
    assert body.attachments == []


# ───────────────────────────────────────────────────────────────
# Capability gate
# ───────────────────────────────────────────────────────────────

def test_capability_reads_props_modalities(monkeypatch):
    monkeypatch.setattr(
        V.requests, "get", lambda *a, **k: FakeResponse({"modalities": {"vision": True}})
    )
    cap = V.resolve_vision_capability("http://worker:8014")
    assert cap.vision is True
    assert cap.source == "props"


def test_capability_false_when_worker_reports_blind(monkeypatch):
    monkeypatch.setattr(
        V.requests, "get", lambda *a, **k: FakeResponse({"modalities": {"vision": False}})
    )
    cap = V.resolve_vision_capability("http://worker:8011")
    assert cap.vision is False
    assert cap.source == "props"


def test_capability_fails_closed_when_props_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(V.requests, "get", boom)
    cap = V.resolve_vision_capability("http://worker:9999")
    assert cap.vision is False
    assert cap.source == "props-unreachable"


def test_unreachable_probe_is_not_cached(monkeypatch):
    """A transient blip must not blind a sighted worker for a whole TTL."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("blip")
        return FakeResponse({"modalities": {"vision": True}})

    monkeypatch.setattr(V.requests, "get", flaky)
    assert V.resolve_vision_capability("http://worker:8014").vision is False
    assert V.resolve_vision_capability("http://worker:8014").vision is True
    assert calls["n"] == 2


def test_successful_probe_is_cached(monkeypatch):
    calls = {"n": 0}

    def counted(*a, **k):
        calls["n"] += 1
        return FakeResponse({"modalities": {"vision": True}})

    monkeypatch.setattr(V.requests, "get", counted)
    V.resolve_vision_capability("http://worker:8014")
    V.resolve_vision_capability("http://worker:8014")
    assert calls["n"] == 1


def test_master_switch_forces_blind(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_vision_enabled", False, raising=False)

    def should_not_be_called(*a, **k):
        raise AssertionError("probed a worker while vision was globally disabled")

    monkeypatch.setattr(V.requests, "get", should_not_be_called)
    cap = V.resolve_vision_capability("http://worker:8014")
    assert cap.vision is False
    assert cap.source == "disabled"


# ───────────────────────────────────────────────────────────────
# Fetch guards
# ───────────────────────────────────────────────────────────────

def test_fetch_refuses_non_allowlisted_host(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    ref = make_ref(source_url="http://evil.internal/api/chat/attachments/" + "a" * 64)
    with pytest.raises(V.AttachmentFetchError, match="not allowlisted"):
        V.fetch_attachment_data_uri(ref)


def test_fetch_refuses_when_allowlist_is_empty(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_allowed_hosts", "", raising=False)
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    with pytest.raises(V.AttachmentFetchError, match="empty"):
        V.fetch_attachment_data_uri(make_ref())


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher"])
def test_fetch_refuses_non_http_schemes(scheme):
    ref = make_ref(source_url=f"{scheme}://orion-hub.test/etc/passwd")
    with pytest.raises(V.AttachmentFetchError, match="not http"):
        V.fetch_attachment_data_uri(ref)


def test_fetch_refuses_unaccepted_mime():
    with pytest.raises(V.AttachmentFetchError, match="not an accepted image type"):
        V.fetch_attachment_data_uri(make_ref(mime="image/svg+xml"))


def test_fetch_refuses_oversize_declaration(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 100, raising=False)
    with pytest.raises(V.AttachmentFetchError, match="gateway limit"):
        V.fetch_attachment_data_uri(make_ref(bytes=101))


def test_fetch_caps_a_lying_content_length(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 100, raising=False)
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=b"\x00" * 5000))
    with pytest.raises(V.AttachmentFetchError, match="exceeded"):
        V.fetch_attachment_data_uri(make_ref(bytes=50))


def test_fetch_refuses_size_mismatch(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES + b"extra"))
    with pytest.raises(V.AttachmentFetchError, match="size mismatch"):
        V.fetch_attachment_data_uri(make_ref())


def test_fetch_returns_data_uri(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    uri = V.fetch_attachment_data_uri(make_ref())
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES


# ───────────────────────────────────────────────────────────────
# Assembly
# ───────────────────────────────────────────────────────────────

def test_images_attach_to_the_last_user_message(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    serialized = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "look at this"},
    ]
    out = V.build_multimodal_messages(serialized, [make_ref()])

    # Untouched messages keep their original string content.
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "first"}
    assert out[2] == {"role": "assistant", "content": "reply"}

    parts = out[3]["content"]
    assert parts[0] == {"type": "text", "text": "look at this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_multiple_images_all_attach(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    refs = [make_ref(sha256=c * 64) for c in "abc"]
    out = V.build_multimodal_messages([{"role": "user", "content": "three"}], refs)
    assert sum(1 for p in out[0]["content"] if p["type"] == "image_url") == 3


def test_input_list_is_not_mutated(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    serialized = [{"role": "user", "content": "hi"}]
    V.build_multimodal_messages(serialized, [make_ref()])
    assert serialized == [{"role": "user", "content": "hi"}]


def test_all_attachments_failing_raises_rather_than_silently_dropping(monkeypatch):
    def boom(*a, **k):
        raise OSError("store down")

    monkeypatch.setattr(V.requests, "get", boom)
    with pytest.raises(V.AttachmentFetchError):
        V.build_multimodal_messages([{"role": "user", "content": "hi"}], [make_ref()])


def test_partial_failure_keeps_the_usable_images(monkeypatch):
    calls = {"n": 0}

    def half_broken(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("one bad object")
        return FakeResponse(body=PNG_BYTES)

    monkeypatch.setattr(V.requests, "get", half_broken)
    refs = [make_ref(sha256="a" * 64), make_ref(sha256="b" * 64)]
    out = V.build_multimodal_messages([{"role": "user", "content": "hi"}], refs)
    assert sum(1 for p in out[0]["content"] if p["type"] == "image_url") == 1


def test_no_user_message_is_an_error(monkeypatch):
    monkeypatch.setattr(V.requests, "get", lambda *a, **k: FakeResponse(body=PNG_BYTES))
    with pytest.raises(V.AttachmentFetchError, match="no user message"):
        V.build_multimodal_messages([{"role": "system", "content": "sys"}], [make_ref()])
