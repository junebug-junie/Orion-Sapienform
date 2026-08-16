"""Vision capability gate + attachment fetch + multimodal assembly.

The load-bearing test is `test_text_only_turn_is_byte_identical`: the whole
safety argument for this feature is that a turn with no attachments serializes
exactly as it did before attachments existed.

Fixtures use REAL png bytes and their REAL sha256, because the fetch path now
verifies the content address. A fixture with a made-up digest would make every
success-path test fail for the wrong reason.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from typing import Any

import pytest

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


def make_ref(**overrides: Any) -> AttachmentRefV1:
    base = {
        "sha256": PNG_SHA,
        "mime": "image/png",
        "bytes": len(PNG_BYTES),
        "width": 8,
        "height": 8,
        # Deliberately a hostile URL: the gateway must ignore it entirely and
        # rebuild the fetch URL from the sha and its own trusted base.
        "source_url": "http://100.92.216.81:8096/internal/secrets",
    }
    base.update(overrides)
    return AttachmentRefV1(**base)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    V.clear_capability_cache()
    monkeypatch.setattr(V.settings, "llm_gateway_vision_enabled", True, raising=False)
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_base_url", BASE_URL, raising=False)
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_allowed_hosts", "orion-hub.test", raising=False
    )
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 1_000_000, raising=False)
    monkeypatch.setattr(V.settings, "llm_gateway_vision_props_cache_ttl_sec", 300.0, raising=False)
    yield
    V.clear_capability_cache()


# ───────────────────────────────────────────────────────────────
# httpx doubles
# ───────────────────────────────────────────────────────────────

class FakeStream:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class FakeGet:
    def __init__(self, payload=None, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def install_client(monkeypatch, *, body: bytes = PNG_BYTES, props=None, capture=None,
                   stream_status: int = 200, raise_on_get=None):
    """Patch httpx.Client with a double, recording how it was constructed."""

    class _Client:
        def __init__(self, **kwargs):
            if capture is not None:
                capture.setdefault("client_kwargs", []).append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            if capture is not None:
                capture["get_url"] = url
            if raise_on_get:
                raise raise_on_get
            return FakeGet(props)

        def stream(self, method, url):
            if capture is not None:
                capture["stream_url"] = url
                capture["stream_method"] = method
            return FakeStream(body, stream_status)

    monkeypatch.setattr(V.httpx, "Client", _Client)


# ───────────────────────────────────────────────────────────────
# The guarantee
# ───────────────────────────────────────────────────────────────

def test_text_only_turn_is_byte_identical():
    serialized = [
        {"role": "system", "content": "you are orion"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    before = json.dumps(serialized, sort_keys=True)
    result = V.build_multimodal_messages(serialized, [])
    assert json.dumps(result, sort_keys=True) == before


def test_chat_body_defaults_to_no_attachments():
    assert ChatBody(messages=[ChatMessage(role="user", content="hi")]).attachments == []


# ───────────────────────────────────────────────────────────────
# Capability gate
# ───────────────────────────────────────────────────────────────

def test_capability_reads_props_modalities(monkeypatch):
    install_client(monkeypatch, props={"modalities": {"vision": True}})
    cap = V.resolve_vision_capability("http://worker:8014")
    assert cap.vision is True and cap.source == "props"


def test_capability_false_when_worker_reports_blind(monkeypatch):
    install_client(monkeypatch, props={"modalities": {"vision": False}})
    cap = V.resolve_vision_capability("http://worker:8011")
    assert cap.vision is False and cap.source == "props"


def test_capability_fails_closed_when_props_unreachable(monkeypatch):
    install_client(monkeypatch, raise_on_get=OSError("connection refused"))
    cap = V.resolve_vision_capability("http://worker:9999")
    assert cap.vision is False and cap.source == "props-unreachable"


def test_unreachable_probe_is_not_cached(monkeypatch):
    """A transient blip must not blind a sighted worker for a whole TTL."""
    calls = {"n": 0}

    class _Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("blip")
            return FakeGet({"modalities": {"vision": True}})

    monkeypatch.setattr(V.httpx, "Client", _Client)
    assert V.resolve_vision_capability("http://worker:8014").vision is False
    assert V.resolve_vision_capability("http://worker:8014").vision is True
    assert calls["n"] == 2


def test_successful_probe_is_cached(monkeypatch):
    cap = {}
    install_client(monkeypatch, props={"modalities": {"vision": True}}, capture=cap)
    V.resolve_vision_capability("http://worker:8014")
    V.resolve_vision_capability("http://worker:8014")
    assert len(cap["client_kwargs"]) == 1


def test_master_switch_forces_blind(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_vision_enabled", False, raising=False)

    def boom(**kw):
        raise AssertionError("probed a worker while vision was globally disabled")

    monkeypatch.setattr(V.httpx, "Client", boom)
    cap = V.resolve_vision_capability("http://worker:8014")
    assert cap.vision is False and cap.source == "disabled"


# ───────────────────────────────────────────────────────────────
# URL derivation — the SSRF control
# ───────────────────────────────────────────────────────────────

def test_url_is_derived_from_the_trusted_base_not_the_ref():
    """The ref's source_url points at an internal service; it must be ignored."""
    ref = make_ref()
    assert "8096" in ref.source_url and "secrets" in ref.source_url
    assert V.resolve_attachment_url(ref) == f"{BASE_URL}/{PNG_SHA}"


@pytest.mark.parametrize(
    "hostile_url",
    [
        "http://100.92.216.81:8096/internal/secrets",
        "http://100.92.216.81:5432/",
        "file:///etc/passwd",
        "http://evil.example/x",
        "http://orion-hub.test:8080/api/chat/attachments/../../etc/passwd",
    ],
)
def test_a_hostile_source_url_cannot_influence_the_fetch(hostile_url):
    """No source_url value changes where the gateway actually goes."""
    assert V.resolve_attachment_url(make_ref(source_url=hostile_url)) == f"{BASE_URL}/{PNG_SHA}"


def _bare_ref(sha: str):
    """A duck-typed ref, so malformed shas can be tested past pydantic."""
    return type("R", (), {"sha256": sha, "mime": "image/png", "bytes": 1,
                          "source_url": "http://x/y"})()


@pytest.mark.parametrize(
    "bad_sha",
    ["../../etc/passwd", "a" * 63, "a" * 65, "g" * 64, "", "a" * 32 + "/" + "b" * 31,
     "%2e%2e%2f" + "a" * 55],
)
def test_a_non_hex_sha_is_refused(bad_sha):
    """The only caller-supplied component of the URL is regex-validated.

    This is what makes the derived URL safe: nothing that could introduce a
    path segment, an authority, or a traversal gets appended to the base.
    """
    with pytest.raises(V.AttachmentFetchError, match="sha256"):
        V.resolve_attachment_url(_bare_ref(bad_sha))


def test_uppercase_hex_is_normalized_not_rejected():
    """Hex is case-insensitive; the store keys on the lowercase digest."""
    url = V.resolve_attachment_url(_bare_ref(PNG_SHA.upper()))
    assert url == f"{BASE_URL}/{PNG_SHA}"


def test_missing_base_url_refuses_everything(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_base_url", "", raising=False)
    with pytest.raises(V.AttachmentFetchError, match="BASE_URL"):
        V.resolve_attachment_url(make_ref())


def test_base_url_host_must_still_be_allowlisted(monkeypatch):
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_base_url",
        "http://somewhere-else.test/api/chat/attachments", raising=False,
    )
    with pytest.raises(V.AttachmentFetchError, match="not allowlisted"):
        V.resolve_attachment_url(make_ref())


def test_empty_allowlist_refuses_everything(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_allowed_hosts", "", raising=False)
    with pytest.raises(V.AttachmentFetchError, match="empty"):
        V.resolve_attachment_url(make_ref())


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher"])
def test_non_http_base_scheme_is_refused(monkeypatch, scheme):
    monkeypatch.setattr(
        V.settings, "llm_gateway_attachment_base_url",
        f"{scheme}://orion-hub.test/attachments", raising=False,
    )
    with pytest.raises(V.AttachmentFetchError, match="not http"):
        V.resolve_attachment_url(make_ref())


# ───────────────────────────────────────────────────────────────
# Fetch guards
# ───────────────────────────────────────────────────────────────

def test_fetch_returns_data_uri_and_hits_the_derived_url(monkeypatch):
    cap = {}
    install_client(monkeypatch, capture=cap)
    uri = V.fetch_attachment_data_uri(make_ref())
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES
    assert cap["stream_url"] == f"{BASE_URL}/{PNG_SHA}"


def test_fetch_does_not_follow_redirects(monkeypatch):
    """An allowlisted host that 302s must not carry the fetch elsewhere."""
    cap = {}
    install_client(monkeypatch, capture=cap)
    V.fetch_attachment_data_uri(make_ref())
    assert cap["client_kwargs"][0]["follow_redirects"] is False


def test_fetch_refuses_content_that_does_not_match_the_address(monkeypatch):
    """The strongest check: content-addressed means the bytes must hash right."""
    install_client(monkeypatch, body=make_png(9, 9))
    with pytest.raises(V.AttachmentFetchError, match="content address mismatch"):
        V.fetch_attachment_data_uri(make_ref())


def test_fetch_refuses_bytes_that_do_not_sniff_as_the_declared_mime(monkeypatch):
    """Declared mime is what reaches the model, so it is not taken on trust."""
    body = b"GIF89a" + b"\x00" * 20
    install_client(monkeypatch, body=body)
    ref = make_ref(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), mime="image/png")
    with pytest.raises(V.AttachmentFetchError, match="sniff"):
        V.fetch_attachment_data_uri(ref)


def test_fetch_refuses_unaccepted_mime():
    with pytest.raises(V.AttachmentFetchError, match="not an accepted image type"):
        V.fetch_attachment_data_uri(make_ref(mime="image/svg+xml"))


def test_fetch_refuses_oversize_declaration(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 10, raising=False)
    with pytest.raises(V.AttachmentFetchError, match="gateway limit"):
        V.fetch_attachment_data_uri(make_ref(bytes=11))


def test_fetch_caps_a_lying_content_length(monkeypatch):
    monkeypatch.setattr(V.settings, "llm_gateway_attachment_max_bytes", 100, raising=False)
    install_client(monkeypatch, body=b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000)
    with pytest.raises(V.AttachmentFetchError, match="exceeded"):
        V.fetch_attachment_data_uri(make_ref(bytes=50))


def test_fetch_refuses_an_http_error(monkeypatch):
    install_client(monkeypatch, stream_status=500)
    with pytest.raises(V.AttachmentFetchError, match="fetch failed"):
        V.fetch_attachment_data_uri(make_ref())


def test_fetch_refuses_an_empty_body(monkeypatch):
    install_client(monkeypatch, body=b"")
    with pytest.raises(V.AttachmentFetchError, match="no bytes"):
        V.fetch_attachment_data_uri(make_ref())


def test_zero_byte_ref_is_rejected_by_the_contract():
    """`bytes: 0` used to skip the size cross-check entirely."""
    with pytest.raises(Exception):
        make_ref(bytes=0)


# ───────────────────────────────────────────────────────────────
# Sniffing
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "data,expected",
    [
        (PNG_BYTES, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 8, "image/gif"),
        (b"GIF87a" + b"\x00" * 8, "image/gif"),
        (b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8, "image/webp"),
        (b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 8, None),  # RIFF but not WebP
        (b"<svg onload=alert(1)>", None),
        (b"", None),
    ],
)
def test_sniff_mime(data, expected):
    assert V.sniff_mime(data) == expected


# ───────────────────────────────────────────────────────────────
# Assembly
# ───────────────────────────────────────────────────────────────

def test_images_attach_to_the_last_user_message(monkeypatch):
    install_client(monkeypatch)
    serialized = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "look at this"},
    ]
    out = V.build_multimodal_messages(serialized, [make_ref()])

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "first"}
    assert out[2] == {"role": "assistant", "content": "reply"}
    parts = out[3]["content"]
    assert parts[0] == {"type": "text", "text": "look at this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_multiple_images_all_attach(monkeypatch):
    install_client(monkeypatch)
    refs = [make_ref(), make_ref(), make_ref()]
    out = V.build_multimodal_messages([{"role": "user", "content": "three"}], refs)
    assert sum(1 for p in out[0]["content"] if p["type"] == "image_url") == 3


def test_input_list_is_not_mutated(monkeypatch):
    install_client(monkeypatch)
    serialized = [{"role": "user", "content": "hi"}]
    V.build_multimodal_messages(serialized, [make_ref()])
    assert serialized == [{"role": "user", "content": "hi"}]


def test_all_attachments_failing_raises_rather_than_silently_dropping(monkeypatch):
    install_client(monkeypatch, stream_status=500)
    with pytest.raises(V.AttachmentFetchError):
        V.build_multimodal_messages([{"role": "user", "content": "hi"}], [make_ref()])


def test_partial_failure_keeps_the_usable_images(monkeypatch):
    calls = {"n": 0}

    class _Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            calls["n"] += 1
            return FakeStream(PNG_BYTES, 500 if calls["n"] == 1 else 200)

    monkeypatch.setattr(V.httpx, "Client", _Client)
    out = V.build_multimodal_messages(
        [{"role": "user", "content": "hi"}], [make_ref(), make_ref()]
    )
    assert sum(1 for p in out[0]["content"] if p["type"] == "image_url") == 1


def test_no_user_message_is_an_error(monkeypatch):
    install_client(monkeypatch)
    with pytest.raises(V.AttachmentFetchError, match="no user message"):
        V.build_multimodal_messages([{"role": "system", "content": "sys"}], [make_ref()])
