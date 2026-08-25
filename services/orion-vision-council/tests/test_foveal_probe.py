"""Regression coverage for the foveal probe (app/foveal_probe.py) --
docs/superpowers/specs/2026-08-12-perception-frontier-design.md's Foveal
tier, manually triggered via POST /debug/foveal-probe.

Three real hops, tested independently plus end-to-end with a fake bus:
resolve the newest local frame -> upload it to the percept store -> ask
orion-llm-gateway's vision-capable `chat` route and return its real reply.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.foveal_probe import (
    DEFAULT_CAPTION_PROMPT,
    FovealNotConfiguredError,
    FovealReplyDecodeError,
    FovealTaskFailedError,
    NoFrameAvailableError,
    PerceptUploadError,
    build_foveal_chat_envelope,
    resolve_latest_frame_path,
    run_foveal_probe,
    upload_frame_bytes,
)
from orion.core.bus.bus_schemas import ChatRequestPayload


# ---------------------------------------------------------------------------
# resolve_latest_frame_path
# ---------------------------------------------------------------------------


def test_resolve_latest_frame_path_returns_none_for_missing_dir(tmp_path) -> None:
    assert resolve_latest_frame_path(str(tmp_path / "does_not_exist")) is None


def test_resolve_latest_frame_path_returns_none_for_empty_dir(tmp_path) -> None:
    assert resolve_latest_frame_path(str(tmp_path)) is None


def test_resolve_latest_frame_path_picks_newest_by_mtime(tmp_path) -> None:
    older = tmp_path / "a.jpg"
    newer = tmp_path / "b.jpg"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    now = time.time()
    # Set mtimes explicitly rather than relying on write-order timing --
    # two writes microseconds apart can land on the same mtime on some
    # filesystems, which would make this test flaky rather than wrong.
    import os

    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))
    assert resolve_latest_frame_path(str(tmp_path)) == newer


def test_resolve_latest_frame_path_ignores_non_jpg_files(tmp_path) -> None:
    (tmp_path / "readme.txt").write_bytes(b"not a frame")
    assert resolve_latest_frame_path(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# upload_frame_bytes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_upload_frame_bytes_returns_sha256_on_match() -> None:
    data = b"fake jpeg bytes"
    real_sha = hashlib.sha256(data).hexdigest()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode())):
        result = upload_frame_bytes(data, base_url="http://percept-store/percepts")
    assert result == real_sha


def test_upload_frame_bytes_rejects_hash_mismatch() -> None:
    """The store returning a DIFFERENT hash than what was actually sent must
    be treated as a failure, not trusted -- this is the one thing this
    function exists to guard against."""
    data = b"fake jpeg bytes"
    wrong_sha = "0" * 64
    with patch("urllib.request.urlopen", return_value=_FakeResponse(json.dumps({"sha256": wrong_sha}).encode())):
        with pytest.raises(PerceptUploadError):
            upload_frame_bytes(data, base_url="http://percept-store/percepts")


def test_upload_frame_bytes_wraps_network_error() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        with pytest.raises(PerceptUploadError):
            upload_frame_bytes(b"data", base_url="http://percept-store/percepts")


def test_upload_frame_bytes_sends_token_header_when_set() -> None:
    data = b"data"
    real_sha = hashlib.sha256(data).hexdigest()
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["headers"] = dict(req.header_items())
        return _FakeResponse(json.dumps({"sha256": real_sha}).encode())

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        upload_frame_bytes(data, base_url="http://percept-store/percepts", token="secret-token")
    assert captured["headers"].get("X-orion-percept-token") == "secret-token"


# ---------------------------------------------------------------------------
# build_foveal_chat_envelope
# ---------------------------------------------------------------------------


def test_build_foveal_chat_envelope_uses_default_caption_prompt_when_no_question() -> None:
    env = build_foveal_chat_envelope(
        sha256="a" * 64,
        percept_store_url="http://percept-store/percepts",
        frame_bytes_len=1234,
        question=None,
        reply_to="orion:council:reply:foveal:123",
        llm_route="chat",
        service_name="vision-council",
        service_version="0.1.0",
    )
    req = ChatRequestPayload.model_validate(env.payload)
    assert req.route == "chat"
    assert req.messages[0].content == DEFAULT_CAPTION_PROMPT
    assert len(req.attachments) == 1
    attachment = req.attachments[0]
    assert attachment.kind == "percept"
    assert attachment.sha256 == "a" * 64
    assert attachment.mime == "image/jpeg"
    assert attachment.bytes == 1234
    assert attachment.source_url == "http://percept-store/percepts/" + "a" * 64
    assert env.reply_to == "orion:council:reply:foveal:123"


def test_build_foveal_chat_envelope_uses_question_as_prompt_when_set() -> None:
    env = build_foveal_chat_envelope(
        sha256="b" * 64,
        percept_store_url="http://percept-store/percepts",
        frame_bytes_len=1,
        question="is the door open?",
        reply_to="orion:council:reply:foveal:456",
        llm_route="chat",
        service_name="vision-council",
        service_version="0.1.0",
    )
    req = ChatRequestPayload.model_validate(env.payload)
    assert req.messages[0].content == "is the door open?"


def test_build_foveal_chat_envelope_uses_supplied_correlation_id() -> None:
    corr = uuid.uuid4()
    env = build_foveal_chat_envelope(
        sha256="c" * 64,
        percept_store_url="http://percept-store/percepts",
        frame_bytes_len=1,
        question=None,
        reply_to="orion:council:reply:foveal:789",
        llm_route="chat",
        service_name="vision-council",
        service_version="0.1.0",
        correlation_id=corr,
    )
    assert env.correlation_id == corr


def test_build_foveal_chat_envelope_uses_real_byte_length() -> None:
    env = build_foveal_chat_envelope(
        sha256="d" * 64,
        percept_store_url="http://percept-store/percepts",
        frame_bytes_len=42,
        question=None,
        reply_to="orion:council:reply:foveal:000",
        llm_route="chat",
        service_name="vision-council",
        service_version="0.1.0",
    )
    req = ChatRequestPayload.model_validate(env.payload)
    assert req.attachments[0].bytes == 42


def test_build_foveal_chat_envelope_whitespace_only_question_falls_back_to_caption_prompt() -> None:
    """A whitespace-only question (e.g. ?question=%20) must fall back to the
    default caption prompt, not send the model a blank instruction --
    code review caught the pre-strip truthiness check missing this."""
    env = build_foveal_chat_envelope(
        sha256="e" * 64,
        percept_store_url="http://percept-store/percepts",
        frame_bytes_len=1,
        question="   ",
        reply_to="orion:council:reply:foveal:111",
        llm_route="chat",
        service_name="vision-council",
        service_version="0.1.0",
    )
    req = ChatRequestPayload.model_validate(env.payload)
    assert req.messages[0].content == DEFAULT_CAPTION_PROMPT


# ---------------------------------------------------------------------------
# run_foveal_probe (end-to-end with a fake bus)
# ---------------------------------------------------------------------------


def _ok_reply(text: str) -> dict:
    return {"content": text}


class _FakeCodec:
    def __init__(self, decode_ok: bool = True, decode_error: str | None = None):
        self._decode_ok = decode_ok
        self._decode_error = decode_error

    def decode(self, data):
        if not self._decode_ok:
            return SimpleNamespace(ok=False, error=self._decode_error, envelope=None)
        return SimpleNamespace(ok=True, error=None, envelope=SimpleNamespace(payload=data))


class _FakeBus:
    def __init__(self, reply_payload, *, decode_ok: bool = True, decode_error: str | None = None):
        self._reply_payload = reply_payload
        self.codec = _FakeCodec(decode_ok=decode_ok, decode_error=decode_error)
        self.published = []

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.published.append((request_channel, envelope, reply_channel, timeout_sec))
        return {"data": self._reply_payload}


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        CHANNEL_LLM_REQUEST="orion:exec:request:LLMGatewayService",
        CHANNEL_LLM_REPLY_PREFIX="orion:council:reply",
        FOVEAL_LLM_ROUTE="chat",
        FOVEAL_HOST_TIMEOUT_SEC=45.0,
        FOVEAL_FRAMES_DIR="/nonexistent",
        FOVEAL_PERCEPT_STORE_URL="http://percept-store/percepts",
        FOVEAL_PERCEPT_STORE_TOKEN="",
        FOVEAL_PERCEPT_UPLOAD_TIMEOUT_SEC=10.0,
        SERVICE_NAME="vision-council",
        SERVICE_VERSION="0.1.0",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_run_foveal_probe_refuses_when_percept_store_unconfigured(tmp_path) -> None:
    settings = _settings(FOVEAL_PERCEPT_STORE_URL="", FOVEAL_FRAMES_DIR=str(tmp_path))
    (tmp_path / "frame.jpg").write_bytes(b"data")
    bus = _FakeBus(reply_payload=_ok_reply("unused"))
    with pytest.raises(FovealNotConfiguredError):
        await run_foveal_probe(bus, settings)
    assert bus.published == []


@pytest.mark.asyncio
async def test_run_foveal_probe_refuses_when_llm_route_unconfigured(tmp_path) -> None:
    """Checked before the real disk read + upload -- a blank route must fail
    fast rather than spend a real upload on every call before the gateway
    eventually replies with an embedded routing error."""
    settings = _settings(FOVEAL_LLM_ROUTE="", FOVEAL_FRAMES_DIR=str(tmp_path))
    (tmp_path / "frame.jpg").write_bytes(b"data")
    bus = _FakeBus(reply_payload=_ok_reply("unused"))
    with pytest.raises(FovealNotConfiguredError):
        await run_foveal_probe(bus, settings)
    assert bus.published == []


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_when_no_frame_exists(tmp_path) -> None:
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path / "empty"))
    bus = _FakeBus(reply_payload={})
    with pytest.raises(NoFrameAvailableError):
        await run_foveal_probe(bus, settings)


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_when_frame_is_empty(tmp_path) -> None:
    """sha256(b"") is well-defined and would pass upload_frame_bytes's hash
    check, so a 0-byte frame (a truncated/partial capture) must be rejected
    before upload rather than silently sent as a 'successful' empty
    attachment."""
    (tmp_path / "frame.jpg").write_bytes(b"")
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload={})
    with pytest.raises(NoFrameAvailableError):
        await run_foveal_probe(bus, settings)
    assert bus.published == []


@pytest.mark.asyncio
async def test_run_foveal_probe_end_to_end_success(tmp_path) -> None:
    frame_bytes = b"a real jpeg would go here"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)

    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    reply = _ok_reply("a person at a desk")
    bus = _FakeBus(reply_payload=reply)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        result = await run_foveal_probe(bus, settings)

    assert result["sha256"] == real_sha
    assert result["caption"] == "a person at a desk"
    assert result["reply"] == reply
    assert result["frame_path"] == str(tmp_path / "frame.jpg")

    # The actual RPC target must be the gateway's existing chat-intake
    # channel -- the retired foveal-host channel is gone entirely, and this
    # must never silently fall back to it.
    request_channel, envelope, reply_channel, timeout_sec = bus.published[0]
    assert request_channel == "orion:exec:request:LLMGatewayService"
    assert reply_channel.startswith("orion:council:reply:foveal:")
    assert timeout_sec == 45.0
    req = ChatRequestPayload.model_validate(envelope.payload)
    assert req.attachments[0].sha256 == real_sha
    assert req.route == "chat"


@pytest.mark.asyncio
async def test_run_foveal_probe_vqa_mode_passes_question_through(tmp_path) -> None:
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    reply = _ok_reply("yes")
    bus = _FakeBus(reply_payload=reply)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        result = await run_foveal_probe(bus, settings, question="is the door open?")

    assert result["caption"] == "yes"
    _, envelope, _, _ = bus.published[0]
    req = ChatRequestPayload.model_validate(envelope.payload)
    assert req.messages[0].content == "is the door open?"


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_when_upload_fails(tmp_path) -> None:
    (tmp_path / "frame.jpg").write_bytes(b"data")
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload={})

    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(PerceptUploadError):
            await run_foveal_probe(bus, settings)
    assert bus.published == []


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_distinct_error_on_decode_failure(tmp_path) -> None:
    """A bus-level decode failure must not be reported through
    PerceptUploadError -- reusing that class here would point debugging at
    percept-store connectivity when the upload had already succeeded."""
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload={}, decode_ok=False, decode_error="schema mismatch")

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        with pytest.raises(FovealReplyDecodeError):
            await run_foveal_probe(bus, settings)


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_when_response_is_empty(tmp_path) -> None:
    """The RPC itself can succeed (envelope decodes fine) while the actual
    answer is empty -- the exact live failure mode found in production
    2026-08-25 with the retired vision-host path (BLIP-family fallback
    rejected as caption_rejected:too_short). The caller must see that
    failure, not a blanket ok:true with an empty caption."""
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload=_ok_reply(""))

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        with pytest.raises(FovealTaskFailedError) as excinfo:
            await run_foveal_probe(bus, settings)
    assert excinfo.value.error_code == "empty_response"


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_on_embedded_error_content(tmp_path) -> None:
    """`_call_llm_raw` (app/main.py) treats a `"[Error: ...]"`-prefixed
    content string as a gateway-reported failure, not a real answer -- the
    foveal probe must apply the identical convention since it shares the
    same reply contract."""
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload=_ok_reply("[Error: no vision-capable route reachable]"))

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        with pytest.raises(FovealTaskFailedError) as excinfo:
            await run_foveal_probe(bus, settings)
    assert "no vision-capable route reachable" in excinfo.value.detail
    assert excinfo.value.error_code == "gateway_error"


@pytest.mark.asyncio
async def test_run_foveal_probe_runs_blocking_io_off_the_event_loop(tmp_path) -> None:
    """resolve_latest_frame_path/read_bytes/upload_frame_bytes must not run
    directly on the event loop -- that would stall CouncilService's own
    always-on _consume/_consume_rpc tasks (app/main.py) for the full upload
    timeout every time this endpoint fires."""
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload=_ok_reply("x"))

    calls = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func, *args, **kwargs):
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ), patch("app.foveal_probe.asyncio.to_thread", side_effect=_spy_to_thread):
        await run_foveal_probe(bus, settings)

    assert len(calls) == 1
