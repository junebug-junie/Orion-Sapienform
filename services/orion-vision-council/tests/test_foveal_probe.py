"""Regression coverage for the foveal probe (app/foveal_probe.py) --
docs/superpowers/specs/2026-08-12-perception-frontier-design.md's Foveal
tier, manually triggered via POST /debug/foveal-probe.

Three real hops, tested independently plus end-to-end with a fake bus:
resolve the newest local frame -> upload it to the percept store -> RPC the
isolated foveal-host channel and return its real reply.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.foveal_probe import (
    FovealHostNotConfiguredError,
    NoFrameAvailableError,
    PerceptUploadError,
    build_foveal_task_envelope,
    resolve_latest_frame_path,
    run_foveal_probe,
    upload_frame_bytes,
)
from orion.schemas.vision import VisionTaskRequestPayload


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
# build_foveal_task_envelope
# ---------------------------------------------------------------------------


def test_build_foveal_task_envelope_caption_mode_when_no_question() -> None:
    env = build_foveal_task_envelope(
        sha256="a" * 64,
        question=None,
        reply_to="orion:vision:reply:foveal:123",
        service_name="vision-council",
        service_version="0.1.0",
    )
    task = VisionTaskRequestPayload.model_validate(env.payload)
    assert task.task_type == "caption_frame"
    assert task.request == {"percept_sha256": "a" * 64}
    assert env.reply_to == "orion:vision:reply:foveal:123"


def test_build_foveal_task_envelope_vqa_mode_when_question_set() -> None:
    env = build_foveal_task_envelope(
        sha256="b" * 64,
        question="is the door open?",
        reply_to="orion:vision:reply:foveal:456",
        service_name="vision-council",
        service_version="0.1.0",
    )
    task = VisionTaskRequestPayload.model_validate(env.payload)
    assert task.task_type == "vqa"
    assert task.request == {"percept_sha256": "b" * 64, "question": "is the door open?"}


def test_build_foveal_task_envelope_uses_supplied_correlation_id() -> None:
    corr = uuid.uuid4()
    env = build_foveal_task_envelope(
        sha256="c" * 64,
        question=None,
        reply_to="orion:vision:reply:foveal:789",
        service_name="vision-council",
        service_version="0.1.0",
        correlation_id=corr,
    )
    assert env.correlation_id == corr


# ---------------------------------------------------------------------------
# run_foveal_probe (end-to-end with a fake bus)
# ---------------------------------------------------------------------------


class _FakeCodec:
    def decode(self, data):
        return SimpleNamespace(ok=True, error=None, envelope=SimpleNamespace(payload=data))


class _FakeBus:
    def __init__(self, reply_payload):
        self._reply_payload = reply_payload
        self.codec = _FakeCodec()
        self.published = []

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.published.append((request_channel, envelope, reply_channel, timeout_sec))
        return {"data": self._reply_payload}


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        CHANNEL_FOVEAL_HOST_REQUEST="orion:exec:request:VisionHostService:circe-vl",
        CHANNEL_FOVEAL_HOST_REPLY_PREFIX="orion:vision:reply:foveal",
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
async def test_run_foveal_probe_refuses_when_channel_unconfigured(tmp_path) -> None:
    settings = _settings(CHANNEL_FOVEAL_HOST_REQUEST="", FOVEAL_FRAMES_DIR=str(tmp_path))
    (tmp_path / "frame.jpg").write_bytes(b"data")
    bus = _FakeBus(reply_payload={"caption": {"text": "unused"}})
    with pytest.raises(FovealHostNotConfiguredError):
        await run_foveal_probe(bus, settings)
    assert bus.published == []


@pytest.mark.asyncio
async def test_run_foveal_probe_raises_when_no_frame_exists(tmp_path) -> None:
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path / "empty"))
    bus = _FakeBus(reply_payload={})
    with pytest.raises(NoFrameAvailableError):
        await run_foveal_probe(bus, settings)


@pytest.mark.asyncio
async def test_run_foveal_probe_end_to_end_success(tmp_path) -> None:
    frame_bytes = b"a real jpeg would go here"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)

    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload={"caption": {"text": "a person at a desk"}})

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        result = await run_foveal_probe(bus, settings)

    assert result["sha256"] == real_sha
    assert result["reply"] == {"caption": {"text": "a person at a desk"}}
    assert result["frame_path"] == str(tmp_path / "frame.jpg")

    # The actual RPC target must be the configured channel -- never hardcoded
    # and never accidentally the shared vision-host channel.
    request_channel, envelope, reply_channel, timeout_sec = bus.published[0]
    assert request_channel == "orion:exec:request:VisionHostService:circe-vl"
    assert reply_channel.startswith("orion:vision:reply:foveal:")
    assert timeout_sec == 45.0
    task = VisionTaskRequestPayload.model_validate(envelope.payload)
    assert task.request["percept_sha256"] == real_sha


@pytest.mark.asyncio
async def test_run_foveal_probe_vqa_mode_passes_question_through(tmp_path) -> None:
    frame_bytes = b"frame"
    real_sha = hashlib.sha256(frame_bytes).hexdigest()
    (tmp_path / "frame.jpg").write_bytes(frame_bytes)
    settings = _settings(FOVEAL_FRAMES_DIR=str(tmp_path))
    bus = _FakeBus(reply_payload={"vqa": {"answer": "yes"}})

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"sha256": real_sha}).encode()),
    ):
        result = await run_foveal_probe(bus, settings, question="is the door open?")

    assert result["reply"] == {"vqa": {"answer": "yes"}}
    _, envelope, _, _ = bus.published[0]
    task = VisionTaskRequestPayload.model_validate(envelope.payload)
    assert task.task_type == "vqa"
    assert task.request["question"] == "is the door open?"


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
