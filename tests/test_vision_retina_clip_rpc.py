"""Regression guard for the bus-reachable twin of POST /capture/clip
(2026-08-22): orion:exec:request:RetinaClipCaptureService. This exists
because carbon accepts no inbound HTTP at all (docs/operations/carbon-
webcam.md), so a remote caller like Hub (via orion-juniper-affective-state)
can only reach this node through the bus. Exercises RetinaService's RPC
handler directly with a mocked bus -- no real Redis, no real ffmpeg, no real
camera. See tests/test_vision_retina_clip_capture.py for the ffmpeg
subprocess layer and tests/test_vision_retina_device_contention.py for the
pause_device()/_device_lock coordination this handler also depends on.

Note: BaseEnvelope.derive_child() always converts a pydantic payload to a
plain dict via model_dump(mode="json") -- reply_envelope.payload is a dict
below, not a RetinaClipCaptureResultPayload instance, same as every other
bus reply in this codebase (see orion-affectgpt-worker's own
_publish_result, same pattern).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clip_capture import ClipCaptureError
from app.frame_store import PerceptUploadError
from app.main import RetinaService
from app.settings import Settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import RetinaClipCaptureResultPayload


def _make_envelope() -> BaseEnvelope:
    corr = uuid.uuid4()
    return BaseEnvelope(
        kind="retina.clip_capture.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        correlation_id=corr,
        reply_to=f"orion:retina:clip:reply:{corr}",
        payload={},
    )


@pytest.fixture
def svc():
    settings = Settings(RETINA_CLIP_ENABLED=True, RETINA_PERCEPT_STORE_URL="http://store/percepts")
    bus = MagicMock()
    bus.publish = AsyncMock()
    return RetinaService(settings=settings, bus=bus)


@pytest.mark.asyncio
async def test_handle_clip_request_publishes_success_reply(svc, monkeypatch):
    async def _fake_capture(self):
        return RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64,
            duration_sec=8.0, video_bytes=100, audio_bytes=200,
        )

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    envelope = _make_envelope()
    await svc._handle_clip_request(envelope)

    assert svc.bus.publish.await_count == 1
    reply_channel, reply_envelope = svc.bus.publish.await_args.args
    assert reply_channel == envelope.reply_to
    payload = reply_envelope.payload
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert payload["video_sha256"] == "a" * 64


@pytest.mark.asyncio
async def test_handle_clip_request_reports_disabled_without_capturing(monkeypatch):
    settings = Settings(RETINA_CLIP_ENABLED=False, RETINA_PERCEPT_STORE_URL="http://store/percepts")
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)

    called = False

    async def _fake_capture(self):
        nonlocal called
        called = True
        raise AssertionError("capture_and_upload_clip must not run when disabled")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    await svc._handle_clip_request(_make_envelope())

    assert not called
    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "disabled"


@pytest.mark.asyncio
async def test_handle_clip_request_reports_busy_without_blocking(svc, monkeypatch):
    async def _never_should_run(self):
        raise AssertionError("capture_and_upload_clip must not run while busy")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _never_should_run)

    async with svc._clip_capture_lock:
        await svc._handle_clip_request(_make_envelope())

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "busy"


@pytest.mark.asyncio
async def test_handle_clip_request_maps_clip_capture_error(svc, monkeypatch):
    async def _fake_capture(self):
        raise ClipCaptureError("ffmpeg exited 240")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    await svc._handle_clip_request(_make_envelope())

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "capture_error"
    assert "ffmpeg exited" in reply_envelope.payload["error"]


@pytest.mark.asyncio
async def test_handle_clip_request_still_replies_on_an_unexpected_exception(svc, monkeypatch):
    """Regression guard: this runs in a fire-and-forget asyncio.create_task
    from _clip_consume_loop -- an uncaught exception here would be silently
    swallowed by asyncio, and the RPC caller would just time out,
    indistinguishable from a genuinely hung device."""

    async def _fake_capture(self):
        raise RuntimeError("something nobody anticipated")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    await svc._handle_clip_request(_make_envelope())

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "unexpected_error"


@pytest.mark.asyncio
async def test_handle_clip_request_maps_percept_upload_error(svc, monkeypatch):
    async def _fake_capture(self):
        raise PerceptUploadError("percept-store unreachable")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    await svc._handle_clip_request(_make_envelope())

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "upload_error"


@pytest.mark.asyncio
async def test_a_capture_in_flight_makes_a_second_bus_request_report_busy(svc, monkeypatch):
    """The whole point of moving _clip_capture_lock onto RetinaService (not
    a module-level global only the HTTP route could see): a capture started
    through one path (HTTP, or here a first bus request) must make a second
    concurrent request -- through either path -- see "busy" immediately
    rather than queueing silently behind it or racing the physical device.
    """
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()

    # capture_and_upload_clip already acquires _clip_capture_lock internally
    # in the real implementation; here we simulate that by holding it
    # ourselves for the duration of the "slow" fake capture.
    async def _slow_capture_holding_lock(self):
        async with self._clip_capture_lock:
            started.set()
            await release.wait()
        return RetinaClipCaptureResultPayload(ok=True, video_sha256="c" * 64, audio_sha256="d" * 64)

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _slow_capture_holding_lock)

    first = asyncio.create_task(svc._handle_clip_request(_make_envelope()))
    await started.wait()

    # Second request arrives while the first still holds the lock.
    await svc._handle_clip_request(_make_envelope())
    calls = svc.bus.publish.await_args_list
    second_reply = calls[-1].args[1]
    assert second_reply.payload["ok"] is False
    assert second_reply.payload["error_code"] == "busy"

    release.set()
    await first
