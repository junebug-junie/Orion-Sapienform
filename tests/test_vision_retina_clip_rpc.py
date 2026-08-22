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


def _make_envelope(target_stream_id: str = "retina-stream-01") -> BaseEnvelope:
    """Defaults to "retina-stream-01" -- the Settings() default RETINA_STREAM_ID
    the svc fixture below uses, so every existing call site in this file (all
    of which call this with no args) keeps exercising the SAME instance's own
    stream, matching by construction rather than needing every test updated
    when target_stream_id became required (2026-08-22)."""
    corr = uuid.uuid4()
    return BaseEnvelope(
        kind="retina.clip_capture.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        correlation_id=corr,
        reply_to=f"orion:retina:clip:reply:{corr}",
        payload={"target_stream_id": target_stream_id},
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
    assert reply_envelope.payload["error_code"] == "disabled"


@pytest.mark.asyncio
async def test_handle_clip_request_reports_not_configured_without_capturing(monkeypatch):
    """Review finding, 2026-08-22: this branch (RETINA_PERCEPT_STORE_URL
    unset) had zero direct test coverage before or after the try/except/else
    refactor that added the camera-identity check ahead of it -- a future
    change that silently reorders/misnests this elif chain would ship
    undetected without this."""
    settings = Settings(RETINA_CLIP_ENABLED=True, RETINA_PERCEPT_STORE_URL="")
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)

    async def _must_not_run(self):
        raise AssertionError("capture_and_upload_clip must not run when not configured")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _must_not_run)

    await svc._handle_clip_request(_make_envelope())

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "not_configured"


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


# --- Camera-identity check (review/design instruction, 2026-08-22) ---------
# Juniper's explicit instruction: "I want this to only run on my carbon
# webcam." orion:exec:request:RetinaClipCaptureService has no built-in
# per-instance routing -- any retina instance subscribed to it with
# RETINA_CLIP_ENABLED=true would otherwise respond to ANY request. This is
# the structural guarantee: every instance checks the request's
# target_stream_id against its own RETINA_STREAM_ID before doing anything
# else, including before checking RETINA_CLIP_ENABLED.


@pytest.mark.asyncio
async def test_mismatched_target_stream_id_refuses_without_capturing(svc, monkeypatch):
    async def _must_not_run(self):
        raise AssertionError("capture_and_upload_clip must not run for the wrong camera")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _must_not_run)

    await svc._handle_clip_request(_make_envelope(target_stream_id="cam0"))

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "wrong_camera"


@pytest.mark.asyncio
async def test_mismatched_target_stream_id_checked_before_clip_enabled(monkeypatch):
    """The camera-identity check must be the FIRST gate -- even an instance
    with RETINA_CLIP_ENABLED=false must report "wrong_camera", not
    "disabled", when the request wasn't meant for it at all."""
    settings = Settings(RETINA_CLIP_ENABLED=False, RETINA_STREAM_ID="carbon")
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)

    await svc._handle_clip_request(_make_envelope(target_stream_id="cam0"))

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["error_code"] == "wrong_camera"


@pytest.mark.asyncio
async def test_matching_target_stream_id_proceeds_normally(monkeypatch):
    settings = Settings(
        RETINA_CLIP_ENABLED=True,
        RETINA_PERCEPT_STORE_URL="http://store/percepts",
        RETINA_STREAM_ID="carbon",
    )
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)

    async def _fake_capture(self):
        return RetinaClipCaptureResultPayload(ok=True, video_sha256="a" * 64, audio_sha256="b" * 64)

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    await svc._handle_clip_request(_make_envelope(target_stream_id="carbon"))

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is True


@pytest.mark.asyncio
async def test_missing_target_stream_id_reports_invalid_request_not_a_crash(svc):
    """target_stream_id is required with no default -- a caller that omits
    it entirely (e.g. a stale/pre-2026-08-22 orchestrator) must get a clean
    error reply, not an unhandled exception that leaves the RPC caller
    waiting for a reply that never comes."""
    corr = uuid.uuid4()
    envelope = BaseEnvelope(
        kind="retina.clip_capture.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        correlation_id=corr,
        reply_to=f"orion:retina:clip:reply:{corr}",
        payload={},  # no target_stream_id at all
    )

    await svc._handle_clip_request(envelope)

    _, reply_envelope = svc.bus.publish.await_args.args
    assert reply_envelope.payload["ok"] is False
    assert reply_envelope.payload["error_code"] == "invalid_request"


# --- HTTP route (POST /capture/clip) camera-identity check, review finding
# 2026-08-22 ------------------------------------------------------------
# The bus RPC path (_handle_clip_request, above) got a required
# target_stream_id check; the pre-existing HTTP route did NOT get the same
# check in the same commit, so the whole camera-identity guarantee was
# fully bypassable via a plain curl for one commit. Fixed by requiring
# ?target_stream_id=<this instance's RETINA_STREAM_ID> on the HTTP route
# too. These exercise app.main.capture_clip_endpoint directly (a duck-typed
# fake Request, no real ASGI/ TestClient machinery -- matches the
# lightweight direct-call convention already used throughout this file and
# services/orion-hub's own route tests).


class _FakeRequest:
    def __init__(self, *, query_params: dict | None = None, headers: dict | None = None):
        self.query_params = query_params or {}
        self.headers = headers or {}


@pytest.mark.asyncio
async def test_http_route_rejects_mismatched_target_stream_id(monkeypatch):
    import app.main as main_module

    settings = Settings(
        RETINA_CLIP_ENABLED=True,
        RETINA_PERCEPT_STORE_URL="http://store/percepts",
        RETINA_STREAM_ID="carbon",
    )
    bus = MagicMock()
    test_svc = RetinaService(settings=settings, bus=bus)
    monkeypatch.setattr(main_module, "service", test_svc)

    async def _must_not_run(self):
        raise AssertionError("capture_and_upload_clip must not run for the wrong camera")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _must_not_run)

    response = await main_module.capture_clip_endpoint(
        _FakeRequest(query_params={"target_stream_id": "cam0"})
    )

    assert response.status_code == 400
    import json as _json

    body = _json.loads(response.body)
    assert body["ok"] is False
    assert body["error_code"] == "wrong_camera"


@pytest.mark.asyncio
async def test_http_route_rejects_a_missing_target_stream_id(monkeypatch):
    """target_stream_id has no default on this route either -- omitting it
    entirely must not silently fall through to a capture."""
    import app.main as main_module

    settings = Settings(
        RETINA_CLIP_ENABLED=True,
        RETINA_PERCEPT_STORE_URL="http://store/percepts",
        RETINA_STREAM_ID="carbon",
    )
    bus = MagicMock()
    test_svc = RetinaService(settings=settings, bus=bus)
    monkeypatch.setattr(main_module, "service", test_svc)

    async def _must_not_run(self):
        raise AssertionError("capture_and_upload_clip must not run without target_stream_id")

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _must_not_run)

    response = await main_module.capture_clip_endpoint(_FakeRequest())

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_http_route_proceeds_when_target_stream_id_matches(monkeypatch):
    import app.main as main_module

    settings = Settings(
        RETINA_CLIP_ENABLED=True,
        RETINA_PERCEPT_STORE_URL="http://store/percepts",
        RETINA_STREAM_ID="carbon",
    )
    bus = MagicMock()
    test_svc = RetinaService(settings=settings, bus=bus)
    monkeypatch.setattr(main_module, "service", test_svc)

    async def _fake_capture(self):
        return RetinaClipCaptureResultPayload(ok=True, video_sha256="a" * 64, audio_sha256="b" * 64)

    monkeypatch.setattr(RetinaService, "capture_and_upload_clip", _fake_capture)

    response = await main_module.capture_clip_endpoint(
        _FakeRequest(query_params={"target_stream_id": "carbon"})
    )

    # A dict/pydantic return (not a JSONResponse) means it fell through to
    # the success path at the end of the route rather than one of the
    # early-return error branches.
    assert response.get("ok") is True
