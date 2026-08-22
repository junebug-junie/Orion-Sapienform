"""Regression guard for RETINA_CLIP_MIN_INTERVAL_SEC / ClipCaptureCooldownError
(review finding, 2026-08-22): the bus RPC path (orion:exec:request:
RetinaClipCaptureService) has no auth beyond bus reachability -- nothing else
stops a bus-connected caller from firing capture requests back-to-back,
which would produce de facto continuous webcam+mic recording. This tests
the cooldown enforcement in RetinaService.capture_and_upload_clip() directly
with a fake capture_clip/upload_bytes -- no real ffmpeg, no real camera, no
real percept-store.
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clip_capture import ClipCaptureCooldownError
from app.main import RetinaService
from app.settings import Settings


class _NoopSource:
    """Stands in for the real capture source -- pause_device() calls
    stop()/start() on this; both are no-ops here since this test only cares
    about the cooldown gate, not device coordination (see
    test_vision_retina_device_contention.py for that)."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read(self):
        return None


@dataclass
class _FakeClipResult:
    video_bytes: bytes
    audio_bytes: bytes
    duration_sec: float


def _fake_upload_bytes(data, **kwargs):
    return hashlib.sha256(data).hexdigest()


def _make_svc(min_interval: float) -> RetinaService:
    settings = Settings(
        RETINA_CLIP_ENABLED=True,
        RETINA_PERCEPT_STORE_URL="http://store/percepts",
        RETINA_CLIP_MIN_INTERVAL_SEC=min_interval,
    )
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _NoopSource()
    return svc


@pytest.mark.asyncio
async def test_second_capture_within_interval_is_rejected(monkeypatch):
    svc = _make_svc(min_interval=100.0)

    async def _fake_capture_clip(**kwargs):
        return _FakeClipResult(video_bytes=b"v", audio_bytes=b"a", duration_sec=1.0)

    monkeypatch.setattr("app.main.capture_clip", _fake_capture_clip)
    monkeypatch.setattr("app.main.upload_bytes", _fake_upload_bytes)

    first = await svc.capture_and_upload_clip()
    assert first.ok is True

    with pytest.raises(ClipCaptureCooldownError, match="minimum interval"):
        await svc.capture_and_upload_clip()


@pytest.mark.asyncio
async def test_first_capture_is_never_subject_to_cooldown(monkeypatch):
    """0.0 sentinel (_last_clip_capture_ts default) must not itself look
    like "0 seconds ago" and reject the very first request."""
    svc = _make_svc(min_interval=100.0)

    async def _fake_capture_clip(**kwargs):
        return _FakeClipResult(video_bytes=b"v", audio_bytes=b"a", duration_sec=1.0)

    monkeypatch.setattr("app.main.capture_clip", _fake_capture_clip)
    monkeypatch.setattr("app.main.upload_bytes", _fake_upload_bytes)

    result = await svc.capture_and_upload_clip()
    assert result.ok is True


@pytest.mark.asyncio
async def test_capture_succeeds_again_after_interval_elapses(monkeypatch):
    svc = _make_svc(min_interval=0.05)

    async def _fake_capture_clip(**kwargs):
        return _FakeClipResult(video_bytes=b"v", audio_bytes=b"a", duration_sec=1.0)

    monkeypatch.setattr("app.main.capture_clip", _fake_capture_clip)
    monkeypatch.setattr("app.main.upload_bytes", _fake_upload_bytes)

    first = await svc.capture_and_upload_clip()
    assert first.ok is True

    await asyncio.sleep(0.08)

    second = await svc.capture_and_upload_clip()
    assert second.ok is True


@pytest.mark.asyncio
async def test_handle_clip_request_maps_cooldown_to_its_own_error_code(monkeypatch):
    """Not just ClipCaptureError's generic "capture_error" -- a caller
    should be able to tell "rate limited" apart from "hardware failed"."""
    svc = _make_svc(min_interval=100.0)

    async def _fake_capture_clip(**kwargs):
        return _FakeClipResult(video_bytes=b"v", audio_bytes=b"a", duration_sec=1.0)

    monkeypatch.setattr("app.main.capture_clip", _fake_capture_clip)
    monkeypatch.setattr("app.main.upload_bytes", _fake_upload_bytes)

    import uuid as uuid_mod

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    def _envelope():
        c = uuid_mod.uuid4()
        return BaseEnvelope(
            kind="retina.clip_capture.request",
            source=ServiceRef(name="test-caller", version="0.0.0"),
            correlation_id=c,
            reply_to=f"orion:retina:clip:reply:{c}",
            payload={"target_stream_id": "retina-stream-01"},  # matches _make_svc's default RETINA_STREAM_ID
        )

    await svc._handle_clip_request(_envelope())  # first: succeeds
    await svc._handle_clip_request(_envelope())  # second: cooldown

    _, second_reply = svc.bus.publish.await_args.args
    assert second_reply.payload["ok"] is False
    assert second_reply.payload["error_code"] == "cooldown"
