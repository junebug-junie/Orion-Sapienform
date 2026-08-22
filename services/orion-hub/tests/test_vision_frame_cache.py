"""Covers services/orion-hub/scripts/vision_frame_cache.py -- the
latest-frame-per-stream cache backing the Vision panel's "Carbon (live)"
dropdown option (2026-08-22). Confirmed live (investigation, 2026-08-22)
that no persisted "latest frame for stream X" lookup exists anywhere else
in this repo -- this is a Hub-side bus subscriber filling that gap. No real
bus, no real Redis -- exercises _handle_message directly against a fake
decoded envelope, matching the direct-call testing convention already used
by test_vision_affect_ambient.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import vision_frame_cache  # noqa: E402


class FakeEnvelope:
    def __init__(self, kind: str, payload: dict):
        self.kind = kind
        self.payload = payload


class FakeDecoded:
    def __init__(self, ok: bool, envelope: Any = None):
        self.ok = ok
        self.envelope = envelope


class FakeBus:
    def __init__(self):
        self.enabled = True
        self._next_decoded: FakeDecoded | None = None

    def set_next_decoded(self, decoded: FakeDecoded) -> None:
        self._next_decoded = decoded

    @property
    def codec(self):
        outer = self

        class _Codec:
            def decode(self, data):
                return outer._next_decoded

        return _Codec()


def _cache(stream_ids={"carbon"}) -> vision_frame_cache.VisionFrameCache:
    c = vision_frame_cache.VisionFrameCache(
        enabled=True, stream_ids=set(stream_ids), channel="orion:vision:frames"
    )
    c._bus = FakeBus()
    return c


def _frame_envelope(*, stream_id: str, sha256: str | None, frame_ts: float = 123.0) -> FakeDecoded:
    payload = {"stream_id": stream_id, "frame_ts": frame_ts, "width": 640, "height": 480}
    if sha256:
        payload["sha256"] = sha256
    return FakeDecoded(ok=True, envelope=FakeEnvelope(kind="vision.frame.pointer", payload=payload))


@pytest.mark.asyncio
async def test_caches_the_latest_pointer_for_an_allowlisted_stream():
    cache = _cache()
    cache._bus.set_next_decoded(_frame_envelope(stream_id="carbon", sha256="a" * 64, frame_ts=100.0))

    await cache._handle_message({"data": b"x"})

    latest = await cache.get_latest("carbon")
    assert latest is not None
    assert latest["sha256"] == "a" * 64
    assert latest["frame_ts"] == 100.0
    assert latest["width"] == 640


@pytest.mark.asyncio
async def test_ignores_a_stream_not_on_the_allowlist():
    cache = _cache(stream_ids={"carbon"})
    cache._bus.set_next_decoded(_frame_envelope(stream_id="cam0", sha256="b" * 64))

    await cache._handle_message({"data": b"x"})

    assert await cache.get_latest("cam0") is None
    assert await cache.get_latest("carbon") is None


@pytest.mark.asyncio
async def test_ignores_a_frame_pointer_with_no_sha256():
    """A pointer using image_path addressing (shared-filesystem node) has
    nothing this cache can serve back out over HTTP to a browser."""
    cache = _cache()
    cache._bus.set_next_decoded(_frame_envelope(stream_id="carbon", sha256=None))

    await cache._handle_message({"data": b"x"})

    assert await cache.get_latest("carbon") is None


@pytest.mark.asyncio
async def test_ignores_a_non_frame_pointer_envelope():
    cache = _cache()
    cache._bus.set_next_decoded(
        FakeDecoded(ok=True, envelope=FakeEnvelope(kind="some.other.kind", payload={}))
    )

    await cache._handle_message({"data": b"x"})

    assert await cache.get_latest("carbon") is None


@pytest.mark.asyncio
async def test_ignores_a_decode_failure():
    cache = _cache()
    cache._bus.set_next_decoded(FakeDecoded(ok=False))

    await cache._handle_message({"data": b"x"})  # must not raise

    assert await cache.get_latest("carbon") is None


@pytest.mark.asyncio
async def test_a_later_frame_overwrites_the_earlier_one():
    cache = _cache()
    cache._bus.set_next_decoded(_frame_envelope(stream_id="carbon", sha256="a" * 64, frame_ts=100.0))
    await cache._handle_message({"data": b"x"})

    cache._bus.set_next_decoded(_frame_envelope(stream_id="carbon", sha256="b" * 64, frame_ts=200.0))
    await cache._handle_message({"data": b"y"})

    latest = await cache.get_latest("carbon")
    assert latest["sha256"] == "b" * 64
    assert latest["frame_ts"] == 200.0


def test_frame_pointer_records_a_real_cached_at_timestamp():
    before = time.time()
    pointer = vision_frame_cache.FramePointer(
        sha256="a" * 64, frame_ts=1.0, width=1, height=1, cached_at=time.time()
    )
    after = time.time()
    assert before <= pointer.cached_at <= after
    assert pointer.to_dict()["sha256"] == "a" * 64
