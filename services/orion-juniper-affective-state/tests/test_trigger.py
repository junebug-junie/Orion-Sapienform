"""Bus-free by design -- mocks OrionBusAsync entirely, no live worker/Redis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.main import JuniperAffectiveStateService
from orion.schemas.affectgpt import AffectGptAssessRequestPayload, AffectGptAssessResultPayload


@dataclass
class FakeDecoded:
    ok: bool
    envelope: Any = None
    error: str | None = None


class FakeEnvelope:
    def __init__(self, payload: dict):
        self.payload = payload


class FakeBus:
    def __init__(self):
        self.enabled = True
        self.published: list[tuple[str, Any]] = []
        self._rpc_reply_payload: dict | None = None
        self._rpc_raises: Exception | None = None
        self.last_reply_channel: str | None = None

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.last_reply_channel = reply_channel
        if self._rpc_raises:
            raise self._rpc_raises
        return {"data": b"fake"}  # decode() below is also faked

    @property
    def codec(self):
        outer = self

        class _Codec:
            def decode(self, data):
                if outer._rpc_reply_payload is None:
                    return FakeDecoded(ok=False, error="no reply configured")
                return FakeDecoded(ok=True, envelope=FakeEnvelope(outer._rpc_reply_payload))

        return _Codec()

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


@pytest.fixture
def req():
    return AffectGptAssessRequestPayload(
        video_path="/opt/affectgpt-src/AffectGPT/demo/sample_00000000.mp4",
        audio_path="/opt/affectgpt-src/AffectGPT/demo/sample_00000000.wav",
        subtitle="I don't know! I, I, I don't have experience in this area.",
    )


@pytest.mark.asyncio
async def test_successful_round_trip_publishes_event(req):
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    svc.bus._rpc_reply_payload = AffectGptAssessResultPayload(
        ok=True, raw_response="In the text, ..."
    ).model_dump()

    result, event = await svc.trigger_assessment(req)

    assert result.ok is True
    assert result.raw_response == "In the text, ..."
    assert event.ok is True
    assert event.input_ref == {"video_path": req.video_path, "audio_path": req.audio_path}
    assert len(svc.bus.published) == 1
    channel, _ = svc.bus.published[0]
    assert channel == "orion:affectgpt:assessment"


@pytest.mark.asyncio
async def test_reply_channel_uses_configured_prefix_not_a_hardcoded_literal(req):
    # Regression guard: this was a hardcoded 'orion:affectgpt:reply' literal
    # (review finding, 2026-08-22) instead of settings.CHANNEL_AFFECTGPT_REPLY_PREFIX.
    from app.settings import Settings

    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    svc.bus._rpc_reply_payload = AffectGptAssessResultPayload(ok=True).model_dump()

    await svc.trigger_assessment(req)

    assert svc.bus.last_reply_channel.startswith(Settings().CHANNEL_AFFECTGPT_REPLY_PREFIX + ":")


@pytest.mark.asyncio
async def test_timeout_produces_failed_event_not_a_crash(req):
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    svc.bus._rpc_raises = TimeoutError("no reply")

    result, event = await svc.trigger_assessment(req)

    assert result.ok is False
    assert result.error_code == "timeout"
    assert event.ok is False
    # A failed assessment is still a real, publishable event -- review
    # finding (2026-08-22): this used to assert published == [] because
    # trigger_assessment's early-return branches skipped _publish_event
    # entirely, silently dropping the four most operationally likely
    # failure modes from orion:affectgpt:assessment. Every path publishes now.
    assert len(svc.bus.published) == 1
    channel, _ = svc.bus.published[0]
    assert channel == "orion:affectgpt:assessment"


@pytest.mark.asyncio
async def test_bus_unavailable_fails_cleanly(req):
    svc = JuniperAffectiveStateService()
    svc.bus = None

    result, event = await svc.trigger_assessment(req)

    assert result.ok is False
    assert result.error_code == "bus_unavailable"
    assert event.ok is False
