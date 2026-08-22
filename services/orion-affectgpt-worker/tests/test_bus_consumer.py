"""Exercises the real bus consumer path (_handle_envelope -> codec.decode ->
run_assessment -> _publish_result) with a fake bus and a stubbed
run_assessment -- the actual GPU inference is out of scope for a unit test,
but the envelope decode/dispatch/reply wiring is real code, previously
untested (review finding, 2026-08-22): every other test only constructed
AffectGptAssessRequestPayload directly, never through the codec.decode path
a live bus message actually takes.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from app.main import AffectGptWorkerService
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.affectgpt import AffectGptAssessRequestPayload, AffectGptAssessResultPayload


class FakeCodecDecoded:
    def __init__(self, ok, envelope=None, error=None):
        self.ok = ok
        self.envelope = envelope
        self.error = error


class FakeBus:
    def __init__(self, decode_result):
        self.enabled = True
        self._decode_result = decode_result
        self.published: list[tuple[str, BaseEnvelope]] = []

    @property
    def codec(self):
        outer = self

        class _Codec:
            def decode(self, data):
                return outer._decode_result

        return _Codec()

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _request_envelope(payload_dict: dict) -> BaseEnvelope:
    return BaseEnvelope(
        kind="affectgpt.assess.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        correlation_id=uuid.uuid4(),
        reply_to="orion:affectgpt:reply:test-corr",
        payload=payload_dict,
    )


@pytest.fixture
def svc():
    return AffectGptWorkerService()


@pytest.mark.asyncio
async def test_handle_envelope_dispatches_and_replies_on_reply_to(svc):
    req_payload = {
        "video_path": "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.mp4",
        "audio_path": "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.wav",
        "subtitle": "",
    }
    envelope = _request_envelope(req_payload)
    svc.bus = FakeBus(FakeCodecDecoded(ok=True, envelope=envelope))
    svc.run_assessment = AsyncMock(
        return_value=AffectGptAssessResultPayload(ok=True, raw_response="In the text, ...")
    )

    await svc._handle_envelope(envelope)

    svc.run_assessment.assert_awaited_once()
    called_payload = svc.run_assessment.await_args.args[0]
    assert isinstance(called_payload, AffectGptAssessRequestPayload)
    assert called_payload.video_path == req_payload["video_path"]

    assert len(svc.bus.published) == 1
    channel, reply_envelope = svc.bus.published[0]
    assert channel == "orion:affectgpt:reply:test-corr"
    assert reply_envelope.kind == "affectgpt.assess.result"
    assert reply_envelope.payload["ok"] is True


@pytest.mark.asyncio
async def test_handle_envelope_falls_back_to_prefix_when_no_reply_to(svc):
    req_payload = {"video_path": "/a.mp4", "audio_path": "/a.wav"}
    envelope = BaseEnvelope(
        kind="affectgpt.assess.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        payload=req_payload,
    )
    svc.bus = FakeBus(FakeCodecDecoded(ok=True, envelope=envelope))
    svc.run_assessment = AsyncMock(return_value=AffectGptAssessResultPayload(ok=True))

    await svc._handle_envelope(envelope)

    channel, _ = svc.bus.published[0]
    assert channel == f"orion:affectgpt:reply:{envelope.correlation_id}"


@pytest.mark.asyncio
async def test_handle_envelope_ignores_invalid_payload_without_crashing(svc):
    envelope = BaseEnvelope(
        kind="affectgpt.assess.request",
        source=ServiceRef(name="test-caller", version="0.0.0"),
        payload={"bogus_field_only": True},  # missing required video_path/audio_path
    )
    svc.bus = FakeBus(FakeCodecDecoded(ok=True, envelope=envelope))
    svc.run_assessment = AsyncMock()

    await svc._handle_envelope(envelope)

    svc.run_assessment.assert_not_awaited()
    assert svc.bus.published == []
