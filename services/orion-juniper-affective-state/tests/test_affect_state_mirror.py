"""2026-08-25: `_publish_event` also mirrors a successful read into the
single Redis key `orion/situational/context.py` polls for chat-turn
grounding (`orion.situational.juniper_affect_state`). Before this, nothing
downstream of `orion:affectgpt:assessment` ever consumed the event except a
manual debug CLI (`scripts/tap_assessments.py`) -- Orion's own chat turns
never found out about a capture. This file is the write side; the read side
has its own tests in `orion/situational/tests/test_juniper_affect_state.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.main import JuniperAffectiveStateService
from orion.schemas.affectgpt import AffectGptAssessRequestPayload, AffectGptAssessResultPayload


class _FakeRedis:
    def __init__(self) -> None:
        self.setex_calls: list[tuple[str, int, str]] = []

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.setex_calls.append((key, ttl_seconds, payload))


class FakeBus:
    """Minimal stand-in carrying both `publish` (the real event stream) and
    `.redis` (the affect-state mirror this file tests) -- test_trigger.py's
    own FakeBus has no `.redis` at all, which is fine there: the mirror
    write is fail-open by contract and silently no-ops against it."""

    def __init__(self) -> None:
        self.enabled = True
        self.published: list[tuple[str, Any]] = []
        self.redis = _FakeRedis()

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


@pytest.fixture
def req():
    return AffectGptAssessRequestPayload(
        video_path="/opt/affectgpt-src/AffectGPT/demo/sample_00000000.mp4",
        audio_path="/opt/affectgpt-src/AffectGPT/demo/sample_00000000.wav",
    )


@pytest.mark.asyncio
async def test_successful_assessment_mirrors_into_affect_state_key(req):
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    event = svc._wrap_event(
        AffectGptAssessResultPayload(
            ok=True,
            raw_response="Juniper appears relaxed and is smiling slightly.",
            subtitle_source="transcribed",
        ),
        req,
        trigger="manual",
    )

    await svc._publish_event(event)

    assert len(svc.bus.redis.setex_calls) == 1
    key, ttl_seconds, payload = svc.bus.redis.setex_calls[0]
    assert key == "orion:juniper_affect:latest"
    assert ttl_seconds > 0
    assert "Juniper appears relaxed and is smiling slightly." in payload
    assert "transcribed" in payload


@pytest.mark.asyncio
async def test_failed_assessment_does_not_mirror(req):
    """A failed capture must not overwrite a real prior read -- the
    reader's own TTL/max-age gate already ages the prior read out."""
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    event = svc._wrap_event(
        AffectGptAssessResultPayload(ok=False, error="worker timeout", error_code="timeout"),
        req,
        trigger="manual",
    )

    await svc._publish_event(event)

    assert svc.bus.redis.setex_calls == []
    # The real event stream publish still happens regardless -- failures
    # are real, observable events, just not affect-state mirror candidates.
    assert len(svc.bus.published) == 1


@pytest.mark.asyncio
async def test_ok_but_empty_raw_response_does_not_mirror(req):
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    event = svc._wrap_event(
        AffectGptAssessResultPayload(ok=True, raw_response=""), req, trigger="ambient"
    )

    await svc._publish_event(event)

    assert svc.bus.redis.setex_calls == []


@pytest.mark.asyncio
async def test_long_raw_response_is_truncated_before_mirroring(req):
    from app.main import _AFFECT_SUMMARY_MAX_CHARS

    long_response = "x" * (_AFFECT_SUMMARY_MAX_CHARS + 200)
    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    event = svc._wrap_event(
        AffectGptAssessResultPayload(ok=True, raw_response=long_response), req, trigger="manual"
    )

    await svc._publish_event(event)

    import json

    _key, _ttl, payload = svc.bus.redis.setex_calls[0]
    mirrored_summary = json.loads(payload)["summary"]
    assert len(mirrored_summary) <= _AFFECT_SUMMARY_MAX_CHARS
    assert mirrored_summary.endswith("…")


@pytest.mark.asyncio
async def test_mirror_write_failure_does_not_break_the_real_publish(req):
    """The mirror write is additive and fail-open -- a Redis problem there
    must never prevent the real orion:affectgpt:assessment publish."""

    class _RaisingRedis:
        async def setex(self, key, ttl_seconds, payload):
            raise ConnectionError("redis unreachable")

    svc = JuniperAffectiveStateService()
    svc.bus = FakeBus()
    svc.bus.redis = _RaisingRedis()
    event = svc._wrap_event(
        AffectGptAssessResultPayload(ok=True, raw_response="calm"), req, trigger="manual"
    )

    await svc._publish_event(event)  # must not raise

    assert len(svc.bus.published) == 1
