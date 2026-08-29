from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


def _bus_with_mock_redis() -> tuple[OrionBusAsync, AsyncMock]:
    bus = OrionBusAsync("redis://unused:6379/0", track_velocity=False)
    redis = AsyncMock()
    bus._redis = redis
    return bus, redis


def _envelope(*, kind: str, payload) -> BaseEnvelope:
    return BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="test-service", version="0"),
        payload=payload,
    )


@pytest.mark.asyncio
async def test_system_error_reply_publishes_on_a_result_channel_locked_to_a_different_schema() -> None:
    """Regression test for 2026-08-29: whisper-tts hit a CUDA OOM, tried to
    publish a system.error reply on orion:tts:result:<corr> (schema-locked to
    TTSResultPayload, which requires audio_b64), that publish silently raised
    a ValueError that the worker's own except-and-publish-error handler
    swallowed, and the hub burned the full 180s HUB_TTS_TIMEOUT_SEC before
    reporting a bare timeout with no real cause. The error payload must be
    allowed through so the caller sees the real failure instead of a timeout.
    """
    bus, redis = _bus_with_mock_redis()
    error_envelope = _envelope(
        kind="system.error",
        payload={"error": "tts_synthesis_failed", "details": "CUDA out of memory"},
    )

    await bus.publish("orion:tts:result:cc50e621-37e3-4c95-9484-e145fca32564", error_envelope)

    redis.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_error_payload_is_still_validated_against_the_result_channel_schema() -> None:
    """The exemption must be narrow: an actual (non-error) reply that doesn't
    match the channel's declared schema should still be rejected."""
    bus, _redis = _bus_with_mock_redis()
    malformed_result = _envelope(
        kind="tts.synthesize.result",
        # Missing the required `audio_b64` field of TTSResultPayload.
        payload={"content_type": "audio/wav"},
    )

    with pytest.raises(ValueError, match="Payload validation failed"):
        await bus.publish("orion:tts:result:cc50e621-37e3-4c95-9484-e145fca32564", malformed_result)


@pytest.mark.asyncio
async def test_system_error_on_the_dedicated_error_channel_is_still_validated() -> None:
    """The redirect only covers a system.error envelope landing on a
    *different* channel's success schema. orion:system:error is itself
    schema-locked to SystemErrorV1 -- that enforcement must stay intact."""
    bus, _redis = _bus_with_mock_redis()
    # SystemErrorV1.error is Optional[str]; a nested dict can't coerce to str.
    malformed_payload = _envelope(kind="system.error", payload={"error": {"bad": "shape"}})

    with pytest.raises(ValueError, match="Payload validation failed"):
        await bus.publish("orion:system:error", malformed_payload)


@pytest.mark.asyncio
async def test_malformed_system_error_reply_is_still_rejected_on_a_different_schema_channel() -> None:
    """The fix redirects validation to SystemErrorV1 rather than skipping it
    outright -- a genuinely malformed error payload (not just a mismatched
    schema) must still be caught, even on a channel whose real schema is
    something else entirely."""
    bus, _redis = _bus_with_mock_redis()
    malformed_payload = _envelope(kind="system.error", payload={"error": {"bad": "shape"}})

    with pytest.raises(ValueError, match="Payload validation failed"):
        await bus.publish("orion:tts:result:cc50e621-37e3-4c95-9484-e145fca32564", malformed_payload)


@pytest.mark.asyncio
async def test_system_error_v1_kind_is_also_recognized() -> None:
    """orion/harness/finalize.py's emit_harness_finalize_system_error() uses
    kind="system.error.v1", not the bare "system.error" every other producer
    uses. Its channel is env-overridable (CHANNEL_SYSTEM_ERROR/
    ORION_ERROR_CHANNEL) -- if ever pointed at a schema-locked channel other
    than the SystemErrorV1 default, this variant must still be recognized."""
    bus, redis = _bus_with_mock_redis()
    error_envelope = _envelope(
        kind="system.error.v1",
        payload={"error": "finalize_failed", "phase": "orion_voice_finalize"},
    )

    await bus.publish("orion:tts:result:cc50e621-37e3-4c95-9484-e145fca32564", error_envelope)

    redis.publish.assert_awaited_once()
