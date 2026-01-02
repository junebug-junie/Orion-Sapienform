# services/orion-whisper-tts/app/tts_worker.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, cast

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload

from .settings import settings
from .tts import TTSEngine

logger = logging.getLogger("orion-whisper-tts.worker")

_tts_engine: TTSEngine | None = None


def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine


async def process_tts_request(
    bus: OrionBusAsync,
    envelope: BaseEnvelope,
    raw_payload: Dict[str, Any],
) -> None:
    """
    Handles a single TTS request from the bus.
    Supports both legacy dict payloads and Typed Envelopes.
    """
    # 1. Extract request data (Hybrid Access Pattern)
    request_data: Optional[TTSRequestPayload] = None
    reply_to = envelope.reply_to

    # Try to parse as typed request
    if envelope.kind == "tts.synthesize.request":
        try:
            request_data = TTSRequestPayload.model_validate(envelope.payload)
        except Exception as e:
            logger.warning(
                "[%s] Invalid typed payload for %s: %s",
                envelope.correlation_id,
                envelope.kind,
                e,
            )

    # Fallback: legacy/loose payload inspection
    if request_data is None:
        # Legacy payload: {"text": "...", "response_channel": "..."}
        text = raw_payload.get("text")
        if not text and envelope.payload:
            text = envelope.payload.get("text")

        if text:
            # Map legacy fields to model
            request_data = TTSRequestPayload(
                text=text,
                voice_id=raw_payload.get("voice_id"),
                language=raw_payload.get("language"),
            )

            # Legacy reply channel handling if envelope.reply_to is missing
            if not reply_to:
                reply_to = raw_payload.get("response_channel")

    if not request_data:
        logger.warning("[%s] Could not parse TTS request from payload.", envelope.correlation_id)
        return

    if not reply_to:
        logger.warning("[%s] No reply_to/response_channel specified. Dropping.", envelope.correlation_id)
        return

    logger.info(
        "[%s] Processing TTS request (len=%d) -> %s",
        envelope.correlation_id,
        len(request_data.text),
        reply_to,
    )

    # 2. Synthesize (run in thread pool)
    try:
        loop = asyncio.get_running_loop()

        def _synthesize():
            engine = get_tts_engine()
            return engine.synthesize_to_b64(request_data.text)

        audio_b64 = await loop.run_in_executor(None, _synthesize)

        # 3. Create Result Payload
        result = TTSResultPayload(
            audio_b64=audio_b64,
            content_type="audio/wav",
            # duration_sec could be calculated if TTS engine returns it
        )

        # 4. Reply
        # Create response envelope (child of request)
        response_envelope = envelope.derive_child(
            kind="tts.synthesize.result",
            source=ServiceRef(
                name=settings.service_name,
                version=settings.service_version,
            ),
            payload=result,
            reply_to=None, # End of RPC chain
        )

        await bus.publish(reply_to, response_envelope)

        logger.info(
            "[%s] Sent TTS reply to %s (bytes=%d)",
            envelope.correlation_id,
            reply_to,
            len(audio_b64),
        )

    except Exception as e:
        logger.error(
            "[%s] FAILED to synthesize TTS: %s",
            envelope.correlation_id,
            e,
            exc_info=True,
        )
        # Publish error if possible
        try:
            error_envelope = envelope.derive_child(
                kind="system.error",
                source=ServiceRef(
                    name=settings.service_name,
                    version=settings.service_version,
                ),
                payload={
                    "error": "tts_synthesis_failed",
                    "details": str(e)
                },
            )
            await bus.publish(reply_to, error_envelope)
        except Exception:
            pass


async def listener_worker(bus: OrionBusAsync) -> None:
    """
    Subscribes to CHANNEL_TTS_INTAKE and processes messages.
    """
    if not bus.enabled:
        logger.error("Bus is disabled. Whisper/TTS listener exiting.")
        return

    logger.info(
        "👂 Subscribing to TTS intake: %s",
        settings.channel_tts_intake,
    )

    async with bus.subscribe(settings.channel_tts_intake) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            try:
                # Decode using Codec
                data = msg.get("data")
                decoded = bus.codec.decode(data)

                if not decoded.ok:
                    logger.warning(
                        "Decode failed on %s: %s",
                        settings.channel_tts_intake,
                        decoded.error
                    )
                    continue

                # Process asynchronously (fire and forget task to not block listener)
                asyncio.create_task(
                    process_tts_request(bus, decoded.envelope, decoded.raw)
                )

            except Exception:
                logger.error("Error processing message loop", exc_info=True)
