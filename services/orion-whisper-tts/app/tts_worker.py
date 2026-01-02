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
    is_legacy = False
    legacy_trace_id: Optional[str] = None

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

    # Check for legacy request
    elif envelope.kind == "legacy.message":
        is_legacy = True

    # Fallback/Legacy payload inspection
    if request_data is None:
        # Legacy payload: {"text": "...", "response_channel": "..."}
        # Note: If it was wrapped by Codec, envelope.payload IS the legacy dict.
        text = None
        if envelope.payload:
            text = envelope.payload.get("text")

        # Also check raw_payload just in case Codec behaved differently
        if not text and raw_payload:
            text = raw_payload.get("text")

        if text:
            # Map legacy fields to model
            # We assume it is legacy if we fell back to this parsing
            is_legacy = True

            payload_src = envelope.payload if envelope.payload else raw_payload

            voice_id = payload_src.get("voice_id")
            language = payload_src.get("language")
            legacy_trace_id = payload_src.get("trace_id")

            request_data = TTSRequestPayload(
                text=text,
                voice_id=voice_id,
                language=language,
            )

            # Legacy reply channel handling if envelope.reply_to is missing
            if not reply_to:
                reply_to = payload_src.get("response_channel")

    if not request_data:
        logger.warning("[%s] Could not parse TTS request from payload.", envelope.correlation_id)
        return

    if not reply_to:
        logger.warning("[%s] No reply_to/response_channel specified. Dropping.", envelope.correlation_id)
        return

    logger.info(
        "[%s] Processing TTS request (len=%d, legacy=%s) -> %s",
        envelope.correlation_id,
        len(request_data.text),
        is_legacy,
        reply_to,
    )

    # 2. Synthesize (run in thread pool)
    try:
        loop = asyncio.get_running_loop()

        def _synthesize():
            engine = get_tts_engine()
            return engine.synthesize_to_b64(request_data.text)

        audio_b64 = await loop.run_in_executor(None, _synthesize)

        # 3. Reply
        if is_legacy:
            # Legacy Reply: Raw Dict
            # Hub expects: {"audio_b64": "...", "trace_id": "..."}
            # Use original trace_id if available, else correlation_id
            final_trace_id = legacy_trace_id or str(envelope.correlation_id)

            legacy_reply = {
                "trace_id": final_trace_id,
                "audio_b64": audio_b64,
                "mime_type": "audio/wav",
            }
            # Publish as dict -> JSON on wire
            await bus.publish(reply_to, legacy_reply)

        else:
            # Typed Reply: Envelope
            result = TTSResultPayload(
                audio_b64=audio_b64,
                content_type="audio/wav",
            )

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
            if is_legacy:
                 await bus.publish(reply_to, {"error": str(e), "ok": False})
            else:
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
