# services/orion-whisper-tts/app/stt_worker.py

from __future__ import annotations

import asyncio
import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.tts import STTRequestPayload, STTResultPayload

from .settings import settings
from .stt import STTEngine

logger = logging.getLogger("orion-whisper-tts.stt")

_stt_engine: STTEngine | None = None


def get_stt_engine() -> STTEngine:
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = STTEngine()
    return _stt_engine


async def stt_listener_worker(bus: OrionBusAsync) -> None:
    if not bus.enabled:
        logger.error("Bus is disabled. Whisper/STT listener exiting.")
        return

    intake_channel = "orion:stt:intake"
    logger.info("👂 Subscribing to STT intake: %s", intake_channel)

    async with bus.subscribe(intake_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg.get("data")
            decoded = bus.codec.decode(data)

            if not decoded.ok:
                logger.warning("Decode failed on %s: %s", intake_channel, decoded.error)
                continue

            envelope = decoded.envelope
            correlation_id = str(envelope.correlation_id)
            result_channel = envelope.reply_to or f"orion:stt:result:{correlation_id}"

            try:
                request = STTRequestPayload.model_validate(envelope.payload)
            except Exception as exc:
                logger.error("[%s] Invalid STT payload: %s", correlation_id, exc, exc_info=True)
                continue

            client_meta = None
            if request.options and isinstance(request.options, dict):
                client_meta = request.options.get("client_audio_meta")
            if client_meta:
                logger.info(
                    "[%s] Processing STT request -> %s client_audio_meta=%s",
                    correlation_id,
                    result_channel,
                    client_meta,
                )
            else:
                logger.info("[%s] Processing STT request -> %s", correlation_id, result_channel)

            try:
                loop = asyncio.get_running_loop()

                def _transcribe() -> tuple[str, dict]:
                    engine = get_stt_engine()
                    client_meta = None
                    if request.options and isinstance(request.options, dict):
                        raw = request.options.get("client_audio_meta")
                        if isinstance(raw, dict):
                            client_meta = raw
                    return engine.transcribe(
                        request.audio_b64,
                        language=request.language or "en",
                        audio_format=request.format or "wav",
                        client_audio_meta=client_meta,
                    )

                text, meta = await asyncio.wait_for(
                    loop.run_in_executor(None, _transcribe),
                    timeout=float(settings.whisper_tts_stt_timeout_sec),
                )

                if client_meta and isinstance(meta, dict):
                    meta = {**meta, "client": client_meta}
                result = STTResultPayload(text=text, metadata=meta)
                response_envelope = envelope.derive_child(
                    kind="stt.transcribe.result",
                    source=ServiceRef(
                        name=settings.service_name,
                        version=settings.service_version,
                    ),
                    payload=result,
                    reply_to=None,
                )
                await bus.publish(result_channel, response_envelope)
                logger.info(
                    "[%s] Sent STT result (len=%d peak=%s)",
                    correlation_id,
                    len(text),
                    meta.get("peak"),
                )
            except Exception as exc:
                logger.error("[%s] FAILED to transcribe STT: %s", correlation_id, exc, exc_info=True)
                try:
                    error_envelope = envelope.derive_child(
                        kind="system.error",
                        source=ServiceRef(
                            name=settings.service_name,
                            version=settings.service_version,
                        ),
                        payload={
                            "error": "stt_transcribe_failed",
                            "details": str(exc),
                        },
                        reply_to=None,
                    )
                    await bus.publish(result_channel, error_envelope)
                except Exception:
                    pass
