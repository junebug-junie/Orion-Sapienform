# services/orion-whisper-tts/app/tts_worker.py

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict

from orion.core.bus.service import OrionBus

from .settings import settings
from .tts import TTSEngine

logger = logging.getLogger("orion-whisper-tts.worker")

_tts_engine: TTSEngine | None = None


def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine


def process_tts_request(payload: Dict[str, Any]) -> None:
    """
    Handles a single TTS request from the bus.

    Expected payload (same as old Brain contract):

      {
        "trace_id": "...",
        "text": "hello world",
        "response_channel": "orion:tts:rpc:<uuid>",
        "source": "hub"
      }
    """
    trace_id = payload.get("trace_id") or str(uuid.uuid4())
    text = payload.get("text") or ""
    response_channel = payload.get("response_channel")

    if not text:
        logger.warning("[%s] TTS request missing 'text'. Discarding.", trace_id)
        return

    if not response_channel:
        logger.warning("[%s] TTS request missing 'response_channel'. Discarding.", trace_id)
        return

    logger.info(
        "[%s] Processing TTS request (len=%d) -> %s",
        trace_id,
        len(text),
        response_channel,
    )

    try:
        engine = get_tts_engine()
        audio_b64 = engine.synthesize_to_b64(text)

        bus = OrionBus(
            url=settings.orion_bus_url,
            enabled=settings.orion_bus_enabled,
        )

        reply_payload = {
            "trace_id": trace_id,
            "audio_b64": audio_b64,
            "mime_type": "audio/wav",
        }
        bus.publish(response_channel, reply_payload)

        logger.info(
            "[%s] Sent TTS reply to %s (bytes=%d)",
            trace_id,
            response_channel,
            len(audio_b64),
        )

    except Exception as e:
        logger.error(
            "[%s] FAILED to synthesize TTS: %s",
            trace_id,
            e,
            exc_info=True,
        )


def listener_worker() -> None:
    """
    Subscribes to CHANNEL_TTS_INTAKE and spawns a worker thread per message.
    """
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )

    if not bus.enabled:
        logger.error("Bus is disabled. Whisper/TTS listener exiting.")
        return

    logger.info(
        "👂 Subscribing to TTS intake: %s",
        settings.channel_tts_intake,
    )

    for message in bus.subscribe(settings.channel_tts_intake):
        if message.get("type") != "message":
            continue

        raw_channel = message.get("channel")
        if isinstance(raw_channel, bytes):
            channel = raw_channel.decode("utf-8")
        else:
            channel = str(raw_channel)

        data = message.get("data")
        if not isinstance(data, dict):
            logger.warning("Received non-dict message on %s: %r", channel, data)
            continue

        try:
            trace_id = data.get("trace_id", "no-trace")
            logger.info(
                "[%s] INTAKE payload snapshot: channel=%s keys=%s",
                trace_id,
                channel,
                list(data.keys()),
            )
        except Exception:
            logger.warning(
                "INTAKE payload snapshot failed for message on %s",
                channel,
                exc_info=True,
            )

        # All traffic on this service is TTS intake
        threading.Thread(
            target=process_tts_request,
            args=(data,),
            daemon=True,
        ).start()
