import logging
from uuid import uuid4

from orion.core.bus.service import OrionBus

from .settings import settings
from .tts import TTSEngine

logger = logging.getLogger("orion-llm-gateway.tts")

_tts_engine: TTSEngine | None = None


def get_tts_engine() -> TTSEngine:
    """
    Lazy-load a singleton TTSEngine inside the Gateway container.
    """
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine


def process_tts_request(payload: dict):
    """
    Handles a single TTS request from the bus.

    Expected payload (unchanged from Brain):

      {
        "trace_id": "...",
        "text": "hello world",
        "response_channel": "orion:tts:rpc:<uuid>",
        "source": "hub"
      }
    """
    trace_id = payload.get("trace_id") or str(uuid4())
    text = payload.get("text") or ""
    response_channel = payload.get("response_channel")

    if not text:
        logger.warning("[TTS] [%s] Request missing 'text'. Discarding.", trace_id)
        return

    if not response_channel:
        logger.warning(
            "[TTS] [%s] Request missing 'response_channel'. Discarding.",
            trace_id,
        )
        return

    logger.info(
        "[TTS] [%s] Processing TTS request (len=%d) source=%s",
        trace_id,
        len(text),
        payload.get("source"),
    )

    try:
        engine = get_tts_engine()
        audio_b64 = engine.synthesize_to_b64(text)

        reply_bus = OrionBus(
            url=settings.orion_bus_url,
            enabled=settings.orion_bus_enabled,
        )
        reply_payload = {
            "trace_id": trace_id,
            "audio_b64": audio_b64,
            "mime_type": "audio/wav",
        }
        reply_bus.publish(response_channel, reply_payload)
        logger.info(
            "[TTS] [%s] Sent TTS reply to %s",
            trace_id,
            response_channel,
        )

    except Exception as e:
        logger.error(
            "[TTS] [%s] FAILED to synthesize TTS: %s",
            trace_id,
            e,
            exc_info=True,
        )


def tts_listener_worker():
    """
    Subscribe to the TTS intake channel on Orion Bus and
    process requests in-process (single-threaded is fine;
    the TTS engine itself is GPU-bound).
    """
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        logger.error("[TTS] Bus is disabled. TTS listener exiting.")
        return

    channel = settings.channel_tts_intake
    logger.info("👂 [TTS] Subscribing to TTS intake: %s", channel)

    for message in bus.subscribe(channel):
        if message.get("type") != "message":
            continue

        raw_channel = message.get("channel")
        if isinstance(raw_channel, bytes):
            chan = raw_channel.decode("utf-8")
        else:
            chan = str(raw_channel)

        data = message.get("data")
        if not isinstance(data, dict):
            logger.warning(
                "[TTS] Received non-dict message on %s: %r",
                chan,
                data,
            )
            continue

        try:
            process_tts_request(data)
        except Exception as e:
            logger.error(
                "[TTS] Error processing TTS message on %s: %s",
                chan,
                e,
                exc_info=True,
            )
