# app/bus_listener.py
import threading
import logging
import json

from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL,
    ORION_BUS_ENABLED,
    CHANNEL_BRAIN_INTAKE,
    CHANNEL_TTS_INTAKE,
    CHANNEL_CORTEX_EXEC_INTAKE,
)
from app.processor import process_brain_or_cortex, process_tts_request

logger = logging.getLogger(__name__)


def _coerce_to_dict(raw):
    """
    Normalize bus payloads so Brain can handle:
      - dict (already parsed)
      - JSON string
      - bytes / bytearray (JSON-encoded)
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            logger.warning(
                "Received bytes payload that is not JSON-decodable: %r",
                raw[:120],
            )
            return None

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            logger.warning(
                "Received string payload that is not JSON-decodable: %r",
                raw[:120],
            )
            return None

    logger.warning("Received non-object message payload of type %r", type(raw))
    return None


def listener_worker():
    """
    Subscribes to:
      - main brain intake (generic LLM RPC)
      - cortex exec intake (semantic-layer exec_step requests)
      - TTS intake
    and spawns a worker thread per message.
    """
    bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    if not bus.enabled:
        logger.error("Bus is disabled. Listener thread exiting.")
        return

    logger.info(
        f"👂 Subscribing to brain intake: {CHANNEL_BRAIN_INTAKE}, "
        f"cortex exec intake: {CHANNEL_CORTEX_EXEC_INTAKE}, "
        f"tts intake: {CHANNEL_TTS_INTAKE}"
    )

    for message in bus.subscribe(
        CHANNEL_BRAIN_INTAKE,
        CHANNEL_TTS_INTAKE,
        CHANNEL_CORTEX_EXEC_INTAKE,
    ):
        if message.get("type") != "message":
            continue

        try:
            channel = message.get("channel")
            raw_data = message.get("data")

            data = _coerce_to_dict(raw_data)
            if data is None:
                # Already logged inside _coerce_to_dict
                continue

            # 🔍 DEBUG: what the brain actually receives from the bus
            try:
                trace_id = data.get("trace_id", "no-trace")
                hist = data.get("history") or []
                logger.warning(
                    "[%s] INTAKE payload snapshot: channel=%s history_len=%d keys=%s",
                    trace_id,
                    channel,
                    len(hist),
                    list(data.keys()),
                )
            except Exception:
                logger.warning(
                    "INTAKE payload snapshot failed for message on %s",
                    channel,
                    exc_info=True,
                )

            # --- Generic brain RPC + Cortex exec both go through the same router ---
            if channel in (CHANNEL_BRAIN_INTAKE, CHANNEL_CORTEX_EXEC_INTAKE):
                threading.Thread(
                    target=process_brain_or_cortex,
                    args=(data,),
                    daemon=True,
                ).start()

            # --- TTS goes through the TTS pipeline ---
            elif channel == CHANNEL_TTS_INTAKE:
                threading.Thread(
                    target=process_tts_request,
                    args=(data,),
                    daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Error processing bus message: {e}", exc_info=True)
