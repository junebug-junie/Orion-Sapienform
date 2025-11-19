# app/bus_listener.py
import threading
import logging
from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL,
    ORION_BUS_ENABLED,
    CHANNEL_BRAIN_INTAKE,
    CHANNEL_TTS_INTAKE,
    CHANNEL_CORTEX_EXEC_INTAKE
)
from app.processor import process_brain_or_cortex, process_tts_request

logger = logging.getLogger(__name__)


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

    for message in bus.subscribe(CHANNEL_BRAIN_INTAKE, CHANNEL_TTS_INTAKE, CHANNEL_CORTEX_EXEC_INTAKE):
        if message["type"] != "message":
            continue

        try:
            data = message["data"]
            channel = message["channel"]

            if not isinstance(data, dict):
                logger.warning(f"Received non-dict message on {channel}: {data}")
                continue

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
