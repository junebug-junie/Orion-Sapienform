# services/orion-brain/app/bus_listener.py

import threading
import logging

from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL,
    ORION_BUS_ENABLED,
    CHANNEL_TTS_INTAKE,
    CHANNEL_CORTEX_EXEC_INTAKE,
)
from app.processor import process_brain_or_cortex, process_tts_request

logger = logging.getLogger(__name__)


def listener_worker():
    """
    Subscribes to:
      - cortex exec intake (semantic-layer exec_step requests)
      - TTS intake
    and spawns a worker thread per message.

    NOTE:
    - Generic brain LLM RPC via CHANNEL_BRAIN_INTAKE has been removed.
      All generic LLM calls should now go through the Orion LLM Gateway
      (LLMGatewayService) instead of directly through Brain.
    """
    bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    if not bus.enabled:
        logger.error("Bus is disabled. Listener thread exiting.")
        return

    logger.info(
        "👂 Subscribing to cortex exec intake: %s, tts intake: %s",
        CHANNEL_CORTEX_EXEC_INTAKE,
        CHANNEL_TTS_INTAKE,
    )

    # Cortex exec + TTS only. Generic brain intake is intentionally not subscribed.
    for message in bus.subscribe(
        CHANNEL_TTS_INTAKE,
        CHANNEL_CORTEX_EXEC_INTAKE,
    ):
        if message.get("type") != "message":
            continue

        try:
            data = message.get("data")

            # 1. Capture the raw channel first
            raw_channel = message.get("channel")

            # 2. Decode bytes to string
            if isinstance(raw_channel, bytes):
                channel = raw_channel.decode("utf-8")
            else:
                channel = str(raw_channel)

            if not isinstance(data, dict):
                logger.warning("Received non-dict message on %s: %r", channel, data)
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

            # --- Cortex exec (LLMGatewayService) goes through the unified router ---
            if channel == CHANNEL_CORTEX_EXEC_INTAKE:
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

            else:
                # This should not normally happen; log so we can see misrouted traffic.
                logger.warning(
                    "Received message on unexpected channel %s: %r",
                    channel,
                    data,
                )

        except Exception as e:
            logger.error("Error processing bus message: %s", e, exc_info=True)
