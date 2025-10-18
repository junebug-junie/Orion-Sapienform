import threading
import json
import logging
from orion.core.bus.service import OrionBus
from app.config import ORION_BUS_URL, ORION_BUS_ENABLED, CHANNEL_BRAIN_INTAKE
from app.processor import process_brain_request

logger = logging.getLogger(__name__)

def listener_worker():
    """
    Subscribes to the main brain intake channel and spawns a thread
    for each request. This is the correct thread-safe pattern.
    """
    bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    if not bus.enabled:
        logger.error("Bus is disabled. Listener thread exiting.")
        return

    logger.info(f"👂 Subscribing to main intake channel: {CHANNEL_BRAIN_INTAKE}")
    #bus.subscribe(CHANNEL_BRAIN_INTAKE)

    for message in bus.subscribe(CHANNEL_BRAIN_INTAKE):
        if message['type'] != 'message':
            continue

        try:
            # We assume the OrionBus wrapper already decodes JSON
            data = message['data']
            if not isinstance(data, dict):
                logger.warning(f"Received non-dict message: {data}")
                continue
            
            # Process each request in its own thread
            threading.Thread(target=process_brain_request, args=(data,), daemon=True).start()

        except Exception as e:
            logger.error(f"Error processing bus message: {e}", exc_info=True)
