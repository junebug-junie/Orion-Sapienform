import logging
import threading
import json
from orion.core.bus.service import OrionBus
from .settings import settings
from .processor import process_rag_request

logger = logging.getLogger(settings.SERVICE_NAME)

def listener_worker():
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error("Bus connection failed. RAG listener thread exiting.")
        return

    listen_channel = settings.SUBSCRIBE_CHANNEL_RAG_REQUEST
    logger.info(f"👂 Subscribing to RAG request channel: {listen_channel}")

    for message in bus.subscribe(listen_channel):
        if not isinstance(message, dict) or message.get('type') != 'message':
            continue

        try:
            data = message['data']
            
            if not isinstance(data, dict):
                logger.warning(f"Received non-dict data on {listen_channel}. Skipping.")
                continue
            
            logger.info(f"Received RAG request for query: \"{data.get('query')[:50]}...\"")
            
            # --- FIX: Do NOT pass the 'bus' object. ---
            # The processor thread will create its own.
            threading.Thread(target=process_rag_request, args=(data,), daemon=True).start()
            # --- END FIX ---

        except Exception as e:
            logger.error(f"Error processing message in listener: {e}", exc_info=True)
