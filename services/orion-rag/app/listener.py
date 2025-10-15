import logging
import threading
from orion.core.bus.service import OrionBus
from .settings import settings
from .processor import process_rag_request

logger = logging.getLogger(settings.SERVICE_NAME)

def listener_worker():
    """
    Creates a thread-local bus connection, listens for incoming RAG requests,
    and spawns a new thread to handle each one concurrently.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error("Bus connection failed. RAG listener thread exiting.")
        return

    listen_channel = settings.SUBSCRIBE_CHANNEL_RAG_REQUEST
    logger.info(f"👂 Subscribing to RAG request channel: {listen_channel}")

    for message in bus.subscribe(listen_channel):
        data = message.get("data")
        if not data:
            continue
        
        logger.info(f"Received RAG request: {data.get('query')}")
        # Process each request in its own thread to not block the main listener.
        # The main bus instance is passed for the initial publish call.
        threading.Thread(target=process_rag_request, args=(bus, data), daemon=True).start()
