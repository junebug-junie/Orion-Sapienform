import asyncio
import logging
import threading
import json

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.resilience import run_subscription
from .settings import settings
from .processor import process_rag_request

logger = logging.getLogger(settings.SERVICE_NAME)

def listener_worker() -> None:
    asyncio.run(_listener_worker_async())


async def _listener_worker_async(stop: asyncio.Event | None = None) -> None:
    # No connect() here: subscribe opens its own connection, and a failed startup
    # ping would end this thread before run_subscription could retry it.
    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)

    listen_channel = settings.SUBSCRIBE_CHANNEL_RAG_REQUEST
    logger.info(f"👂 Subscribing to RAG request channel: {listen_channel}")

    def _handle(message: dict) -> None:
        try:
            data = message["data"]
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", "ignore")
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                logger.warning(f"Received non-dict data on {listen_channel}. Skipping.")
                return

            logger.info(f"Received RAG request for query: \"{data.get('query')[:50]}...\"")

            # Thread per request; return None so the next message is read at once.
            threading.Thread(target=process_rag_request, args=(data,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error processing message in listener: {e}", exc_info=True)

    try:
        await run_subscription(bus, listen_channel, _handle, stop_event=stop, name="rag-request")
    finally:
        await bus.close()
