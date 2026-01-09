import asyncio
import logging
import threading
import json

from orion.core.bus.async_service import OrionBusAsync
from .settings import settings
from .processor import process_rag_request

logger = logging.getLogger(settings.SERVICE_NAME)

def listener_worker() -> None:
    asyncio.run(_listener_worker_async())


async def _listener_worker_async() -> None:
    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)
    await bus.connect()

    listen_channel = settings.SUBSCRIBE_CHANNEL_RAG_REQUEST
    logger.info(f"👂 Subscribing to RAG request channel: {listen_channel}")

    try:
        async with bus.subscribe(listen_channel) as pubsub:
            async for message in bus.iter_messages(pubsub):
                try:
                    data = message["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", "ignore")
                    if isinstance(data, str):
                        data = json.loads(data)
                    if not isinstance(data, dict):
                        logger.warning(f"Received non-dict data on {listen_channel}. Skipping.")
                        continue

                    logger.info(f"Received RAG request for query: \"{data.get('query')[:50]}...\"")

                    threading.Thread(target=process_rag_request, args=(data,), daemon=True).start()
                except Exception as e:
                    logger.error(f"Error processing message in listener: {e}", exc_info=True)
    finally:
        await bus.close()
