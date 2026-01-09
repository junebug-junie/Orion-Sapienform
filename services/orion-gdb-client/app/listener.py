import asyncio
import time

from orion.core.bus.async_service import OrionBusAsync
from app.settings import settings
from app.gdb_client import process_raw_collapse, process_enrichment
from app.utils import logger

def listener_worker() -> None:
    asyncio.run(_listener_worker_async())


async def _listener_worker_async() -> None:
    """
    Creates its own thread-local bus connection and processes messages from multiple
    channels, routing them to the appropriate handler.
    """
    # Create the OrionBus instance INSIDE the thread for thread-safety.
    if not settings.ORION_BUS_ENABLED:
        logger.error("Bus is not initialized or disabled. Listener cannot start.")
        return

    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    await bus.connect()

    # A dictionary mapping channels to the function that should process them.
    channels_to_handlers = {
        settings.CHANNEL_COLLAPSE_TRIAGE: process_raw_collapse,
        settings.CHANNEL_TAGS_ENRICHED: process_enrichment,
    }
    
    logger.info(f"📡 Starting GDB listener on channels: {list(channels_to_handlers.keys())}")

    try:
        # The updated OrionBus can subscribe to a list of channels and will yield
        # a message object that includes the source channel.
        async with bus.subscribe(*channels_to_handlers.keys()) as pubsub:
            async for message in bus.iter_messages(pubsub):
                channel = message.get("channel")
                data = message.get("data")
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", "ignore")
                if isinstance(data, str):
                    try:
                        import json

                        data = json.loads(data)
                    except Exception:
                        data = None
                handler = channels_to_handlers.get(channel)

                if not all([channel, data, handler]):
                    logger.warning(f"Skipping malformed message or message from unhandled channel: {message}")
                    continue

                entry_id = data.get("id") if isinstance(data, dict) else None
                if not entry_id:
                    logger.warning("Skipping message without 'id' on channel %s", channel)
                    continue

                try:
                    # Call the correct handler based on the channel the message came from
                    triples_ingested = handler(entry_id, data)
                    logger.info("✅ Ingested %s from %s → GraphDB (%d triples)", entry_id, channel, triples_ingested)

                    # Publish confirmation
                    await bus.publish(
                        settings.CHANNEL_RDF_CONFIRM,
                        {
                            "id": entry_id,
                            "status": "success",
                            "triples": triples_ingested,
                            "source_channel": channel,
                        },
                    )

                except Exception as e:
                    logger.exception("❌ Error processing message for GraphDB: %s", e)
                    await bus.publish(
                        settings.CHANNEL_RDF_ERROR,
                        {"id": entry_id, "status": "error", "error": str(e)},
                    )
                    time.sleep(1)

    except Exception as e:
        logger.exception("💥 Listener fatal error: %s", e)
    finally:
        await bus.close()
