import time
from orion.core.bus.service import OrionBus
from app.settings import settings
from app.gdb_client import process_raw_collapse, process_enrichment
from app.utils import logger

def listener_worker():
    """
    Creates its own thread-local bus connection and processes messages from multiple
    channels, routing them to the appropriate handler.
    """
    # Create the OrionBus instance INSIDE the thread for thread-safety.
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    if not bus.enabled:
        logger.error("Bus is not initialized or disabled. Listener cannot start.")
        return

    # A dictionary mapping channels to the function that should process them.
    channels_to_handlers = {
        settings.CHANNEL_COLLAPSE_TRIAGE: process_raw_collapse,
        settings.CHANNEL_TAGS_ENRICHED: process_enrichment,
    }
    
    logger.info(f"📡 Starting GDB listener on channels: {list(channels_to_handlers.keys())}")

    try:
        # The updated OrionBus can subscribe to a list of channels and will yield
        # a message object that includes the source channel.
        for message in bus.subscribe(*channels_to_handlers.keys()):
            channel = message.get("channel")
            data = message.get("data")
            handler = channels_to_handlers.get(channel)

            if not all([channel, data, handler]):
                logger.warning(f"Skipping malformed message or message from unhandled channel: {message}")
                continue

            entry_id = data.get("id")
            if not entry_id:
                logger.warning("Skipping message without 'id' on channel %s", channel)
                continue

            try:
                # Call the correct handler based on the channel the message came from
                triples_ingested = handler(entry_id, data)
                logger.info("✅ Ingested %s from %s → GraphDB (%d triples)", entry_id, channel, triples_ingested)

                # Publish confirmation
                bus.publish(
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
                bus.publish(
                    settings.CHANNEL_RDF_ERROR,
                    {"id": entry_id, "status": "error", "error": str(e)},
                )
                time.sleep(1)

    except Exception as e:
        logger.exception("💥 Listener fatal error: %s", e)

