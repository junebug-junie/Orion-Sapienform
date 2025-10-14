import httpx
import json
import logging
import time
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus.service import OrionBus

logger = logging.getLogger(settings.SERVICE_NAME)

def _push_to_graphdb(nt_data: str, graph_name: str, event: dict):
    """
    Pushes N-Triples data to GraphDB with a retry mechanism.
    """
    if not nt_data or not graph_name:
        logger.warning(f"Skipping push for event {event.get('id')} due to empty RDF data.")
        return

    url = f"{settings.GRAPHDB_URL}/repositories/{settings.GRAPHDB_REPO}/statements?context=<{graph_name}>"
    headers = {"Content-Type": "application/n-triples"}
    
    for attempt in range(settings.RETRY_LIMIT):
        try:
            with httpx.Client(timeout=10) as client:
                res = client.post(url, content=nt_data, headers=headers)
                res.raise_for_status() # Raise exception for non-2xx responses
                logger.info(f"✅ RDF inserted ({event.get('id')}) → {graph_name}")
                return # Success
        except httpx.HTTPStatusError as e:
            logger.warning(f"⚠️ Insert failed (attempt {attempt + 1}/{settings.RETRY_LIMIT}): {e.response.status_code} {e.response.text}")
        except Exception as e:
            logger.error(f"❌ GraphDB connection error (attempt {attempt + 1}/{settings.RETRY_LIMIT}): {e}")
        time.sleep(settings.RETRY_INTERVAL)
    
    logger.error(f"🚨 Failed to push event {event.get('id')} after {settings.RETRY_LIMIT} attempts.")
    # Here you might want to publish an error to the bus or save to a dead-letter queue.


def listener_worker():
    """
    A single, efficient worker that creates its own bus connection and subscribes
    to all relevant channels at once. This is a robust and thread-safe pattern.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error("Bus connection failed. RDF-Writer listener thread exiting.")
        return

    channels_to_subscribe = settings.get_all_subscribe_channels()
    logger.info(f"👂 Subscribing to channels: {channels_to_subscribe}")

    # The new OrionBus can subscribe to a list of channels and will yield
    # a message object that includes the source channel.
    for message in bus.subscribe(*channels_to_subscribe):
        source_channel = message.get("channel")
        data = message.get("data")

        if not source_channel or not data:
            continue
        
        logger.debug(f"📥 Received event from {source_channel}: {data.get('id')}")

        try:
            nt_data, graph_name = build_triples(data)
            _push_to_graphdb(nt_data, graph_name, data)
        except Exception as e:
            logger.exception(f"❌ Unhandled error processing event {data.get('id')} from {source_channel}: {e}")

