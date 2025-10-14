import json
import time
from orion.core.bus.service import OrionBus
from app.settings import settings
from app.gdb_client import build_graph, push_graph
from app.utils import logger


def listener_worker(bus: OrionBus):
    """
    Worker that subscribes to OrionBus and processes messages into GraphDB.
    Publishes confirm/error events after each attempt.
    Runs in its own thread started from main.py.
    """
    if not bus or not bus.enabled:
        logger.error("Bus is not initialized or disabled. Listener cannot start.")
        return

    channel = settings.SUBSCRIBE_CHANNEL
    logger.info(f"📡 Starting GDB listener on channel: {channel}")

    try:
        for data in bus.subscribe(channel):
            entry_id = data.pop("id", None)
            if not entry_id:
                logger.warning("Skipping message without 'id'")
                continue

            try:
                g = build_graph(entry_id, data)
                push_graph(g)
                triple_count = len(g)
                logger.info("✅ Ingested %s → GraphDB (%d triples)", entry_id, triple_count)

                # 🔁 Publish confirmation
                bus.publish(
                    settings.CHANNEL_RDF_CONFIRM,
                    {
                        "id": entry_id,
                        "status": "success",
                        "triples": triple_count,
                        "repo": settings.GRAPHDB_REPO,
                        "service": settings.SERVICE_NAME,
                    },
                )

            except Exception as e:
                logger.exception("❌ Error pushing to GraphDB: %s", e)
                bus.publish(
                    settings.CHANNEL_RDF_ERROR,
                    {
                        "id": entry_id or "(unknown)",
                        "status": "error",
                        "error": str(e),
                        "repo": settings.GRAPHDB_REPO,
                        "service": settings.SERVICE_NAME,
                    },
                )
                time.sleep(1)

    except Exception as e:
        logger.exception("💥 Listener fatal: %s", e)
