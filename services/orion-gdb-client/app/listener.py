import json
import time
from orion.core.bus.service import OrionBus
from app.settings import settings
from app.gdb_client import build_graph, push_graph
from app.utils import logger


def listener_worker(bus: OrionBus):
    """
    Worker that subscribes to OrionBus and processes messages into GraphDB.
    Runs in its own thread started from main.py.
    """
    if not bus or not bus.enabled:
        logger.error("Bus is not initialized or disabled. Listener cannot start.")
        return

    channel = settings.SUBSCRIBE_CHANNEL
    logger.info(f"Starting GDB listener on channel: {channel}")

    try:
        for data in bus.subscribe(channel):          # ✅ fixed
            entry_id = data.pop("id", None)
            if not entry_id:
                logger.warning("Skipping message without 'id'")
                continue

            try:
                g = build_graph(entry_id, data)
                push_graph(g)
                logger.info("✅ Ingested %s → GraphDB (%d triples)", entry_id, len(g))
            except Exception as e:
                logger.exception("❌ Error pushing to GraphDB: %s", e)
                time.sleep(1)
    except Exception as e:
        logger.exception("💥 Listener fatal: %s", e)
