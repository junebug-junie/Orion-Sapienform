import logging
import threading

from fastapi import FastAPI
from .settings import settings
from .vector_store import vector_store
from .listener import listener_worker # Import the refactored worker
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1
import asyncio
import uuid
from datetime import datetime, timezone

# SystemHealthV1 requires boot_id and last_seen_ts. Constructing it without them
# raises inside the heartbeat loop's own try/except, which logs a warning and
# sleeps -- so the service looks alive while publishing no heartbeat at all.
# Confirmed live 2026-08-29 on orion-gpu-cluster-power: one failure per 30s tick,
# indefinitely, and nothing downstream noticed the silence.
# BOOT_ID identifies THIS process run, so a consumer can tell a restart from a
# continuous uptime (same convention as services/orion-whisper-tts/app/main.py).
BOOT_ID = str(uuid.uuid4())


# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)


@app.on_event("startup")
def startup_event():
    """
    Initializes the vector store and starts the main bus listener thread.
    """
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    
    # Initialize the vector store client (connects to DB, loads model)
    vector_store.initialize()
    
    if settings.ORION_BUS_ENABLED:
        logger.info("Starting listener thread...")
        # Start the listener worker from the new module
        threading.Thread(target=listener_worker, daemon=True).start()

        # Start Heartbeat Thread
        threading.Thread(target=heartbeat_worker_thread, daemon=True).start()
    else:
        logger.warning("Bus is disabled; RAG service will be idle.")

def heartbeat_worker_thread():
    """Synchronous worker for heartbeats."""

    async def _run():
        bus = OrionBusAsync(settings.ORION_BUS_URL)
        await bus.connect()
        try:
            while True:
                try:
                    payload = SystemHealthV1(
                        service=settings.SERVICE_NAME,
                        version=settings.SERVICE_VERSION,
                        node="rag-node",
                        status="ok",
                        boot_id=BOOT_ID,
                        last_seen_ts=datetime.now(timezone.utc),
                        # heartbeat_interval_sec must match this loop's real period. Left at the
                        # schema default of 10.0, orion-equilibrium-service computes
                        # grace = interval * EQUILIBRIUM_GRACE_MULTIPLIER (3.0) = 30.0s and marks the
                        # service "down" once delta > grace (service.py's status check). Publishing
                        # every 30s leaves ZERO margin, so any event-loop delay or bus latency flips
                        # it to down, emits a spurious transition and pushes distress_score.
                        heartbeat_interval_sec=30.0,
                    ).model_dump(mode="json")

                    await bus.publish("orion:system:health", BaseEnvelope(
                        kind="system.health.v1",
                        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                        payload=payload
                    ))
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

                await asyncio.sleep(30)
        finally:
            await bus.close()

    # Create a new loop for this thread
    asyncio.run(_run())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "vector_db_collection": vector_store.collection_name if vector_store.collection else "Not Connected",
    }

