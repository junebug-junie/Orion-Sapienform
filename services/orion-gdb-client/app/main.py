import uuid
from datetime import datetime, timezone
import threading
from fastapi import FastAPI
from app.settings import settings
from app.gdb_client import wait_for_graphdb, ensure_repo_exists
from app.listener import listener_worker
from app.utils import logger
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1
import asyncio

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

BOOT_ID = str(uuid.uuid4())

@app.on_event("startup")
def startup_event():
    """
    Waits for GraphDB to be ready, ensures the repository exists,
    and starts the bus listener in a background thread.
    """
    logger.info("🚀 Starting %s (%s)", settings.SERVICE_NAME, settings.SERVICE_VERSION)

    wait_for_graphdb()
    ensure_repo_exists()

    if settings.ORION_BUS_ENABLED:
        logger.info("Starting listener thread...")
        # The listener_worker function now creates its own thread-local bus connection.
        # We simply need to start it in a background thread.
        t = threading.Thread(
            target=listener_worker,
            daemon=True,
            name="gdb-client-listener",
        )
        t.start()
        logger.info("Listener thread started.")

        # Start Heartbeat Thread
        hb_thread = threading.Thread(target=heartbeat_worker_thread, daemon=True, name="heartbeat")
        hb_thread.start()
    else:
        logger.warning("⚠️ OrionBus disabled — listener will not start.")

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
                        boot_id=BOOT_ID,
                        last_seen_ts=datetime.now(timezone.utc),
                        node="gdb-node",
                        status="ok"
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


@app.on_event("shutdown")
def shutdown_event():
    """Logs the shutdown of the service."""
    # The bus connection is managed within the daemon thread, so no explicit
    # disconnect is needed here.
    logger.info("🛑 Shutting down service.")


@app.get("/health")
def health():
    """Provides a health check endpoint for the service."""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "graphdb_repo": settings.GRAPHDB_REPO,
        "bus_enabled": settings.ORION_BUS_ENABLED,
        "bus_channels": [
            settings.CHANNEL_COLLAPSE_TRIAGE,
            settings.CHANNEL_TAGS_ENRICHED,
        ],
        "status": "ok",
    }

