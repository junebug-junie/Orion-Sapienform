import threading
from fastapi import FastAPI
from app.settings import settings
from app.gdb_client import wait_for_graphdb, ensure_repo_exists
from app.listener import listener_worker
from orion.core.bus.service import OrionBus
from app.utils import logger

app = FastAPI(title=settings.service_name, version=settings.service_version)

# Keep a reference here (not imported from listener)
bus: OrionBus | None = None


@app.on_event("startup")
def startup_event():
    """
    Initialize GraphDB and OrionBus, then start the listener thread.
    """
    global bus
    logger.info("🚀 Starting %s (%s)", settings.service_name, settings.service_version)

    # Wait until GraphDB is reachable
    wait_for_graphdb()
    ensure_repo_exists()

    if settings.ORION_BUS_ENABLED:
        logger.info("Initializing OrionBus connection → %s", settings.ORION_BUS_URL)
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

        if bus:
            t = threading.Thread(
                target=listener_worker,
                args=(bus,),                # ✅ Pass the initialized bus
                daemon=True,
                #name="gdb-client-listener",
            )
            t.start()
            logger.info(
                "Listener thread started (channel: %s)", settings.SUBSCRIBE_CHANNEL
            )
        else:
            logger.error("❌ Failed to initialize OrionBus. Listener not started.")
    else:
        logger.warning("⚠️ OrionBus disabled — listener will not start.")


@app.on_event("shutdown")
def shutdown_event():
    """
    Gracefully disconnect the OrionBus on service shutdown.
    """
    global bus
    if bus:
        logger.info("🛑 Shutting down OrionBus connection.")
        try:
            bus.disconnect()
        except Exception as e:
            logger.error("Error disconnecting bus: %s", e)


@app.get("/health")
def health():
    """
    Basic liveness probe for Compose/Kubernetes.
    """
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "graphdb_repo": settings.graphdb_repo,
        "bus_enabled": settings.ORION_BUS_ENABLED,
        "bus_channel": settings.SUBSCRIBE_CHANNEL,
        "status": "ok",
    }
