import threading
from fastapi import FastAPI
from app.settings import settings
from app.gdb_client import wait_for_graphdb, ensure_repo_exists
from app.listener import listener_worker
from app.utils import logger

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)


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
    else:
        logger.warning("⚠️ OrionBus disabled — listener will not start.")


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

