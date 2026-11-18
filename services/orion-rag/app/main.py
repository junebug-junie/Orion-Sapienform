import logging
import threading

from fastapi import FastAPI
from .settings import settings
from .vector_store import vector_store
from .listener import listener_worker # Import the refactored worker
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly


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
    else:
        logger.warning("Bus is disabled; RAG service will be idle.")

heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Standard SystemHealthV1 heartbeat (orion:system:health) on its own bus
    connection. Replaces a hand-rolled 30s loop in a private thread that reported
    the made-up node name "rag-node". Pattern:
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@app.on_event("startup")
async def start_heartbeat() -> None:
    """Runs after startup_event. A failure is logged, never fatal."""
    global heartbeat_chassis
    if not settings.ORION_BUS_ENABLED:
        return
    try:
        chassis = build_heartbeat_chassis()
        await chassis.start_background()
        heartbeat_chassis = chassis
        logger.info(
            "system_health_heartbeat_started service=%s node=%s interval_sec=%s",
            settings.SERVICE_NAME,
            settings.NODE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None


@app.on_event("shutdown")
async def stop_heartbeat() -> None:
    global heartbeat_chassis
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "vector_db_collection": vector_store.collection_name if vector_store.collection else "Not Connected",
    }

