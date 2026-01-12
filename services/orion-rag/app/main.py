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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "vector_db_collection": vector_store.collection_name if vector_store.collection else "Not Connected",
    }

