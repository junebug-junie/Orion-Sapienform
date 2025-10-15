import logging
import threading

from fastapi import FastAPI
from .settings import settings
from .vector_store import vector_store
from .listener import listener_worker # Import the refactored worker

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "vector_db_collection": vector_store.collection_name if vector_store.collection else "Not Connected",
    }

