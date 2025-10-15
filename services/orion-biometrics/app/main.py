import threading
import logging
import json
from fastapi import FastAPI

from orion.core.sql_router.db import init_models
from orion.core.sql_router.worker import start_worker_thread
from orion.core.bus.telemetry import start_telemetry_loop

from app.models import BiometricsSQL
from app.metrics import collect_biometrics
from app.settings import settings

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# ✅ 1. DEFINE THE CUSTOM WRITER FUNCTION
def biometrics_db_writer(model_class, raw_message: dict, db):
    """
    Custom writer that correctly parses the raw Redis message before creating a model instance.
    This function intercepts the raw message dictionary from the worker.
    """
    # Only process messages of type 'message'
    if raw_message.get('type') != 'message':
        return None

    try:
        # Extract, decode, and parse the actual data payload
        payload_bytes = raw_message['data']
        data = json.loads(payload_bytes)
        
        # Create the SQLAlchemy object with the clean data
        obj = model_class(**data)
        db.add(obj)
        db.commit()
        return obj
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse or process telemetry message: {e}", exc_info=True)
        db.rollback()
        return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "telemetry_publish_channel": settings.TELEMETRY_PUBLISH_CHANNEL,
        "external_subscribe_channel": settings.EXTERNAL_SUBSCRIBE_CHANNEL,
    }

@app.on_event("startup")
def startup():
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    init_models([BiometricsSQL])

    # 🛰️ THREAD 1: Publish this service's telemetry (GPU/CPU) to the bus.
    logger.info(f"Starting telemetry publisher to channel '{settings.TELEMETRY_PUBLISH_CHANNEL}'")
    threading.Thread(
        target=start_telemetry_loop,
        args=(settings.TELEMETRY_PUBLISH_CHANNEL, collect_biometrics, settings.ORION_BUS_URL),
        kwargs={"interval": settings.TELEMETRY_INTERVAL, "label": settings.SERVICE_NAME},
        daemon=True
    ).start()

    # 📥 THREAD 2: Start a worker to persist this service's OWN telemetry.
    logger.info(f"Starting DB writer for self-telemetry on channel '{settings.TELEMETRY_PUBLISH_CHANNEL}'")
    start_worker_thread(
        bus_url=settings.ORION_BUS_URL,
        channel=settings.TELEMETRY_PUBLISH_CHANNEL,
        model_class=BiometricsSQL,
        writer_func=biometrics_db_writer  # ✅ 2. PASS THE CUSTOM FUNCTION HERE
    )

    # 📥 THREAD 3 (OPTIONAL): Start a worker for an EXTERNAL data channel.
    if settings.EXTERNAL_SUBSCRIBE_CHANNEL:
        logger.info(f"Starting DB writer for external data on channel '{settings.EXTERNAL_SUBSCRIBE_CHANNEL}'")
        start_worker_thread(
            bus_url=settings.ORION_BUS_URL,
            channel=settings.EXTERNAL_SUBSCRIBE_CHANNEL,
            model_class=BiometricsSQL,
            writer_func=biometrics_db_writer  # ✅ 3. PASS IT HERE, TOO
        )
