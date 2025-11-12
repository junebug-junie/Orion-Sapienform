import threading
import logging
from fastapi import FastAPI

from orion.core.bus.telemetry import start_telemetry_loop
from app.metrics import collect_biometrics
from app.settings import settings

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "telemetry_publish_channel": settings.TELEMETRY_PUBLISH_CHANNEL,
        "bus_url": settings.ORION_BUS_URL,
        "node": settings.NODE_NAME,
    }

@app.on_event("startup")
def startup():
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} (node={settings.NODE_NAME})")
    logger.info(f"Publishing telemetry to '{settings.TELEMETRY_PUBLISH_CHANNEL}'")

    threading.Thread(
        target=start_telemetry_loop,
        args=(settings.TELEMETRY_PUBLISH_CHANNEL, collect_biometrics, settings.ORION_BUS_URL),
        kwargs={"interval": settings.TELEMETRY_INTERVAL, "label": settings.SERVICE_NAME},
        daemon=True
    ).start()
