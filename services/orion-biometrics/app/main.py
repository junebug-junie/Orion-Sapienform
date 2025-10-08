# orion_biometrics/main.py
import threading

from fastapi import FastAPI

from orion.core.sql_router.db import init_models
from orion.core.sql_router.worker import start_worker_thread
from orion.core.sql_router.router import build_writer_router

from app.models import BiometricsSQL
from orion.core.bus.telemetry import start_telemetry_loop
from app.metrics import collect_biometrics
from app.settings import settings

app = FastAPI(title=settings.SERVICE_NAME)

# optional API write route
app.include_router(
    build_writer_router(BiometricsSQL, db_writer_func=lambda cls, d, db: cls(**d)),
    prefix="/api"
)

@app.get("/collect")
def collect():
    return collect_biometrics()

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "subscribe_channel": settings.SUBSCRIBE_CHANNEL,
        "publish_channel": settings.PUBLISH_CHANNEL,
    }

@app.on_event("startup")
def startup():
    init_models([BiometricsSQL])

    # 🛰️ Publish telemetry (GPU/CPU) every 30s
    if settings.PUBLISH_CHANNEL:
        threading.Thread(
            target=start_telemetry_loop,
            args=(settings.PUBLISH_CHANNEL, collect_biometrics, settings.ORION_BUS_URL),
            kwargs={"interval": 30, "label": "orion-biometrics"},
            daemon=True
        ).start()

    # 📥 Worker for external metrics (e.g. piped-in crash dumps)
    if settings.SUBSCRIBE_CHANNEL:
        start_worker_thread(
            bus_url=settings.ORION_BUS_URL,
            channel=settings.SUBSCRIBE_CHANNEL,
            model_class=BiometricsSQL
        )

    # 📥 Worker to persist *our own* telemetry channel
    if settings.PUBLISH_CHANNEL:
        start_worker_thread(
            bus_url=settings.ORION_BUS_URL,
            channel=settings.PUBLISH_CHANNEL,
            model_class=BiometricsSQL
        )
