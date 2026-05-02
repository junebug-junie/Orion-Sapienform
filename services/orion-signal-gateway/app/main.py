"""orion-signal-gateway FastAPI entrypoint."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI

from .instrumentation import configure_tracing
from .service import GatewayService
from .settings import settings

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(settings.SERVICE_NAME)

_gateway: GatewayService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gateway
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    configure_tracing(
        service_name=settings.SERVICE_NAME,
        otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        console_export=settings.OTEL_CONSOLE_EXPORT,
    )
    _gateway = GatewayService()
    await _gateway.start()
    try:
        yield
    finally:
        if _gateway:
            await _gateway.stop()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
        "bus_url": settings.ORION_BUS_URL,
    }


@app.get("/signals/active")
def signals_active() -> Dict[str, Any]:
    """Return most recent OrionSignalV1 per organ_id from the in-memory window."""
    if _gateway is None:
        return {"as_of": None, "signals": {}}
    window = _gateway.get_signal_window()
    signals = {k: v.model_dump(mode="json") for k, v in window.get_all().items()}
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
