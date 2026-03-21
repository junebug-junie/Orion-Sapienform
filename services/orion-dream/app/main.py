# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from orion.core.bus.bus_service_chassis import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.dream import DreamTriggerPayload
from app.settings import settings
from app.dream_api import app as dream_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dream-app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Dream execution is owned by cortex-orch (dream.trigger -> dream_cycle).
    This service provides HTTP readout and publishes compatibility triggers only.
    """
    logger.info("🌙 Orion Dream module starting up (readout façade; triggers go to cortex-orch)…")

    yield

    logger.info("💤 Orion Dream module shutting down…")

app = FastAPI(
    title="Orion Dream Module",
    description="Generates nightly dreams from Orion’s stored memories.",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

# Mount dream_api routes
app.mount("/dreams", dream_api)

@app.post("/dreams/run", summary="Manually run the dream cycle")
async def run_dream_endpoint(mode: str = "standard"):
    """
    Publishes `dream.trigger` on the bus; cortex-orch normalizes it to `dream_cycle`.
    """
    if not settings.ORION_BUS_ENABLED:
        return {"error": "Bus disabled"}

    # cortex-orch consumes it and normalizes to dream_cycle
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()

    try:
        trigger = DreamTriggerPayload(mode=mode)
        env = BaseEnvelope(
            kind="dream.trigger",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
            payload=trigger.model_dump(mode="json")
        )
        await bus.publish(settings.CHANNEL_DREAM_TRIGGER, env)
        return {"status": "triggered", "mode": mode}
    finally:
        await bus.close()
