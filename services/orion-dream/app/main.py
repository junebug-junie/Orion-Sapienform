# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.dream import DreamTriggerPayload
from app.settings import settings
from app.dream_api import app as dream_api
from app.dream_cycle import run_dream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dream-app")

def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=settings.SHUTDOWN_GRACE_SEC,
    )

async def handle_dream_trigger(env: BaseEnvelope):
    logger.info(f"Received dream trigger from {env.source}")
    try:
        # Validate payload
        # Note: In Hunter, we might receive different kinds.
        if env.kind != "dream.trigger":
            # If we subscribed to pattern, we might get other things, but here we expect dream.trigger
            # Or maybe we support multiple.
            logger.info(f"Ignored kind: {env.kind}")
            return

        payload = DreamTriggerPayload.model_validate(env.payload)

        # Trigger the legacy dream cycle (refactored to be async aware if needed)
        # Assuming run_dream is safe to call.
        # If run_dream expects arguments from payload, pass them.
        # Currently run_dream() takes no args in the original code, but we should upgrade it eventually.
        # For now, we just trigger it.
        logger.info(f"Triggering dream cycle with mode={payload.mode}")
        status = await run_dream()
        logger.info(f"Dream cycle finished: {status}")

    except Exception as e:
        logger.error(f"Error handling dream trigger: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌙 Orion Dream module starting up…")

    # Use Hunter to listen for triggers
    hunter = Hunter(
        chassis_cfg(),
        pattern=settings.CHANNEL_DREAM_TRIGGER,
        handler=handle_dream_trigger
    )
    await hunter.start_background()

    yield

    await hunter.stop()
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
    Triggers the dream cycle via the event bus.
    """
    if not settings.ORION_BUS_ENABLED:
        return {"error": "Bus disabled"}

    # We publish a message to ourselves (or whoever is listening, which is us)
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
