# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import logging
from fastapi import FastAPI
from fastapi import HTTPException
from contextlib import asynccontextmanager

from orion.core.bus.bus_service_chassis import OrionBusAsync
from orion.core.bus.enforce import enforcer
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.dream import DreamTriggerPayload
from app.settings import settings
from app.dream_api import router as dream_router

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

# Dream readout routes
app.include_router(dream_router, prefix="/dreams")

@app.post("/dreams/run", summary="Manually run the dream cycle")
async def run_dream_endpoint(mode: str = "standard"):
    """
    Publishes `dream.trigger` on the bus; cortex-orch normalizes it to `dream_cycle`.
    """
    if not settings.ORION_BUS_ENABLED:
        return {"error": "Bus disabled"}

    trigger = DreamTriggerPayload(mode=mode)
    channel = settings.CHANNEL_DREAM_TRIGGER
    catalog_entry = enforcer.entry_for(channel)
    catalog_schema_id = catalog_entry.get("schema_id") if catalog_entry else None
    logger.info(
        "Dream trigger request received channel=%s kind=%s payload_schema=%s catalog_schema=%s catalog_present=%s mode=%s",
        channel,
        "dream.trigger",
        type(trigger).__name__,
        catalog_schema_id,
        bool(catalog_entry),
        mode,
    )
    logger.debug("Dream trigger payload=%s", trigger.model_dump(mode="json"))

    # cortex-orch consumes it and normalizes to dream_cycle
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    try:
        logger.info("Connecting dream trigger bus url=%s", settings.ORION_BUS_URL)
        await bus.connect()
        env = BaseEnvelope(
            kind="dream.trigger",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
            payload=trigger.model_dump(mode="json")
        )
        logger.info(
            "Publishing dream trigger channel=%s kind=%s payload_schema=%s correlation_id=%s",
            channel,
            env.kind,
            type(trigger).__name__,
            env.correlation_id,
        )
        await bus.publish(channel, env)
        return {"status": "triggered", "mode": mode}
    except Exception as exc:
        logger.exception(
            "Dream trigger publish failed channel=%s kind=%s payload_schema=%s catalog_schema=%s bus_url=%s",
            channel,
            "dream.trigger",
            type(trigger).__name__,
            catalog_schema_id,
            settings.ORION_BUS_URL,
        )
        hint = (
            "Verify Redis connectivity plus orion/bus/channels.yaml and "
            "orion/schemas/registry.py registrations for dream.trigger."
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "dream_trigger_publish_failed",
                "channel": channel,
                "kind": "dream.trigger",
                "schema_id": type(trigger).__name__,
                "catalog_schema_id": catalog_schema_id,
                "message": str(exc),
                "hint": hint,
            },
        ) from exc
    finally:
        await bus.close()
