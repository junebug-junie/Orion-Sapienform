from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter
from orion.schemas.gpu_pool import GPU_POOL_ACTUATE_REQUEST_CHANNEL

from . import actuator_bus, pool_fence
from .settings import settings

heartbeat_chassis: HeartbeatOnly | None = None
actuator_chassis: Hunter | None = None


def _chassis_config() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
    )


def build_heartbeat_chassis() -> HeartbeatOnly:
    return HeartbeatOnly(_chassis_config())


def build_actuator_chassis() -> Hunter:
    """Stage 4.2: pool actuation intake. Concurrent handlers so a `status` or a second request is
    answered (busy) while a 15-minute load runs; admission itself is serialized in actuator_bus."""
    holder: dict = {}

    async def on_request(env):
        await actuator_bus.handle(env.payload, holder["publish"], env.correlation_id)

    hunter = Hunter(_chassis_config(), handler=on_request, patterns=[GPU_POOL_ACTUATE_REQUEST_CHANNEL],
                    concurrent_handlers=True)
    holder["publish"] = actuator_bus.bus_publisher(hunter.bus)
    return hunter


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis, actuator_chassis
    try:
        interrupted = await asyncio.to_thread(pool_fence.recover_interrupted)
        if interrupted:
            logger.warning(f"[HOST] gpu_pool_action_interrupted action_id={interrupted['action_id']}")
    except Exception as exc:  # noqa: BLE001 -- the fence re-reads (and fails closed) per request
        logger.error(f"[HOST] gpu_pool_fence_recover_failed error={exc}")
    if settings.ORION_BUS_ENABLED:
        try:
            actuator_chassis = build_actuator_chassis()
            await actuator_chassis.start_background()
            logger.info(f"[HOST] gpu_pool_actuator_started actuator={settings.GPU_POOL_ACTUATOR_NAME}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_start_failed error={exc}")
            actuator_chassis = None
    if actuator_chassis is None:
        # The actuator Hunter already publishes the heartbeat (BaseChassis); a second chassis would
        # be a second SystemHealthV1 stream + bus connection. HeartbeatOnly is only the fallback.
        try:
            heartbeat_chassis = build_heartbeat_chassis()
            await heartbeat_chassis.start_background()
            logger.info(f"[HOST] system_health_heartbeat_started service={settings.SERVICE_NAME}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] system_health_heartbeat_start_failed error={exc}")
            heartbeat_chassis = None
    yield
    if actuator_chassis is not None:
        try:
            await actuator_chassis.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_stop_error error={exc}")
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] system_health_heartbeat_stop_error error={exc}")


app = FastAPI(title="Orion GPU Lane Controller", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}
