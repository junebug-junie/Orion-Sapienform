from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.schemas.gpu_pool import GPU_POOL_ACTUATE_REQUEST_CHANNEL

from . import actuator_bus, pool_fence
from .settings import settings

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


BUS_RETRY_DELAY_SEC = 15.0


async def _start_bus_with_retry() -> None:
    """Start the actuator (which also publishes the heartbeat) and keep retrying: a bus that is
    slow at boot must not leave the controller permanently deaf. A fresh chassis per attempt,
    since a failed start_background leaves its chassis marked started."""
    global actuator_chassis
    while True:
        chassis = build_actuator_chassis()
        try:
            await chassis.start_background()
            actuator_chassis = chassis
            logger.info(f"[HOST] gpu_pool_actuator_started actuator={settings.GPU_POOL_ACTUATOR_NAME}")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_start_failed error={exc!r} retry_in={BUS_RETRY_DELAY_SEC}s")
            try:
                await chassis.stop()
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(BUS_RETRY_DELAY_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global actuator_chassis
    bus_task = None
    try:
        interrupted = await asyncio.to_thread(pool_fence.recover_interrupted)
        if interrupted:
            logger.warning(f"[HOST] gpu_pool_action_interrupted action_id={interrupted['action_id']}")
    except Exception as exc:  # noqa: BLE001 -- the fence re-reads (and fails closed) per request
        logger.error(f"[HOST] gpu_pool_fence_recover_failed error={exc}")
    if settings.ORION_BUS_ENABLED:
        bus_task = asyncio.create_task(_start_bus_with_retry(), name="gpu-lane-bus-start")
    yield
    if bus_task is not None:
        bus_task.cancel()
    if actuator_chassis is not None:
        try:
            await actuator_chassis.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_stop_error error={exc}")


app = FastAPI(title="Orion GPU Lane Controller", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}
