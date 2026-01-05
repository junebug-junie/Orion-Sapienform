from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.codec import OrionCodec

from .conn_manager import manager
from .settings import settings
from .worker import handle_candidate, handle_trace, set_publisher_bus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orion-spark-introspector")


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=settings.heartbeat_interval_sec,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared publisher bus
    pub_bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled, codec=OrionCodec())
    await pub_bus.connect()

    # Pass bus to worker
    set_publisher_bus(pub_bus)

    async def multiplexer(env):
        if env.kind == "cognition.trace":
            await handle_trace(env)
        else:
            await handle_candidate(env)

    patterns = [settings.channel_spark_candidate, settings.channel_cognition_trace_pub]

    svc = Hunter(
        _cfg(),
        patterns=patterns,
        handler=multiplexer,
    )
    logger.info("Starting Spark Introspector Hunter patterns=%s", patterns)

    # Run Hunter in background
    # Hunter.start() runs a loop, so we wrap it in a task
    hunter_task = asyncio.create_task(svc.start())

    yield

    # Cleanup
    hunter_task.cancel()
    try:
        await hunter_task
    except asyncio.CancelledError:
        pass
    await pub_bus.close()


app = FastAPI(lifespan=lifespan)

# Mount static files
# We use relative path assuming CWD is services/orion-spark-introspector (where app module is)
# Wait, if run as `python -m app.main` from services/orion-spark-introspector/
# directory is "app/static".
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/ui")
async def get_ui():
    return FileResponse("app/static/index.html")


@app.websocket("/ws/tissue")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
