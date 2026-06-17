from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import check_bus_consumer_readiness, redis_pubsub_numsub

from . import routes
from .bus_runtime import start_services
from .settings import settings

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    rabbit, hunter = await start_services(stop_event)
    app.state.rabbit = rabbit
    app.state.hunter = hunter
    app.state.stop_event = stop_event
    try:
        yield
    finally:
        stop_event.set()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.include_router(routes.router, prefix="/api")


@app.get("/health")
def health():
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/ready")
async def ready(request: Request):
    rabbit = getattr(request.app.state, "rabbit", None)
    hunter = getattr(request.app.state, "hunter", None)
    exec_channel = f"{settings.EXEC_REQUEST_PREFIX}:CollapseMirrorService"

    if rabbit is None or hunter is None:
        return JSONResponse(
            {
                "ok": False,
                "bus_consumer_ready": False,
                "error": "bus services not started",
                "intake_channel": settings.CHANNEL_COLLAPSE_INTAKE,
                "exec_channel": exec_channel,
            },
            status_code=503,
        )

    redis = getattr(getattr(hunter, "bus", None), "redis", None)
    if redis is None:
        return JSONResponse(
            {
                "ok": False,
                "bus_consumer_ready": False,
                "error": "redis unavailable",
                "intake_channel": settings.CHANNEL_COLLAPSE_INTAKE,
                "exec_channel": exec_channel,
            },
            status_code=503,
        )

    intake_result, exec_result = await asyncio.gather(
        check_bus_consumer_readiness(
            redis,
            intake_channel=settings.CHANNEL_COLLAPSE_INTAKE,
            service_name=settings.SERVICE_NAME,
            health_channel=settings.ORION_HEALTH_CHANNEL,
            heartbeat_ttl_sec=float(settings.HEARTBEAT_INTERVAL_SEC) * 3.0,
            check_heartbeat=False,
        ),
        check_bus_consumer_readiness(
            redis,
            intake_channel=exec_channel,
            service_name=settings.SERVICE_NAME,
            health_channel=settings.ORION_HEALTH_CHANNEL,
            heartbeat_ttl_sec=float(settings.HEARTBEAT_INTERVAL_SEC) * 3.0,
            check_heartbeat=False,
        ),
    )

    numsub = await redis_pubsub_numsub(
        redis,
        [
            settings.CHANNEL_COLLAPSE_INTAKE,
            exec_channel,
            settings.CHANNEL_COLLAPSE_SQL_WRITE,
        ],
    )

    ok = intake_result.ok and exec_result.ok
    body = {
        "ok": ok,
        "bus_consumer_ready": ok,
        "dependency_status": "available" if ok else "unavailable",
        "intake": intake_result.model_dump(mode="json"),
        "exec": exec_result.model_dump(mode="json"),
        "subscriber_count_by_channel": numsub,
        "error": intake_result.error or exec_result.error,
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/")
def read_root():
    return {"message": f"{settings.SERVICE_NAME} is alive"}
