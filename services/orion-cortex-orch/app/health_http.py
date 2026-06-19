from __future__ import annotations

import asyncio
from typing import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1


def create_health_app(
    *,
    redis_getter: Callable[[], object | None],
    intake_channel: str,
    service_name: str,
    service_version: str,
) -> FastAPI:
    app = FastAPI(title=f"{service_name}-health", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": service_name, "version": service_version}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        redis = redis_getter()
        if redis is None:
            body = BusConsumerReadinessV1(
                ok=False,
                http_alive=True,
                bus_consumer_ready=False,
                intake_channel=intake_channel,
                subscriber_count=0,
                dependency_status="unavailable",
                error="redis unavailable",
            )
            return JSONResponse(body.model_dump(mode="json"), status_code=503)

        result = await check_bus_consumer_readiness(
            redis,
            intake_channel=intake_channel,
            service_name=service_name,
            check_heartbeat=False,
        )
        body = bus_consumer_readiness_v1(result, http_alive=True)
        status_code = 200 if body.ok else 503
        return JSONResponse(body.model_dump(mode="json"), status_code=status_code)

    return app


async def start_health_server(*, app: FastAPI, host: str, port: int) -> asyncio.Task:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    return asyncio.create_task(server.serve(), name=f"{app.title}-http")
