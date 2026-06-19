from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.schemas.telemetry.system_health import BusConsumerReadinessV1

from .service import MeshGuardianService
from .settings import settings

guardian = MeshGuardianService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await guardian.start()
    try:
        yield
    finally:
        await guardian.stop()


app = FastAPI(title="orion-mesh-guardian", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "enabled": settings.enabled,
        "auto_remediate": settings.auto_remediate,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    redis = getattr(guardian.bus, "redis", None)
    bus_ok = redis is not None and guardian.bus.enabled
    eq_ok = guardian.equilibrium_subscriber_alive() or not settings.enabled
    ok = bus_ok and eq_ok
    body = BusConsumerReadinessV1(
        ok=ok,
        http_alive=True,
        bus_consumer_ready=bus_ok,
        intake_channel=settings.channel_equilibrium_snapshot,
        subscriber_count=1 if eq_ok else 0,
        dependency_status="available" if ok else "unavailable",
        error=None if ok else "guardian dependencies not ready",
    )
    status_code = 200 if ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)
