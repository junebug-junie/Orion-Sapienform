from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import Rabbit

from .http_models import RecallRequestBody, RecallResponseBody
from .service import chassis_cfg
from .settings import settings
from .worker import handle_recall, process_recall
from orion.core.contracts.recall import RecallQueryV1


@asynccontextmanager
async def lifespan(app: FastAPI):
    rabbit = Rabbit(
        chassis_cfg(),
        request_channel=settings.RECALL_BUS_INTAKE,
        handler=None,  # set below once rabbit exists
    )

    # avoid referencing 'rabbit' before assignment inside a lambda
    def _handler(env):
        return handle_recall(env, bus=rabbit.bus)

    rabbit.handler = _handler  # type: ignore[attr-defined]

    await rabbit.start_background()
    app.state.rabbit = rabbit

    yield

    await rabbit.stop()


app = FastAPI(title="Orion Recall", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    # liveness: process is up
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
    }


@app.get("/ready")
def ready():
    # readiness: minimally confirm the bus worker started
    rabbit = getattr(app.state, "rabbit", None)
    return {"ok": rabbit is not None}


@app.post("/recall", response_model=RecallResponseBody)
async def recall_endpoint(body: RecallRequestBody):
    q = RecallQueryV1(
        fragment=body.query_text,
        session_id=body.session_id,
        verb="http.reflect",
        intent=None,
        profile=settings.RECALL_DEFAULT_PROFILE,
    )
    bundle, _ = await process_recall(q, corr_id="http")
    return {"bundle": bundle}
