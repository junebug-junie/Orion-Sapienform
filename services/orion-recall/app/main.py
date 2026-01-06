from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import Rabbit

from .http_models import RecallRequestBody, RecallResponseBody
from .service import chassis_cfg
from .settings import settings
from .worker import handle_recall
from orion.core.contracts.recall import RecallQueryV1
from .worker import process_recall


@asynccontextmanager
async def lifespan(app: FastAPI):
    rabbit = Rabbit(
        chassis_cfg(),
        request_channel=settings.RECALL_BUS_INTAKE,
        handler=lambda env: handle_recall(env, bus=rabbit.bus),
    )
    await rabbit.start_background()

    yield

    await rabbit.stop()


app = FastAPI(title="Orion Recall", version=settings.SERVICE_VERSION, lifespan=lifespan)


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
