from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import Rabbit

from .http_models import RecallRequestBody, RecallResponseBody
from .pipeline import run_recall_pipeline
from .types import PhiContext, RecallQuery
from .service import chassis_cfg, handle
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    rabbit = Rabbit(chassis_cfg(), request_channel=settings.RECALL_BUS_INTAKE, handler=handle)
    await rabbit.start_background(stop_event)
    yield
    stop_event.set()
    await rabbit.stop()


app = FastAPI(title="Orion Recall", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.post("/recall", response_model=RecallResponseBody)
async def recall_endpoint(body: RecallRequestBody):
    q = RecallQuery(
        text=body.query_text,
        max_items=body.max_items,
        time_window_days=body.time_window_days,
        mode=body.mode,
        tags=body.tags,
        phi=PhiContext(**(body.phi or {})) if body.phi else None,
        trace_id=body.trace_id,
    )
    result = run_recall_pipeline(q)
    return RecallResponseBody(
        fragments=[
            {
                "id": fr.id,
                "kind": fr.kind,
                "source": fr.source,
                "text": fr.text,
                "ts": fr.ts,
                "tags": fr.tags,
                "salience": fr.salience,
                "valence": fr.valence,
                "arousal": fr.arousal,
                "meta": fr.meta,
            }
            for fr in result.fragments
        ],
        debug=result.debug,
    )
