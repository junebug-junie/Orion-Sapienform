from __future__ import annotations

from contextlib import asynccontextmanager
import logging

import requests
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import Rabbit

from .http_models import RecallRequestBody, RecallResponseBody
from .service import chassis_cfg
from .settings import settings
from .worker import handle_recall, process_recall, _persist_decision

from orion.core.contracts.recall import RecallQueryV1

logger = logging.getLogger("orion-recall.main")


def _check_rdf_endpoint() -> None:
    if not settings.RECALL_RDF_ENDPOINT_URL:
        logger.info("RDF endpoint not configured; skipping RDF health check.")
        return
    try:
        resp = requests.get(
            settings.RECALL_RDF_ENDPOINT_URL,
            auth=(settings.RECALL_RDF_USER, settings.RECALL_RDF_PASS),
            timeout=settings.RECALL_RDF_TIMEOUT_SEC,
        )
        logger.info("RDF endpoint check status=%s url=%s", resp.status_code, settings.RECALL_RDF_ENDPOINT_URL)
    except Exception as exc:
        logger.warning("RDF endpoint check failed url=%s error=%s", settings.RECALL_RDF_ENDPOINT_URL, exc)


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
    _check_rdf_endpoint()

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
        profile=body.profile or settings.RECALL_DEFAULT_PROFILE,
    )
    corr = str(uuid4())
    bundle, decision = await process_recall(q, corr_id=corr, diagnostic=bool(body.diagnostic))

    # Best-effort persist to Postgres (creates recall_telemetry table in the same DB)
    _persist_decision(decision)

    debug: dict = {}
    if body.diagnostic:
        debug = {
            "corr_id": corr,
            "decision": decision.model_dump(mode="json"),
            "backend_counts": decision.backend_counts,
            "selected_ids": decision.selected_ids,
        }
    return {"bundle": bundle.model_dump(mode="json"), "debug": debug}


@app.get("/debug/settings")
def debug_settings():
    return {
        "RECALL_ENABLE_RDF": settings.RECALL_ENABLE_RDF,
        "RECALL_RDF_ENDPOINT_URL": settings.RECALL_RDF_ENDPOINT_URL,
        "GRAPHDB_URL": settings.GRAPHDB_URL,
        "GRAPHDB_REPO": settings.GRAPHDB_REPO,
        "RECALL_PG_DSN": settings.RECALL_PG_DSN,
        "RECALL_ENABLE_VECTOR": settings.RECALL_ENABLE_VECTOR,
        "RECALL_DEFAULT_PROFILE": settings.RECALL_DEFAULT_PROFILE,
    }
