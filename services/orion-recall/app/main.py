from __future__ import annotations

from contextlib import asynccontextmanager
import logging

import requests
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import Rabbit

from .http_models import RecallCompareRequestBody, RecallCompareResponseBody, RecallRequestBody, RecallResponseBody
from .recall_eval import run_recall_eval_case, run_recall_eval_suite
from .recall_v2 import run_recall_v2_shadow
from .service import chassis_cfg
from .settings import settings
from .worker import handle_recall, process_recall, _persist_decision, set_recall_pg_pool

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
        concurrent_handlers=settings.RECALL_RABBIT_CONCURRENT_HANDLERS,
    )

    # avoid referencing 'rabbit' before assignment inside a lambda
    def _handler(env):
        return handle_recall(env, bus=rabbit.bus)

    rabbit.handler = _handler  # type: ignore[attr-defined]

    await rabbit.start_background()
    app.state.rabbit = rabbit
    _check_rdf_endpoint()

    pool = None
    try:
        import asyncpg  # type: ignore
    except Exception:
        asyncpg = None
    if asyncpg is not None and settings.RECALL_PG_DSN and bool(getattr(settings, "RECALL_ENABLE_CARDS", False)):
        try:
            pool = await asyncpg.create_pool(
                dsn=settings.RECALL_PG_DSN,
                min_size=1,
                max_size=4,
            )
            set_recall_pg_pool(pool)
            app.state.recall_pg_pool = pool
            logger.info("Recall Postgres pool ready for memory cards.")
        except Exception as exc:
            logger.warning("Recall Postgres pool not started: %s", exc)
            set_recall_pg_pool(None)
    else:
        set_recall_pg_pool(None)

    yield

    if pool is not None:
        try:
            await pool.close()
        except Exception:
            pass
        set_recall_pg_pool(None)
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


@app.post("/debug/recall/compare", response_model=RecallCompareResponseBody)
async def recall_compare_endpoint(body: RecallCompareRequestBody):
    q = RecallQueryV1(
        fragment=body.query_text,
        session_id=body.session_id,
        node_id=body.node_id,
        verb="http.recall.compare",
        intent=None,
        profile=body.profile or settings.RECALL_DEFAULT_PROFILE,
    )
    corr = str(uuid4())
    bundle_v1, decision_v1 = await process_recall(q, corr_id=corr, diagnostic=True)
    bundle_v2, debug_v2 = await run_recall_v2_shadow(q)
    compare = {
        "query": body.query_text,
        "selected_count_delta": len(bundle_v2.items) - len(bundle_v1.items),
        "v1_latency_ms": decision_v1.latency_ms,
        "v2_latency_ms": int(debug_v2.get("latency_ms") or 0),
        "v1_backend_counts": decision_v1.backend_counts,
        "v2_backend_counts": bundle_v2.stats.backend_counts,
        "v1_pressure_events": list((decision_v1.recall_debug or {}).get("pressure_events") or []),
    }
    return {
        "v1": {
            "bundle": bundle_v1.model_dump(mode="json"),
            "decision": decision_v1.model_dump(mode="json"),
        },
        "v2": {
            "bundle": bundle_v2.model_dump(mode="json"),
            "debug": debug_v2,
            "anchors": debug_v2.get("plan"),
            "filters": debug_v2.get("filters"),
            "candidates": debug_v2.get("ranked_cards"),
        },
        "compare": compare,
    }


@app.post("/debug/recall/eval-case")
async def recall_eval_case_endpoint(case: dict):
    return await run_recall_eval_case(case)


@app.get("/debug/recall/eval-suite")
async def recall_eval_suite_endpoint():
    return await run_recall_eval_suite()


@app.get("/debug/settings")
def debug_settings():
    return {
        "RECALL_ENABLE_RDF": settings.RECALL_ENABLE_RDF,
        "RECALL_RDF_ENDPOINT_URL": settings.RECALL_RDF_ENDPOINT_URL,
        "GRAPHDB_URL": settings.GRAPHDB_URL,
        "GRAPHDB_REPO": settings.GRAPHDB_REPO,
        "RECALL_PG_DSN": settings.RECALL_PG_DSN,
        "RECALL_ENABLE_VECTOR": settings.RECALL_ENABLE_VECTOR,
        "RECALL_VECTOR_BASE_URL": settings.RECALL_VECTOR_BASE_URL,
        "RECALL_VECTOR_COLLECTIONS": settings.RECALL_VECTOR_COLLECTIONS,
        "RECALL_VECTOR_EMBEDDING_URL": settings.RECALL_VECTOR_EMBEDDING_URL,
        "RECALL_VECTOR_TIMEOUT_SEC": settings.RECALL_VECTOR_TIMEOUT_SEC,
        "RECALL_VECTOR_MAX_ITEMS": settings.RECALL_VECTOR_MAX_ITEMS,
        "VECTOR_DB_HOST": settings.VECTOR_DB_HOST,
        "VECTOR_DB_PORT": settings.VECTOR_DB_PORT,
        "VECTOR_DB_COLLECTION": settings.VECTOR_DB_COLLECTION,
        "RECALL_DEFAULT_PROFILE": settings.RECALL_DEFAULT_PROFILE,
    }
