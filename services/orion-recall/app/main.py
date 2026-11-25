from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import time

import requests
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import Rabbit
from orion.core.bus.bus_schemas import ServiceRef
from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1

from .http_models import (
    LivedKnowRequestBody,
    LivedKnowResponseBody,
    RecallCompareRequestBody,
    RecallCompareResponseBody,
    RecallRequestBody,
    RecallResponseBody,
)
from .recall_eval import run_recall_eval_case, run_recall_eval_suite
from .recall_v2 import run_recall_v2_shadow
from .service import chassis_cfg
from .settings import settings
from .storage.falkor_entity_relatedness import (
    fetch_bridging_turns,
    fetch_entity_mention_timeline,
    fetch_related_entities,
)
from . import worker as _worker
from .worker import handle_recall, process_recall, persist_decision_async, set_lived_memory, set_recall_pg_pool

from orion.core.contracts.recall import RecallDecisionV1, RecallQueryV1

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
    try:
        from orion.graph.backend_config import strip_graph_credentials

        if settings.RECALL_RDF_ENDPOINT_URL:
            logger.info(
                "recall_graph_backend_selected backend=sparql query_url=%s",
                strip_graph_credentials(settings.RECALL_RDF_ENDPOINT_URL),
            )
    except Exception:
        if settings.RECALL_RDF_ENDPOINT_URL:
            logger.info("recall_graph_backend_selected backend=sparql query_url=%s", settings.RECALL_RDF_ENDPOINT_URL)

    # Advertise that this process reads the assertion-core graph shapes, so the referent/
    # assertion projectors' readiness gate can open. Background thread; never blocks or raises.
    from orion.substrate.reader_capability import advertise_at_startup

    advertise_at_startup()
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
    lived_enabled = bool(getattr(settings, "RECALL_LIVED_MEMORY_ENABLED", False))
    if asyncpg is not None and settings.RECALL_PG_DSN and (
        bool(getattr(settings, "RECALL_ENABLE_CARDS", False)) or lived_enabled
    ):
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

    lived_stop = asyncio.Event()
    lived_task = None
    if pool is not None and lived_enabled:
        lived_task = _start_lived_memory(pool, rabbit.bus, lived_stop)

    yield

    lived_stop.set()
    if lived_task is not None:
        lived_task.cancel()
        try:
            await lived_task
        except (asyncio.CancelledError, Exception):
            pass
        set_lived_memory(None)

    if pool is not None:
        try:
            await pool.close()
        except Exception:
            pass
        set_recall_pg_pool(None)
    await rabbit.stop()


def _start_lived_memory(pool, bus, stop: "asyncio.Event"):
    """Lived-memory recall (issue #3533): the per-turn reader plus its Chroma index loop."""
    from orion.introspect.semantic_index import SearchConfig

    from .lived_memory import LivedMemoryConfig, LivedMemoryRecall, index_loop
    from .lived_rewrite import BusRewriter

    search = SearchConfig(
        chroma_url=settings.RECALL_LIVED_MEMORY_CHROMA_URL,
        embed_url=settings.RECALL_LIVED_MEMORY_EMBED_URL,
        collection=settings.RECALL_LIVED_MEMORY_COLLECTION,
        min_similarity=0.0,
    )
    if not search.enabled:
        logger.warning("lived_memory_disabled reason=missing_chroma_or_embed_url")
        return None
    source = ServiceRef(name="orion-recall", version=settings.SERVICE_VERSION, node=settings.NODE_NAME)
    rewriter = None
    if settings.RECALL_LIVED_MEMORY_REWRITE_ENABLED and bus is not None:
        rewriter = BusRewriter(
            bus,
            channel=settings.CHANNEL_LLM_INTAKE,
            route=settings.RECALL_LIVED_MEMORY_REWRITE_ROUTE,
            timeout_sec=settings.RECALL_LIVED_MEMORY_REWRITE_TIMEOUT_SEC,
            source=source,
        )
    set_lived_memory(
        LivedMemoryRecall(
            pool,
            LivedMemoryConfig(
                search=search,
                rerank_url=settings.RECALL_LIVED_MEMORY_RERANK_URL,
                min_fit=settings.RECALL_LIVED_MEMORY_MIN_FIT,
                max_lines=settings.RECALL_LIVED_MEMORY_MAX_LINES,
                timeout_sec=settings.RECALL_LIVED_MEMORY_TIMEOUT_SEC,
                min_query_tokens=settings.RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS,
                short_gate=settings.RECALL_LIVED_MEMORY_SHORT_GATE_ENABLED,
                rewrite_timeout_sec=settings.RECALL_LIVED_MEMORY_REWRITE_TIMEOUT_SEC,
                link_top_k=settings.RECALL_LIVED_MEMORY_LINK_TOP_K,
                recent_slots=settings.RECALL_LIVED_MEMORY_RECENT_SLOTS,
            ),
            rewriter=rewriter,
        )
    )
    logger.info(
        "lived_memory_enabled collection=%s min_fit=%s rewrite_route=%s link_top_k=%s recent_slots=%s",
        search.collection, settings.RECALL_LIVED_MEMORY_MIN_FIT,
        settings.RECALL_LIVED_MEMORY_REWRITE_ROUTE if rewriter else "off", settings.RECALL_LIVED_MEMORY_LINK_TOP_K,
        settings.RECALL_LIVED_MEMORY_RECENT_SLOTS,
    )
    return asyncio.create_task(
        index_loop(
            pool, search, bus=bus, source=source,
            interval_sec=settings.RECALL_LIVED_MEMORY_INDEX_INTERVAL_SEC,
            batch=settings.RECALL_LIVED_MEMORY_INDEX_BATCH, stop=stop,
        ),
        name="lived_memory_index",
    )


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
async def ready():
    rabbit = getattr(app.state, "rabbit", None)
    if rabbit is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.RECALL_BUS_INTAKE,
            subscriber_count=0,
            dependency_status="unavailable",
            error="rabbit not started",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(getattr(rabbit, "bus", None), "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.RECALL_BUS_INTAKE,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=settings.RECALL_BUS_INTAKE,
        service_name=settings.SERVICE_NAME,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        heartbeat_ttl_sec=float(settings.HEARTBEAT_INTERVAL_SEC) * 3.0,
        check_heartbeat=False,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


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
    await persist_decision_async(decision)

    debug: dict = {}
    if body.diagnostic:
        debug = {
            "corr_id": corr,
            "decision": decision.model_dump(mode="json"),
            "backend_counts": decision.backend_counts,
            "selected_ids": decision.selected_ids,
        }
    return {"bundle": bundle.model_dump(mode="json"), "debug": debug}


# A deliberate lookup has no chat reply waiting on it, so it may take longer than a turn's budget.
KNOW_TIMEOUT_SEC = 8.0
KNOW_GRACE_SEC = 2.0
KNOW_VERB = "dig.know"
KNOW_PROFILE = "lived_memory.know"
KNOW_PROFILE_ADHOC = "lived_memory.know.adhoc"  # no turn id: an operator or eval run, not Orion's read demand
_know_gate = asyncio.Semaphore(1)
_know_writes: set = set()


@app.post("/lived/know", response_model=LivedKnowResponseBody)
async def lived_know_endpoint(body: LivedKnowRequestBody):
    """``python -P -m orion.dig know <words>`` (issue #3535): the lived-memory retriever a chat turn
    uses, on Orion's own words. Logged to recall_telemetry (verb ``dig.know``) under the turn's id.
    ``selected_ids`` holds the strong hits only (read demand counts what cleared the bar); the weak
    ids are in ``lived_trace.weak_ids``. A call with no turn id is tagged ``lived_memory.know.adhoc``.

    Never answers "nothing" for a search that did not run: off, busy, timeout and error are statuses."""
    reader = _worker._lived_memory
    corr = str(body.turn_id or "").strip()
    profile = KNOW_PROFILE if corr else KNOW_PROFILE_ADHOC
    corr = corr or f"adhoc:{uuid4()}"
    if reader is None:
        return {"status": "off", "backend": "", "degraded": [], "hits": [], "corr_id": corr}
    if _know_gate.locked():
        # One lookup at a time: the vector host refuses a third request in flight, and a chat
        # turn's rerank must not be pushed into degraded mode by a loop of lookups.
        return {"status": "busy", "backend": "", "degraded": [], "hits": [], "corr_id": corr}
    started = time.perf_counter()
    async with _know_gate:
        try:
            result = await asyncio.wait_for(
                reader.know(body.words, max_hits=body.max_hits, timeout_sec=KNOW_TIMEOUT_SEC),
                timeout=KNOW_TIMEOUT_SEC + KNOW_GRACE_SEC,
            )
        except Exception as exc:  # noqa: BLE001 -- said as "could not search", never as "no memory"
            status = "timeout" if isinstance(exc, asyncio.TimeoutError) else "error"
            logger.warning("lived_know_failed status=%s error=%s", status, type(exc).__name__)
            from .lived_memory import KnowResult, LivedRecall

            result = KnowResult(strong=[], weak=[], status=status,
                                recall=LivedRecall(debug={"outer": status if status == "timeout"
                                                          else f"error:{type(exc).__name__}"}))
    hits = [
        {"strength": strength, "kind": s.rec.kind, "purpose": s.rec.purpose, "id": s.rec.doc_id,
         "date": s.rec.date, "text": s.rec.text, "fit": round(float(s.fit), 4)}
        for strength, group in (("strong", result.strong), ("weak", result.weak))
        for s in group
    ]
    debug = dict(result.recall.debug)
    debug["status"] = result.status
    debug["injected"] = [{"id": s.rec.doc_id} for s in result.strong]
    debug["weak_ids"] = [s.rec.doc_id for s in result.weak]
    decision = RecallDecisionV1(
        corr_id=corr,
        verb=KNOW_VERB,
        profile=profile,
        query=body.words[:1000],
        selected_ids=[s.rec.doc_id for s in result.strong],
        backend_counts={"lived_memory": len(result.strong), "lived_memory_weak": len(result.weak),
                        **({"lived_memory_unverified": result.unverified} if result.unverified else {})},
        latency_ms=int((time.perf_counter() - started) * 1000),
        recall_debug={"lived_memory": debug},
        timings_ms={f"lived_memory_{k}": int(v) for k, v in (debug.get("timings_ms") or {}).items()
                    if isinstance(v, (int, float))},
    )
    # The row is written off the reply path: a slow insert must not turn a finished lookup into
    # "could not run" on Orion's side.
    task = asyncio.ensure_future(persist_decision_async(decision))
    _know_writes.add(task)
    task.add_done_callback(_know_writes.discard)
    logger.info(
        "lived_know corr_id=%s status=%s strong=%s weak=%s unverified=%s backend=%s latency_ms=%s",
        corr, result.status, len(result.strong), len(result.weak), result.unverified,
        result.recall.backend or "none", decision.latency_ms,
    )
    return {
        "status": result.status,
        "backend": result.recall.backend or "",
        "degraded": list(result.recall.degraded_reasons),
        "hits": hits,
        "unverified": result.unverified,
        "corr_id": corr,
    }


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


@app.get("/debug/entity-graph/related")
async def entity_graph_related_endpoint(name: str, max_results: int = 10):
    results = await fetch_related_entities(name=name, max_results=max_results)
    return {"name": name, "related": results}


@app.get("/debug/entity-graph/bridge")
async def entity_graph_bridge_endpoint(a: str, b: str, max_results: int = 5):
    return await fetch_bridging_turns(entity_a=a, entity_b=b, max_results=max_results)


@app.get("/debug/entity-graph/timeline")
async def entity_graph_timeline_endpoint(name: str, max_results: int = 100):
    results = await fetch_entity_mention_timeline(name=name, max_results=max_results)
    return {"name": name, "mentions": results}


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
        "RECALL_VECTOR_EMBEDDING_URL": settings.RECALL_VECTOR_EMBEDDING_URL,
        "RECALL_VECTOR_TIMEOUT_SEC": settings.RECALL_VECTOR_TIMEOUT_SEC,
        "RECALL_VECTOR_MAX_ITEMS": settings.RECALL_VECTOR_MAX_ITEMS,
        "VECTOR_DB_HOST": settings.VECTOR_DB_HOST,
        "VECTOR_DB_PORT": settings.VECTOR_DB_PORT,
        "VECTOR_DB_COLLECTION": settings.VECTOR_DB_COLLECTION,
        "RECALL_DEFAULT_PROFILE": settings.RECALL_DEFAULT_PROFILE,
    }
