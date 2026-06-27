from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import random
from datetime import datetime, timezone
from uuid import uuid4
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.codec import OrionCodec
from orion.core.bus.queue_service_chassis import QueueRabbit
from orion.core.bus.work_queue import RedisStreamWorkQueue

from . import introspection_guard as ig
from .conn_manager import manager
from .queue_worker import get_spark_queue_status, handle_spark_introspection_job
from .settings import settings
from .worker import (
    _PRODUCER_BOOT_ID,
    close_spark_stream_wq,
    handle_candidate,
    handle_self_state,
    handle_semantic_upsert,
    handle_signal,
    handle_spark_meta_patch,
    handle_trace,
    set_publisher_bus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orion-spark-introspector")


def _debug_token_equal(provided: str, expected: str) -> bool:
    """Constant-time compare on fixed-length SHA-256 digests (avoids length leaks from compare_digest)."""
    pe = provided.encode("utf-8")
    ee = expected.encode("utf-8")
    return hmac.compare_digest(hashlib.sha256(pe).digest(), hashlib.sha256(ee).digest())


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
        elif env.kind == "vector.upsert.v1":
            await handle_semantic_upsert(env)
        elif env.kind == "spark.signal.v1":
            await handle_signal(env)
        elif env.kind == "substrate.self_state.v1":
            await handle_self_state(env)
        elif env.kind == "chat.history.spark_meta.patch.v1":
            await handle_spark_meta_patch(env)
        else:
            await handle_candidate(env)

    patterns = [
        settings.channel_spark_candidate,
        settings.channel_cognition_trace_pub,
        settings.channel_spark_signal,
        settings.channel_vector_semantic_upsert,
        settings.channel_substrate_self_state,
        settings.channel_chat_history_spark_meta_patch,
    ]

    svc = Hunter(
        _cfg(),
        patterns=patterns,
        handler=multiplexer,
    )
    logger.info(f"Starting Spark Introspector Hunter patterns={patterns}")

    # Run Hunter in background
    hunter_task = asyncio.create_task(svc.start())

    queue_task: asyncio.Task | None = None
    qr: QueueRabbit | None = None
    if settings.spark_introspection_queue_enabled:
        cons = (settings.spark_introspection_queue_consumer or "").strip()
        consumer = cons if cons else f"spark-introspector:{settings.node_name}:{_PRODUCER_BOOT_ID}"
        logger.info(
            "spark_queue_worker_enabled stream=%s group=%s consumer=%s",
            settings.spark_introspection_queue_stream,
            settings.spark_introspection_queue_group,
            consumer,
        )
        wq = RedisStreamWorkQueue(
            settings.spark_introspection_redis_url or settings.orion_bus_url,
            codec=OrionCodec(),
        )
        qr = QueueRabbit(
            _cfg(),
            stream=settings.spark_introspection_queue_stream,
            group=settings.spark_introspection_queue_group,
            consumer=consumer,
            handler=handle_spark_introspection_job,
            work_queue=wq,
            concurrent_handlers=True,
            max_inflight=max(1, int(settings.spark_introspection_queue_max_inflight)),
            read_count=max(1, int(settings.spark_introspection_queue_read_count)),
            block_ms=int(settings.spark_introspection_queue_block_ms),
            max_attempts=int(settings.spark_introspection_queue_max_attempts),
            dlq_stream=settings.spark_introspection_queue_dlq,
            reclaim_pending=bool(settings.spark_introspection_queue_reclaim_pending),
            reclaim_min_idle_ms=int(settings.spark_introspection_queue_reclaim_min_idle_ms),
            stale_policy=str(settings.spark_introspection_queue_stale_policy),
            heartbeat_enabled=False,
        )
        queue_task = asyncio.create_task(qr.start())
    else:
        logger.info("spark_queue_worker_disabled reason=queue_disabled")

    yield

    # Stop PubSub intake first so new candidates are not accepted while the queue worker drains.
    hunter_task.cancel()
    try:
        await hunter_task
    except asyncio.CancelledError:
        pass

    if qr is not None:
        await qr.stop()
    if queue_task is not None:
        try:
            await queue_task
        except asyncio.CancelledError:
            pass
    await close_spark_stream_wq()
    await ig.close_redis_client()

    await pub_bus.close()


app = FastAPI(lifespan=lifespan)

# Router for shared endpoints
router = APIRouter()

@router.get("/ui")
async def get_ui():
    return FileResponse("app/static/index.html")

@router.websocket("/ws/tissue")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/debug/spark/introspection-queue")
async def spark_introspection_queue_debug(
    x_spark_introspection_debug_token: str | None = Header(
        default=None,
        alias="X-Spark-Introspection-Debug-Token",
    ),
):
    if not settings.spark_introspection_queue_debug_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    expected = (settings.spark_introspection_queue_debug_token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="debug endpoint enabled but SPARK_INTROSPECTION_QUEUE_DEBUG_TOKEN is unset",
        )
    provided = (x_spark_introspection_debug_token or "").strip()
    if not _debug_token_equal(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return await get_spark_queue_status()


@router.post("/api/test-pulse")
async def trigger_test_pulse():
    """Forces a random update to the connected UI clients to verify WS connection."""
    
    # 1. Tissue Update (Graph)
    tissue_payload = {
        "type": "tissue.update",
        "telemetry_id": str(uuid4()),
        "correlation_id": f"TEST-{str(uuid4())[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "phi": random.random(),
            "novelty": random.random(),
            "valence": random.random(),
            "arousal": random.random()
        },
        "metadata": {"source": "manual_test_pulse", "trace_verb": "pulse"}
    }
    
    # 2. Introspection Update (Text)
    intro_payload = {
        "type": "introspection.update",
        "correlation_id": tissue_payload["correlation_id"],
        "timestamp": tissue_payload["timestamp"],
        "text": "This is a simulated introspection generated by the Test Signal. If you see this, the websocket pipe for thoughts is active."
    }

    # Broadcast both
    await manager.broadcast(tissue_payload)
    await asyncio.sleep(0.1) # Small delay to ensure order
    await manager.broadcast(intro_payload)
    
    return {"status": "broadcast_sent", "payloads": [tissue_payload, intro_payload]}

# Include router at root (handles stripped prefix /spark -> /)
app.include_router(router)
# Include router with prefix (handles unstripped /spark/...)
app.include_router(router, prefix="/spark")

# Mount static files at both locations
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/spark/static", StaticFiles(directory="app/static"), name="static_spark")

# Redirects
@app.get("/")
async def root():
    return RedirectResponse(url="/spark/ui")

@app.get("/spark")
async def spark_root():
    return RedirectResponse(url="/spark/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
