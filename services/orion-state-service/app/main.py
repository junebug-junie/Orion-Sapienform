from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from redis.asyncio import Redis

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, Rabbit
# ADDED: BaseEnvelope needed for manual replies
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.schemas.telemetry.spark_ack import SparkStateSnapshotAckV1
from orion.schemas.telemetry.biometrics import BiometricsSummaryV1, BiometricsInductionV1, BiometricsClusterV1
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply

from .settings import settings
from .store import StateStore
from .pg import fetch_recent_spark_telemetry_metadata

STORE: Optional[StateStore] = None
# ADDED: Global reference to Rabbit chassis for RPC replies
RABBIT: Optional[Rabbit] = None


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(settings.heartbeat_interval_sec),
        health_channel=settings.health_channel,
        error_channel=settings.error_channel,
        shutdown_timeout_sec=float(settings.shutdown_grace_sec),
    )


async def _hydrate_from_postgres(store: StateStore) -> int:
    try:
        metas = await fetch_recent_spark_telemetry_metadata(
            postgres_uri=settings.postgres_uri,
            table=settings.spark_telemetry_table,
            limit=int(settings.hydrate_limit),
        )
    except Exception as e:
        logger.warning(f"Hydration from Postgres failed: {e}")
        return 0

    snaps: list[SparkStateSnapshotV1] = []
    for meta in metas:
        raw = meta.get("spark_state_snapshot")
        if not isinstance(raw, dict):
            continue
        try:
            snap = SparkStateSnapshotV1.model_validate(raw)
            snaps.append(snap)
        except Exception:
            continue

    # metas are newest-first; ingest in reverse so seq ordering behaves
    snaps = list(reversed(snaps))
    return await store.hydrate_from_snapshots(snaps, note="hydrate:postgres")


async def _handle_snapshot(env: BaseEnvelope) -> None:
    global STORE, RABBIT
    if STORE is None:
        return

    if env.kind != "spark.state.snapshot.v1":
        return

    # Handle typed payload (if registry deserialized it) or raw dict
    payload_obj = env.payload
    if hasattr(payload_obj, "model_dump"):
        payload_obj = payload_obj.model_dump(mode="json")
    if not isinstance(payload_obj, dict):
        payload_obj = {}

    try:
        snap = SparkStateSnapshotV1.model_validate(payload_obj)
    except Exception as e:
        # schema violation -> drop (Hunter will already have log/metrics upstream)
        logger.warning(f"Invalid snapshot payload dropped: {e}")
        return

    accepted = await STORE.ingest_snapshot(snap, note="bus", write_cache=True)
    if accepted:
        logger.info(
            f"Ingested spark snapshot node={snap.source_node} "
            f"seq={snap.seq} ts={snap.snapshot_ts.isoformat()}"
        )

    # --- FIX: Send ACK to prevent RPC timeouts ---
    # If the sender provided a reply_to address, they are waiting for us.
    if env.reply_to and RABBIT:
        ack_payload = SparkStateSnapshotAckV1(
            ok=True,
            snapshot_seq=snap.seq,
            snapshot_ts=snap.snapshot_ts,
            source_service=settings.service_name,
            source_node=settings.node_name,
        )
        ack_env = BaseEnvelope(
            kind="spark.state.snapshot.ack.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            payload=ack_payload.model_dump(mode="json"),
        )
        # Fire and forget the reply
        await RABBIT.send_reply(ack_env, env.reply_to)
        if settings.emit_legacy_snapshot_ack:
            legacy_env = BaseEnvelope(
                kind="spark.state.snapshot.ack",
                source=_source(),
                correlation_id=env.correlation_id,
                payload=ack_payload.model_dump(mode="json"),
            )
            await RABBIT.send_reply(legacy_env, env.reply_to)


async def _handle_get_latest(env: BaseEnvelope) -> BaseEnvelope:
    global STORE
    if STORE is None:
        out = StateLatestReply(ok=True, status="missing", note="store_not_ready")
        return BaseEnvelope(
            kind="state.latest.reply.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=out.model_dump(mode="json"),
        )

    # --- FIX: Accept the correct schema kind ---
    # We now strictly look for "state.get_latest.v1" (plus legacy fallback)
    valid_kinds = ("state.get_latest.v1", "orion.state.request", "StateGetLatestRequest", "legacy.message")
    
    if env.kind not in valid_kinds:
        out = StateLatestReply(ok=False, status="missing", note=f"unsupported_kind:{env.kind}")
        return BaseEnvelope(
            kind="state.latest.reply.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=out.model_dump(mode="json"),
        )

    # Handle typed payload (if registry deserialized it) or raw dict
    payload_obj = env.payload
    if hasattr(payload_obj, "model_dump"):
        payload_obj = payload_obj.model_dump(mode="json")
    if not isinstance(payload_obj, dict):
        payload_obj = {}

    if env.kind == "legacy.message" and isinstance(payload_obj.get("payload"), dict):
        payload_obj = payload_obj.get("payload") or {}

    try:
        req = StateGetLatestRequest.model_validate(payload_obj)
    except Exception as e:
        out = StateLatestReply(ok=False, status="missing", note=f"validation_failed:{e}")
        return BaseEnvelope(
            kind="state.latest.reply.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=out.model_dump(mode="json"),
        )

    reply = await STORE.get_latest(req, biometrics_stale_after_sec=float(settings.biometrics_stale_after_sec))
    return BaseEnvelope(
        kind="state.latest.reply.v1",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=reply.model_dump(mode="json"),
    )


async def _handle_biometrics(env: BaseEnvelope) -> None:
    global STORE
    if STORE is None:
        return

    payload_obj = env.payload
    if hasattr(payload_obj, "model_dump"):
        payload_obj = payload_obj.model_dump(mode="json")
    if not isinstance(payload_obj, dict):
        payload_obj = {}

    try:
        if env.kind == "biometrics.summary.v1":
            summary = BiometricsSummaryV1.model_validate(payload_obj)
            await STORE.ingest_biometrics_summary(summary)
        elif env.kind == "biometrics.induction.v1":
            induction = BiometricsInductionV1.model_validate(payload_obj)
            await STORE.ingest_biometrics_induction(induction)
        elif env.kind == "biometrics.cluster.v1":
            cluster = BiometricsClusterV1.model_validate(payload_obj)
            await STORE.ingest_biometrics_cluster(cluster)
    except Exception as exc:
        logger.warning("Invalid biometrics payload dropped kind=%s error=%s", env.kind, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STORE, RABBIT

    # Cache redis (shared with bus, but dedicated client)
    redis = Redis.from_url(settings.cache_redis_url, decode_responses=False)
    STORE = StateStore(redis=redis, key_prefix=settings.state_cache_prefix, primary_node=settings.primary_state_node)

    # 1) Best-effort warm start from Redis
    loaded = await STORE.load_global_from_cache()
    if loaded:
        logger.info("Loaded global state from Redis cache")
    else:
        logger.info("No Redis cache state found")

    # 2) Durable warm start from Postgres
    hydrated = await _hydrate_from_postgres(STORE)
    logger.info(f"Hydrated {hydrated} snapshots from Postgres")
    logger.info(
        "Legacy spark.state.snapshot.ack emission %s",
        "enabled" if settings.emit_legacy_snapshot_ack else "disabled",
    )

    # 3) Start chassis workers
    hunter = Hunter(_cfg(), patterns=[settings.channel_spark_state_snapshot], handler=_handle_snapshot)
    biometrics_hunter = Hunter(
        _cfg(),
        patterns=[
            settings.channel_biometrics_summary,
            settings.channel_biometrics_induction,
            settings.channel_biometrics_cluster,
        ],
        handler=_handle_biometrics,
    )
    rabbit = Rabbit(_cfg(), request_channel=settings.state_request_channel, handler=_handle_get_latest)

    # --- FIX: Set global reference so _handle_snapshot can reply ---
    RABBIT = rabbit

    stop = asyncio.Event()
    hunter_task = asyncio.create_task(hunter.start_background(stop))
    biometrics_task = asyncio.create_task(biometrics_hunter.start_background(stop))
    rabbit_task = asyncio.create_task(rabbit.start_background(stop))

    try:
        yield
    finally:
        stop.set()
        for t in (hunter_task, biometrics_task, rabbit_task):
            try:
                await asyncio.wait_for(t, timeout=float(settings.shutdown_grace_sec) + 1.0)
            except Exception:
                pass
        try:
            await redis.close()
        except Exception:
            pass


app = FastAPI(title="orion-state-service", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": settings.service_name, "node": settings.node_name, "version": settings.service_version}


@app.get("/state/latest")
async def http_get_latest(
    scope: str = Query("global", pattern="^(global|node)$"),
    node: Optional[str] = Query(None),
) -> JSONResponse:
    global STORE
    if STORE is None:
        raise HTTPException(status_code=503, detail="state_store_not_ready")
    req = StateGetLatestRequest(scope=scope, node=node)
    reply = await STORE.get_latest(req)
    return JSONResponse(reply.model_dump(mode="json"))


@app.get("/state/debug")
async def http_debug() -> Dict[str, Any]:
    global STORE
    if STORE is None:
        return {"ok": False, "note": "state_store_not_ready"}
    return await STORE.debug_state()
