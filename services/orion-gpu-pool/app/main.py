"""orion-gpu-pool: the one lease queue for every GPU on circe.

Bus first (spec, "Transport and telemetry"): leases, state and operator control are bus RPC;
HTTP is only /health and a read-only /v1/pool debug mirror. Only the Postgres advisory-lock
holder subscribes to anything, so a second replica can never double-grant.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter, Rabbit
from orion.core.bus.rpc_health_publish import RpcHealthPublisher
from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.discovery import Probe, load_profiles
from orion.gpu_pool.lease_graph import build_lease_graph
from orion.schemas.gpu_pool import (
    GPU_LEASE_REPLY_KIND, GPU_POOL_CONTROL_REPLY_KIND, GPU_POOL_CONTROL_REQUEST_CHANNEL,
    GPU_POOL_LEASE_REQUEST_CHANNEL, GPU_POOL_STATE_KIND, GPU_POOL_STATE_REQUEST_CHANNEL,
    LLM_WORKER_ANNOUNCE_CHANNEL, GpuLeaseReplyV1, GpuLeaseRequestV1, GpuPoolControlReplyV1,
    GpuPoolControlV1, GpuPoolStateRequestV1, LlmWorkerAnnounceV1,
)

from app.runtime import PoolRuntime
from app.settings import get_settings
from app.store import PostgresStore

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-gpu-pool.main")

runtime: PoolRuntime | None = None
_bus: OrionBusAsync | None = None
_tasks: list[asyncio.Task] = []
_chassis: list[Any] = []
_publisher: RpcHealthPublisher | None = None
_pool: Any = None
_store: PostgresStore | None = None
_stop = asyncio.Event()


def _cfg() -> ChassisConfig:
    s = _settings
    return ChassisConfig(service_name=s.service_name, service_version=s.service_version, node_name=s.node_name,
                         bus_url=s.orion_bus_url, bus_enabled=s.orion_bus_enabled,
                         heartbeat_interval_sec=s.heartbeat_interval_sec)


def _source() -> ServiceRef:
    return ServiceRef(name=_settings.service_name, version=_settings.service_version, node=_settings.node_name)


def _reply(env: BaseEnvelope, kind: str, payload: Any) -> BaseEnvelope:
    return BaseEnvelope(kind=kind, source=_source(), correlation_id=env.correlation_id,
                        payload=payload.model_dump(mode="json"))


async def _probe(client: httpx.AsyncClient, role: str, url: str, kind: str, health: str) -> Probe:
    now = datetime.now(timezone.utc)
    try:
        if kind == "llm":
            r = await client.get(f"{url}/props")
            r.raise_for_status()
            return Probe(True, r.json(), checked_at=now)
        r = await client.get(f"{url}{health}")
        return Probe(r.status_code < 500, error=None if r.status_code < 500 else f"HTTP {r.status_code}",
                     checked_at=now)
    except Exception as exc:  # noqa: BLE001
        return Probe(False, error=f"{type(exc).__name__}: {exc}"[:300], checked_at=now)


async def _on_lease(env: BaseEnvelope) -> BaseEnvelope | None:
    try:
        req = GpuLeaseRequestV1.model_validate(env.payload or {})
    except Exception as exc:  # noqa: BLE001
        return _reply(env, GPU_LEASE_REPLY_KIND, GpuLeaseReplyV1(status="unavailable", reason=f"invalid:{exc}"[:300]))
    if req.verb == "acquire":
        out = await runtime.acquire(req)
    elif not req.lease_id:
        out = GpuLeaseReplyV1(status="unknown_lease", reason="lease_id required")
    elif req.verb == "heartbeat":
        out = await runtime.heartbeat(req.lease_id)
    elif req.verb == "release":
        out = await runtime.release(req.lease_id, req.outcome or "ok", req.detail)
    else:
        out = await runtime.cancel(req.lease_id)
    return _reply(env, GPU_LEASE_REPLY_KIND, out)


async def _on_state(env: BaseEnvelope) -> BaseEnvelope | None:
    req = GpuPoolStateRequestV1.model_validate(env.payload or {})
    return _reply(env, GPU_POOL_STATE_KIND, await runtime.snapshot(include_leases=req.include_leases))


async def _on_control(env: BaseEnvelope) -> BaseEnvelope | None:
    try:
        ctl = GpuPoolControlV1.model_validate(env.payload or {})
    except Exception as exc:  # noqa: BLE001
        return _reply(env, GPU_POOL_CONTROL_REPLY_KIND, GpuPoolControlReplyV1(ok=False, reason=f"invalid:{exc}"[:300]))
    out = await runtime.control(ctl)
    logger.info("gpu_pool_control verb=%s actor=%s ok=%s reason=%s", ctl.verb, ctl.actor, out.ok, out.reason)
    return _reply(env, GPU_POOL_CONTROL_REPLY_KIND, out)


async def _on_announce(env: BaseEnvelope) -> None:
    try:
        await runtime.on_announce(LlmWorkerAnnounceV1.model_validate(env.payload or {}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpu_pool_announce_invalid err=%s", exc)


async def _tick_forever() -> None:
    import os

    checked = 0.0
    while not _stop.is_set():
        now = asyncio.get_running_loop().time()
        if _store is not None and now - checked >= 10:
            checked = now
            if not await _store.leader_alive():
                # Another replica may now hold the lock; writing on would risk double grants.
                logger.critical("gpu_pool_leader_lock_lost -- exiting so the container restarts")
                os._exit(3)
        try:
            await runtime.tick()
        except Exception:  # noqa: BLE001 -- one bad tick must not stop the pool
            logger.exception("gpu_pool_tick_failed")
        try:
            await asyncio.wait_for(_stop.wait(), timeout=_settings.tick_sec)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime, _bus, _publisher, _pool, _store
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    cfg = load_pool_config(_settings.config_path)
    profiles = load_profiles(_settings.profiles_path)
    logger.info("gpu_pool_config digest=%s cards=%s roles=%s mode=%s", cfg.digest, sorted(cfg.cards),
                sorted(cfg.roles), _settings.mode)

    heartbeat = HeartbeatOnly(_cfg())
    await heartbeat.start_background()
    _chassis.append(heartbeat)

    # One pool for the checkpointer and the projection (see feedback: the saver needs a pool).
    _pool = AsyncConnectionPool(conninfo=_settings.postgres_uri, min_size=2, max_size=10, open=False,
                                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await _pool.open()
    saver = AsyncPostgresSaver(_pool)
    await saver.setup()
    _store = PostgresStore(_pool, conninfo=_settings.postgres_uri)
    await _store.check_schema()
    logger.info("gpu_pool_waiting_for_leader_lock")
    await _store.leader()
    logger.info("gpu_pool_leader_acquired")

    _bus = OrionBusAsync(url=_settings.orion_bus_url, enabled=_settings.orion_bus_enabled)
    await _bus.connect()
    _publisher = RpcHealthPublisher(
        enabled=_settings.rpc_health_publish_enabled, bus_getter=lambda: _bus, service=_settings.service_name,
        node=_settings.node_name, instance="main", source=_source(),
        interval_sec=_settings.rpc_health_publish_interval_sec, include_channel_latency=True)
    _publisher.start()

    client = httpx.AsyncClient(timeout=_settings.probe_timeout_sec)

    async def prober(role, url, kind, health):
        return await _probe(client, role, url, kind, health)

    runtime = PoolRuntime(
        cfg=cfg, profiles=profiles, store=_store, graph=build_lease_graph(lambda: cfg, saver), bus=_bus,
        prober=prober, mode=_settings.mode, operator_token=_settings.operator_token,
        service_name=_settings.service_name, announce_stale_sec=_settings.announce_stale_sec,
        probe_interval_sec=_settings.probe_interval_sec, state_publish_sec=_settings.state_publish_sec,
        replay_payload_max_bytes=_settings.replay_payload_max_bytes)
    await runtime.start()

    for channel, handler in ((GPU_POOL_LEASE_REQUEST_CHANNEL, _on_lease),
                             (GPU_POOL_STATE_REQUEST_CHANNEL, _on_state),
                             (GPU_POOL_CONTROL_REQUEST_CHANNEL, _on_control)):
        rabbit = Rabbit(_cfg(), request_channel=channel, handler=handler, concurrent_handlers=True)
        await rabbit.start_background()
        _chassis.append(rabbit)
    hunter = Hunter(_cfg(), handler=_on_announce, patterns=[LLM_WORKER_ANNOUNCE_CHANNEL])
    await hunter.start_background()
    _chassis.append(hunter)
    _stop.clear()
    _tasks.append(asyncio.create_task(_tick_forever()))
    logger.info("gpu_pool_ready channels=%s", [GPU_POOL_LEASE_REQUEST_CHANNEL, GPU_POOL_STATE_REQUEST_CHANNEL,
                                              GPU_POOL_CONTROL_REQUEST_CHANNEL, LLM_WORKER_ANNOUNCE_CHANNEL])
    try:
        yield
    finally:
        _stop.set()
        for task in _tasks:
            task.cancel()
        await asyncio.gather(*_tasks, return_exceptions=True)
        for c in _chassis:
            try:
                await c.stop()
            except Exception:  # noqa: BLE001
                pass
        if _publisher is not None:
            await _publisher.stop()
        await client.aclose()
        if _bus is not None:
            await _bus.close()
        if _store is not None:
            await _store.close()
        if _pool is not None:
            await _pool.close()


app = FastAPI(title="orion-gpu-pool", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": runtime is not None, "service": _settings.service_name, "mode": _settings.mode,
            "config_digest": runtime.cfg.digest if runtime else None}


@app.get("/v1/pool")
async def pool_state() -> dict[str, Any]:
    """Read-only debug mirror of orion:gpu_pool:state, for curl from a shell."""
    if runtime is None:
        raise HTTPException(503, "starting")
    return (await runtime.snapshot()).model_dump(mode="json")


@app.get("/v1/leases/{lease_id}/history")
async def lease_history(lease_id: str) -> dict[str, Any]:
    if runtime is None:
        raise HTTPException(503, "starting")
    return {"lease_id": lease_id, "history": await runtime.history(lease_id)}
