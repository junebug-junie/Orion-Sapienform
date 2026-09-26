from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter
from orion.core.bus.rpc_health_publish import RpcHealthPublisher
from orion.schemas.durable_run import DURABLE_RUN_REQUEST_KIND, DURABLE_RUN_RECEIPT_KIND, DurableRunRequestV1, DurableRunReceiptV1
from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, ResourceEventV1
from orion.schemas.resource_admission import (
    CapacityAcquireV1, CapacityTokenV1, CapacityAcquireResultV1,
    CapacityRenewResultV1, CapacityReleaseResultV1,
)

from app.settings import get_settings

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-durable-runs.main")

runner: Any = None
rpc_bus: OrionBusAsync | None = None
hunter: Hunter | None = None
heartbeat: HeartbeatOnly | None = None
_stop = asyncio.Event()
_sweep_task: asyncio.Task[None] | None = None
_checkpointer_cm: Any = None
admission: Any = None
_admission_task: asyncio.Task | None = None
capacity: Any = None
rpc_health_publisher: RpcHealthPublisher | None = None


def _rpc_bus_getter() -> OrionBusAsync | None:
    return rpc_bus


def build_rpc_health_publisher() -> RpcHealthPublisher:
    """Publishes the long-lived rpc_bus's RPC-health window: the runner's rpc_request
    calls (harness turn, verb dispatch) and the GPU pool lease RPCs (``gpu_pool_lease``).
    Stage 4.5 removed every outbound HTTP hop (gateway /routes, lane /slots, elastic
    controller, cabinet, thought). Single container -> instance="main"."""
    s = _settings
    return RpcHealthPublisher(
        enabled=s.rpc_health_publish_enabled and s.orion_bus_enabled,
        bus_getter=_rpc_bus_getter,
        service=s.service_name,
        node=s.node_name,
        instance="main",
        source=ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name),
        interval_sec=s.rpc_health_publish_interval_sec,
        include_channel_latency=s.rpc_health_channel_latency_enabled,
    )


def _chassis_cfg() -> ChassisConfig:
    s = _settings
    return ChassisConfig(
        service_name=s.service_name,
        service_version=s.service_version,
        node_name=s.node_name,
        bus_url=s.orion_bus_url,
        bus_enabled=s.orion_bus_enabled,
        heartbeat_interval_sec=s.heartbeat_interval_sec,
    )


async def _handle_request(env: BaseEnvelope) -> None:
    if env.kind == GPU_POOL_EVENT_KIND:
        # The pool's lease lifecycle: a waiting run wakes on "granted" for its own holder.
        if admission is not None and isinstance(env.payload, dict):
            await admission.on_pool_event(env.payload)
        return
    if env.kind == RESOURCE_EVENT_KIND:
        if admission is not None:
            await admission.wakeup(ResourceEventV1.model_validate(env.payload))
        return
    if env.kind != DURABLE_RUN_REQUEST_KIND:
        logger.warning("durable_run_request_unexpected_kind kind=%s", env.kind)
        return
    try:
        request = DurableRunRequestV1.model_validate(env.payload or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("durable_run_request_invalid corr=%s err=%s", env.correlation_id, exc)
        return
    if runner is None:
        logger.warning("durable_run_request_dropped reason=runner_not_started run=%s", request.run_id)
        return
    logger.info("durable_run_request run=%s workflow=%s corr=%s", request.run_id, request.workflow, request.correlation_id)
    if request.admission is not None:
        if admission is None:
            logger.error("durable_admission_disabled run=%s", request.run_id)
            return
        receipt = DurableRunReceiptV1(**await admission.submit(request))
        if env.reply_to and rpc_bus is not None:
            await rpc_bus.publish(env.reply_to, BaseEnvelope(kind=DURABLE_RUN_RECEIPT_KIND,
                source=ServiceRef(name=_settings.service_name, version=_settings.service_version, node=_settings.node_name),
                correlation_id=env.correlation_id, payload=receipt.model_dump(mode="json")))
        return
    await runner.start_run(request)


async def _open_checkpointer():
    """LangGraph's Postgres saver on a CONNECTION POOL, tables created on
    first boot.

    Live incident 2026-09-06 20:52Z: `from_conn_string` gives the saver ONE
    psycopg async connection. A node that awaits for an hour (the harness
    turn RPC) and a concurrent `aget_state` (a second run's kickoff, the
    failure handler) contend for that single connection and the runner
    deadlocked -- no state rows, second request never seeded, health still
    listing the first run as active. Concurrency here is the norm, not the
    exception, so the saver gets a pool (LangGraph's own recommendation).
    """
    global _checkpointer_cm
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    _checkpointer_cm = AsyncConnectionPool(
        conninfo=_settings.postgres_uri,
        min_size=1,
        max_size=8,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await _checkpointer_cm.open()
    saver = AsyncPostgresSaver(_checkpointer_cm)
    await saver.setup()
    return saver


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, rpc_bus, hunter, heartbeat, _sweep_task, admission, _admission_task, capacity
    global rpc_health_publisher
    from app.runner import DurableRunner

    try:
        heartbeat = HeartbeatOnly(_chassis_cfg())
        await heartbeat.start_background()
    except Exception as exc:  # noqa: BLE001
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat = None

    if _settings.enabled:
        saver = await _open_checkpointer()
        if _settings.orion_bus_enabled:
            rpc_bus = OrionBusAsync(url=_settings.orion_bus_url)
            await rpc_bus.connect()
            rpc_health_publisher = build_rpc_health_publisher()
            rpc_health_publisher.start()
        runner = DurableRunner(_settings, bus=rpc_bus, checkpointer=saver)
        if _settings.capacity_enabled:
            from orion.durable_admission.capacity import PostgresCapacityStore
            from orion.durable_admission.store import PostgresAdmissionStore
            # reserve_waiting=False: pending durable demands are frozen since stage 4.5 (the pool
            # places runs); no broker exists to honour a drain reservation for them any more.
            capacity = PostgresCapacityStore(PostgresAdmissionStore(_checkpointer_cm),
                                             ttl_seconds=_settings.lease_seconds, reserve_waiting=False)
            await capacity.snapshot()  # Additive migration must be applied first.
        if _settings.admission_enabled:
            from app.admission_runtime import AdmissionRuntime
            admission = AdmissionRuntime(_settings, runner, _checkpointer_cm)
            # Admission migration is operator-managed, unlike saver migrations.
            # Fail startup if it has not been applied; never accept into memory.
            await admission.store.list_pending(limit=1)
            await admission.store.pending_outbox(limit=1)
        if _settings.resume_on_boot:
            try:
                counts = await runner.resume_unfinished()
                logger.info("durable_runs_resume_on_boot %s", counts)
            except Exception:  # noqa: BLE001
                logger.exception("durable_runs_resume_on_boot_failed")
        _stop.clear()
        _sweep_task = asyncio.create_task(runner.sweep_forever(_stop))
        if admission is not None:
            _admission_task = asyncio.create_task(admission.run(_stop))
        if _settings.orion_bus_enabled:
            patterns = [_settings.request_channel, RESOURCE_EVENT_CHANNEL]
            if admission is not None:
                patterns.append(GPU_POOL_EVENT_CHANNEL)  # wakes waiting runs on their hold's grant
            hunter = Hunter(_chassis_cfg(), handler=_handle_request, patterns=patterns)
            await hunter.start_background()
            logger.info("durable_runs_listening channel=%s", _settings.request_channel)
    else:
        logger.info("durable_runs disabled (DURABLE_RUNS_ENABLED=false)")

    try:
        yield
    finally:
        _stop.set()
        if _sweep_task is not None:
            _sweep_task.cancel()
        if _admission_task is not None:
            _admission_task.cancel()
            await asyncio.gather(_admission_task, return_exceptions=True)
        if admission is not None:
            await admission.close()
        for closer in (hunter, heartbeat):
            if closer is not None:
                try:
                    await closer.stop()
                except Exception:  # noqa: BLE001
                    pass
        if rpc_health_publisher is not None:
            await rpc_health_publisher.stop()
            rpc_health_publisher = None
        if rpc_bus is not None:
            try:
                await rpc_bus.close()
            except Exception:  # noqa: BLE001
                pass
        if _checkpointer_cm is not None:
            try:
                await _checkpointer_cm.close()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="orion-durable-runs", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": _settings.service_name,
        "enabled": _settings.enabled,
        "active_runs": runner.active_run_ids if runner is not None else [],
        "admission_enabled": admission is not None,
        "capacity_enabled": capacity is not None,
        "admitted_active_runs": sorted(admission.active) if admission is not None else [],
    }


@app.get("/runs/unfinished")
async def unfinished() -> dict[str, Any]:
    if runner is None:
        return {"threads": []}
    threads = await runner.unfinished_threads()
    return {
        "threads": [
            {"thread_id": t, "next_node": n, "checkpoint_ts": ts.isoformat() if ts else None, "workflow": w}
            for t, n, ts, w in threads
        ]
    }


def _admission():
    if admission is None:
        raise HTTPException(503, "durable resource admission is disabled")
    return admission


@app.post("/runs", status_code=202)
async def submit(request: DurableRunRequestV1):
    try:
        return await _admission().submit(request)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/runs/{run_id}")
async def run_status(run_id: str):
    try:
        return await _admission().status(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.post("/runs/{run_id}/{action}")
async def run_control(run_id: str, action: str):
    if action not in {"pause", "resume", "cancel"}:
        raise HTTPException(400, "action must be pause, resume or cancel")
    try:
        return await _admission().control(run_id, action)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.post("/runs/{run_id}/release-outreach-lease")
async def release_outreach_lease(run_id: str):
    """Hub Door-A finished composing: release the run's GPU pool hold kept past finish.

    Only acts when the run is already ``terminal=completed`` (the Door-A hold). A live in-flight
    investigation cannot be released through this path. Idempotent when the hold is already gone.
    """
    try:
        return await _admission().release_outreach(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _capacity():
    if capacity is None:
        raise HTTPException(503, "Gateway capacity authority is disabled")
    return capacity


@app.post("/capacity/acquire", response_model=CapacityAcquireResultV1)
async def acquire_capacity(request: CapacityAcquireV1):
    try:
        return await _capacity().acquire(request)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/capacity/renew", response_model=CapacityRenewResultV1)
async def renew_capacity(token: CapacityTokenV1):
    return await _capacity().renew(token)


@app.post("/capacity/release", response_model=CapacityReleaseResultV1)
async def release_capacity(token: CapacityTokenV1):
    result = await _capacity().release(token)
    if admission is not None:
        admission._wake.set()
    return result


@app.get("/capacity")
async def capacity_snapshot():
    return await _capacity().snapshot()
