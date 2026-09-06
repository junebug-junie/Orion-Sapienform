from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter
from orion.schemas.durable_run import DURABLE_RUN_REQUEST_KIND, DurableRunRequestV1

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
    await runner.start_run(request)


async def _open_checkpointer():
    """LangGraph's Postgres saver, tables created on first boot. Its context
    manager owns the connection pool; we hold it open for the process life."""
    global _checkpointer_cm
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(_settings.postgres_uri)
    saver = await _checkpointer_cm.__aenter__()
    await saver.setup()
    return saver


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, rpc_bus, hunter, heartbeat, _sweep_task
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
        runner = DurableRunner(_settings, bus=rpc_bus, checkpointer=saver)
        if _settings.resume_on_boot:
            try:
                counts = await runner.resume_unfinished()
                logger.info("durable_runs_resume_on_boot %s", counts)
            except Exception:  # noqa: BLE001
                logger.exception("durable_runs_resume_on_boot_failed")
        _stop.clear()
        _sweep_task = asyncio.create_task(runner.sweep_forever(_stop))
        if _settings.orion_bus_enabled:
            hunter = Hunter(_chassis_cfg(), handler=_handle_request, patterns=[_settings.request_channel])
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
        for closer in (hunter, heartbeat):
            if closer is not None:
                try:
                    await closer.stop()
                except Exception:  # noqa: BLE001
                    pass
        if rpc_bus is not None:
            try:
                await rpc_bus.close()
            except Exception:  # noqa: BLE001
                pass
        if _checkpointer_cm is not None:
            try:
                await _checkpointer_cm.__aexit__(None, None, None)
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
    }


@app.get("/runs/unfinished")
async def unfinished() -> dict[str, Any]:
    if runner is None:
        return {"threads": []}
    threads = await runner.unfinished_threads()
    return {"threads": [{"thread_id": t, "next_node": n, "checkpoint_ts": ts.isoformat() if ts else None} for t, n, ts in threads]}
