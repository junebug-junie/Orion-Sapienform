from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .bus_listener import run_bus_worker
from .cancel_listener import run_cancel_worker
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="[ORION-HARNESS-GOV] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-harness-governor.main")


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from run_bus_worker/run_cancel_worker's
    own bus connections (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting orion-harness-governor service=%s v=%s port=%s",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    app.state.bus_stop_event = asyncio.Event()
    # Two independent dispatch loops, same code, one per compute lane -- see
    # bus_listener.run_bus_worker's docstring. A long agent-lane turn
    # (curiosity, Mode=Agent+Compute=Agent) and a chat-lane turn each get
    # their own queue now, instead of sharing the one line that let a
    # 40-minute agent-lane run silently block real chat (confirmed live
    # 2026-09-07). Each still opens its own dedicated pubsub connection (that
    # is what actually lets the two loops poll independently) but they share
    # ONE command connection for replies/artifacts, so this is one extra bus
    # connection overall, not two. `lane=` alone decides the channel -- see
    # `_LANE_DEFAULT_CHANNELS` -- so these two calls cannot pass a
    # mismatched lane/channel pair.
    # `enabled=` explicit, not the class default of True: OrionBusAsync.connect()
    # only no-ops on a disabled bus if `.enabled` actually reflects settings --
    # run_bus_worker's own enabled checks (inside the tasks below) never even
    # run until AFTER this line, so without this a disabled bus would still
    # open a real connection right here.
    app.state.dispatch_bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    await app.state.dispatch_bus.connect()
    # KNOWN RISK, not fixed here: both loops' `claude -p` subprocesses run
    # against the SAME shared checkout (HARNESS_FCC_WORKSPACE), protected only
    # by orion/fcc/turn_lock.py's SHARED (non-exclusive) lock. That lock was
    # designed for the rare hub-bridge-vs-governor case; this split makes
    # governor-vs-governor concurrency (a real chat turn overlapping a
    # curiosity investigation) the COMMON case instead. Neither lock nor
    # workspace changed here -- true per-turn workspace isolation is a
    # separate, larger change. See this PR's report for the tradeoff.
    app.state.bus_task_chat = asyncio.create_task(
        run_bus_worker(stop_event=app.state.bus_stop_event, lane="chat", bus=app.state.dispatch_bus)
    )
    app.state.bus_task_agent = asyncio.create_task(
        run_bus_worker(stop_event=app.state.bus_stop_event, lane="agent", bus=app.state.dispatch_bus)
    )
    app.state.cancel_task = asyncio.create_task(run_cancel_worker(app.state.bus_stop_event))
    app.state.heartbeat_chassis = build_heartbeat_chassis()
    try:
        await app.state.heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        app.state.heartbeat_chassis = None
    yield
    app.state.bus_stop_event.set()
    for task in (
        app.state.bus_task_chat,
        app.state.bus_task_agent,
        getattr(app.state, "cancel_task", None),
    ):
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    dispatch_bus = getattr(app.state, "dispatch_bus", None)
    if dispatch_bus is not None:
        with suppress(Exception):
            await dispatch_bus.close()
    heartbeat_chassis = getattr(app.state, "heartbeat_chassis", None)
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)


app = FastAPI(title="Orion Harness Governor", lifespan=lifespan, version=settings.service_version)


def _lane_alive(task: "asyncio.Task | None") -> bool | None:
    """None means "not applicable" (bus/governor disabled by config -- the
    loop returning immediately is expected, not a crash). True/False is only
    meaningful when the bus and governor are actually supposed to be running,
    so a dashboard or alert built on this field can't mistake "turned off on
    purpose" for "the dispatch loop died"."""
    if not settings.orion_bus_enabled or not settings.orion_harness_governor_enabled:
        return None
    return task is not None and not task.done()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": settings.service_name,
            "version": settings.service_version,
            "bus_enabled": settings.orion_bus_enabled,
            "governor_enabled": settings.orion_harness_governor_enabled,
            "channel_harness_run_request": settings.channel_harness_run_request,
            "channel_harness_run_request_agent": settings.channel_harness_run_request_agent,
            # A done-but-not-cancelled task, while bus+governor are both
            # enabled, means that lane's dispatch loop died silently -- worth
            # surfacing directly rather than only as an absence of processed
            # turns some time later. `null` here means disabled by config,
            # not a crash -- see `_lane_alive`.
            "lane_chat_alive": _lane_alive(getattr(app.state, "bus_task_chat", None)),
            "lane_agent_alive": _lane_alive(getattr(app.state, "bus_task_agent", None)),
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
