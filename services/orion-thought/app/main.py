from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Body
from orion.schemas.reverie_visual import VisualRunRequestV1, VisualExecutionReceiptV1, VisualProductionReceiptV1
from fastapi.responses import JSONResponse

from datetime import datetime, timezone

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .bus_listener import run_bus_worker
from .chain import run_reverie_chain_worker
from .reasoning_activity import run_reasoning_worker
from .reasoning_activity import store as reasoning_store
from .reverie import run_reverie_worker
from .rpc_health import build_publisher, fold_bus
from .settings import settings
from .store import warm_pool
from .visual_chain import run_visual_chain_watchdog, run_visual_chain_worker

logging.basicConfig(
    level=logging.INFO,
    format="[ORION-THOUGHT] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-thought.main")

heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from the bus worker/reverie/
    reasoning tasks below (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
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
    global heartbeat_chassis
    logger.info(
        "Starting orion-thought service=%s v=%s port=%s",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None
    app.state.bus_stop_event = asyncio.Event()
    app.state.bus_task = asyncio.create_task(run_bus_worker(app.state.bus_stop_event))
    # Spontaneous-thought mode — no-op unless ORION_REVERIE_ENABLED (default off).
    app.state.reverie_stop_event = asyncio.Event()
    app.state.reverie_task = asyncio.create_task(run_reverie_worker(app.state.reverie_stop_event))
    # Reverie chain mode — no-op unless ORION_REVERIE_CHAIN_ENABLED (default off).
    app.state.reverie_chain_stop_event = asyncio.Event()
    app.state.reverie_chain_task = asyncio.create_task(
        run_reverie_chain_worker(app.state.reverie_chain_stop_event)
    )
    # Reasoning-activity projection — always-on consumer. Harmless (empty
    # projection) when no producer is publishing reasoning_call events.
    app.state.reasoning_stop_event = asyncio.Event()
    app.state.reasoning_task = asyncio.create_task(
        run_reasoning_worker(app.state.reasoning_stop_event)
    )
    # Reverie VISUAL chain (Patch 2) — no-op unless ORION_VISUAL_CHAIN_ENABLED
    # (default off).
    app.state.visual_chain_stop_event = asyncio.Event()
    app.state.visual_chain_task = asyncio.create_task(
        run_visual_chain_worker(app.state.visual_chain_stop_event)
    )
    app.state.visual_chain_watchdog_stop_event = asyncio.Event()
    app.state.visual_chain_watchdog_task = asyncio.create_task(
        run_visual_chain_watchdog(app.state.visual_chain_watchdog_stop_event)
    )
    # Warm store.py's shared Postgres pool so the first real caller of any
    # kind (reverie/salience/etc. writes) doesn't pay a cold TCP+auth
    # handshake cost -- unconditional since every store.py consumer shares
    # the one engine, not gated behind any single feature flag. Reference
    # kept on app.state -- an unreferenced asyncio task can be
    # garbage-collected mid-execution (see asyncio docs on create_task).
    app.state.pool_warmup_task = asyncio.create_task(warm_pool())
    # RPC-health publish (orion:rpc_health:snapshot): one dedicated long-lived bus that
    # drains the process sink every worker/per-call bus folds into (app/rpc_health.py).
    app.state.rpc_health_bus = None
    app.state.rpc_health_publisher = build_publisher(settings, lambda: app.state.rpc_health_bus)
    if app.state.rpc_health_publisher.enabled:
        from orion.core.bus.async_service import OrionBusAsync

        try:
            rpc_health_bus = OrionBusAsync(url=settings.orion_bus_url)
            await rpc_health_bus.connect()
            app.state.rpc_health_bus = rpc_health_bus
            app.state.rpc_health_publisher.start()
        except Exception as exc:
            logger.warning("rpc_health_publish_start_failed error=%s", exc)
    yield
    await app.state.rpc_health_publisher.stop()
    if app.state.rpc_health_bus is not None:
        with suppress(Exception):
            await app.state.rpc_health_bus.close()
        app.state.rpc_health_bus = None
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None
    app.state.bus_stop_event.set()
    app.state.reverie_stop_event.set()
    app.state.reverie_chain_stop_event.set()
    app.state.reasoning_stop_event.set()
    app.state.visual_chain_stop_event.set()
    app.state.visual_chain_watchdog_stop_event.set()
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(app.state.bus_task, timeout=125.0)
    if not app.state.bus_task.done():
        app.state.bus_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.bus_task
    for task in (
        app.state.reverie_task,
        app.state.reverie_chain_task,
        app.state.reasoning_task,
        app.state.visual_chain_task,
        app.state.visual_chain_watchdog_task,
        app.state.pool_warmup_task,
    ):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Orion Thought", lifespan=lifespan, version=settings.service_version)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": settings.service_name,
            "version": settings.service_version,
            "bus_enabled": settings.orion_bus_enabled,
            "channel_thought_request": settings.channel_thought_request,
        }
    )


@app.get("/visual-chain/activity")
async def visual_chain_activity() -> JSONResponse:
    from .store import load_visual_activity
    activity = await asyncio.to_thread(load_visual_activity)
    return JSONResponse(activity.model_dump(mode="json"))


@app.post("/visual-chain/run-once")
async def visual_chain_run_once(request: VisualRunRequestV1 | None = Body(default=None)) -> JSONResponse:
    """Execute a typed request; `ran` is legacy, only `outcome` proves production."""
    from orion.core.bus.async_service import OrionBusAsync
    from orion.reverie.baseline import load_baseline_policy, validate_eligibility
    from .visual_chain import run_visual_chain_once, visual_chain_in_flight_for
    from .store import claim_visual_attempt, finish_visual_attempt, persist_visual_execution_receipt, replay_visual_attempt

    request = request if isinstance(request, VisualRunRequestV1) else VisualRunRequestV1()
    policy = load_baseline_policy()
    attempt_id = None
    # Replay does not grant authority to execute. Retrieve the exact immutable
    # dispatch before checking authorization freshness for a new attempt.
    if request.dispatch_id:
        try:
            replay = await asyncio.to_thread(replay_visual_attempt, request)
        except Exception:
            logger.exception("visual execution replay unavailable")
            return JSONResponse({"ok": False, "ran": False, "outcome": "unknown", "reason": "replay_unavailable"})
        if replay is not None:
            return JSONResponse(replay)
    if request.visual_baseline:
        reason = validate_eligibility(request.visual_baseline, policy=policy)
        if reason or settings.visual_chain_enabled:
            return JSONResponse({"ok": False, "ran": False, "outcome": "failed",
                                 "reason": reason or "legacy_visual_worker_enabled"})
    if policy.enabled:
        try:
            attempt_id, replay = await asyncio.to_thread(
                claim_visual_attempt, request, retry_sec=policy.retry_sec, now=datetime.now(timezone.utc)
            )
        except Exception:
            logger.exception("visual execution claim unavailable")
            return JSONResponse({"ok": False, "ran": False, "outcome": "unknown", "reason": "claim_unavailable"})
        if replay is not None:
            return JSONResponse(replay)
    bus = OrionBusAsync(url=settings.orion_bus_url)
    chain = None
    try:
        await bus.connect()
        chain = await run_visual_chain_once(bus=bus, attempt_id=attempt_id, run_request=request.model_dump(mode="json"))
    except Exception:
        # A failed waiter is ambiguous: its blocking GPU thread may still run.
        logger.exception("visual run failed without a terminal chain")
        result = {"ok": False, "ran": False, "outcome": "unknown", "reason": "execution_unresolved", "attempt_id": attempt_id}
    else:
        production = chain.chain_json.get("production_receipt") if chain else None
        outcome = ("produced" if production else "deferred_busy" if chain is None else
                   "deferred_thermal" if chain.terminal_reason == "thermal_refused" else
                   "deferred_resource" if chain.terminal_reason == "resource_deferred" else
                   "unknown" if chain.terminal_reason == "run_deadline_exceeded" else "failed")
        thermal = chain.chain_json.get("thermal_gate") if chain else {"reason": "thermal_not_evaluated"}
        receipt = VisualExecutionReceiptV1(
            request=request, attempt_id=attempt_id or (chain.chain_id if chain else None),
            outcome=outcome, gate_reason=chain.terminal_reason if chain else "already_in_flight",
            thermal_gate=thermal or {},
            source_selection_status="selected" if chain and chain.context_selection else "source_selection_not_reached",
            source_kind=chain.context_selection.source_kind if chain and chain.context_selection else None,
            source_refs=([chain.context_selection.reverie.thought_id, chain.context_selection.reverie.text_chain_id]
                         if chain and chain.context_selection and chain.context_selection.reverie else
                         [chain.context_selection.source.source_id]
                         if chain and chain.context_selection and chain.context_selection.source else []),
            artifact_persisted=production is not None,
            production_receipt=VisualProductionReceiptV1.model_validate(production) if production else None,
        )
        result = {"ok": outcome not in {"failed", "unknown"}, "ran": chain is not None,
                  "outcome": outcome, "attempt_id": receipt.attempt_id,
                  "chain_id": chain.chain_id if chain else None,
                  "terminal_reason": chain.terminal_reason if chain else None,
                  "reason": receipt.gate_reason,
                  "refused": outcome == "deferred_thermal", "detail": thermal,
                  "artifact_persisted": receipt.artifact_persisted,
                  "artifact_sha256": receipt.production_receipt.sha256 if receipt.production_receipt else None,
                  "produced_at": receipt.production_receipt.produced_at.isoformat() if receipt.production_receipt else None,
                  "execution_receipt": receipt.model_dump(mode="json")}
        if chain is None:
            held_sec = visual_chain_in_flight_for()
            result["held_sec"] = None if held_sec is None else round(held_sec, 1)
        else:
            await asyncio.to_thread(persist_visual_execution_receipt, chain.chain_id, receipt)
    finally:
        fold_bus(bus)  # per-call bus: keep its RPC-health window (app/rpc_health.py)
        with suppress(Exception):
            await bus.close()
    if attempt_id:
        try:
            await asyncio.to_thread(finish_visual_attempt, attempt_id, result)
        except Exception:
            # Durable active marker remains for positive-evidence reconciliation.
            logger.exception("visual attempt completion persistence failed")
    return JSONResponse(result)


@app.get("/projections/reasoning_activity")
async def reasoning_activity() -> JSONResponse:
    projection = reasoning_store.snapshot(datetime.now(timezone.utc))
    return JSONResponse({"ok": True, "projection": projection.model_dump(mode="json")})


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
