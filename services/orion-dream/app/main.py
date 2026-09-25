# ==================================================
# main.py — Unified entrypoint for Orion Dream
# ==================================================
import asyncio
import logging
from datetime import timezone
from fastapi import FastAPI
from fastapi import HTTPException
from contextlib import asynccontextmanager

from orion.core.bus.bus_service_chassis import OrionBusAsync
from orion.core.bus.enforce import enforcer
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.dream import DreamTriggerPayload
from app.settings import settings
from app.dream_api import router as dream_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dream-app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Dream execution is owned by cortex-orch (dream.trigger -> dream_cycle).
    This service provides HTTP readout and publishes compatibility triggers only.
    With ORION_DREAM_CYCLE_ENABLED it also runs the dream cycle v2 sleep loop.
    """
    logger.info("🌙 Orion Dream module starting up (readout façade; triggers go to cortex-orch)…")

    stop = asyncio.Event()
    loop_task = None
    if settings.ORION_DREAM_CYCLE_ENABLED:
        from app.cycle import sleep_loop

        # The bus connects lazily inside the first LLM call and reconnects
        # after a failure, so a bus that is down at boot does not kill the loop.
        loop_task = asyncio.create_task(sleep_loop(build_cycle_deps(), stop))
        logger.info("dream cycle v2 sleep loop started")

    yield

    stop.set()
    if loop_task is not None:
        try:
            await asyncio.wait_for(loop_task, timeout=settings.SHUTDOWN_GRACE_SEC)
        except Exception:
            loop_task.cancel()
    bus = _CYCLE_STATE.get("bus")
    if bus is not None:
        await bus.close()
    logger.info("💤 Orion Dream module shutting down…")


# Process-local cycle state: the lazily-connected bus, plus in-process floors
# for the two cycle clocks. The floors keep the min interval and the replay
# window honest when persistence fails (e.g. migration not applied): without
# them the db reads "never slept" and the loop would re-run every tick.
_CYCLE_STATE: dict = {}
_BUS_LOCK = asyncio.Lock()


async def _cycle_bus():
    if not settings.ORION_BUS_ENABLED:
        return None
    async with _BUS_LOCK:
        bus = _CYCLE_STATE.get("bus")
        if bus is None:
            bus = OrionBusAsync(settings.ORION_BUS_URL)
            await bus.connect()
            _CYCLE_STATE["bus"] = bus
        return bus


async def _drop_cycle_bus():
    async with _BUS_LOCK:
        bus = _CYCLE_STATE.pop("bus", None)
    if bus is not None:
        try:
            await bus.close()
        except Exception:
            pass


def _aware(dt):
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _latest(*values):
    return max((_aware(v) for v in values if v is not None), default=None)


def build_cycle_deps():
    """Real IO for the dream cycle. Tests build CycleDeps with fakes instead."""
    from app import cycle_store, llm, rem_store
    from app.cycle import CycleDeps

    async def _complete(prompt: str) -> str:
        bus = await _cycle_bus()
        if bus is None:
            raise RuntimeError("bus disabled: no LLM for recombination")
        try:
            return await llm.complete(bus, prompt)
        except Exception:
            # A timeout is usually the gateway, not the connection -- but a
            # dead connection looks the same from here, and reconnecting on
            # the next call is cheap.
            await _drop_cycle_bus()
            raise

    async def _rem(cycle_id: str, since):
        from app.rem_compaction import run_rem_compaction_once

        # run_rem_compaction_once calls its loader/persister synchronously;
        # do the DB IO in threads here and hand it pure in-memory seams.
        requests = await asyncio.to_thread(
            rem_store.load_pending_requests, settings.DREAM_REM_MAX_REQUESTS, since
        )
        staged: list = []
        try:
            bus = await _cycle_bus()
        except Exception:
            bus = None
        delta = await run_rem_compaction_once(
            bus,
            request_loader=lambda _n: requests,
            delta_persister=lambda d: staged.append(d) or True,
            dream_id=cycle_id,
        )
        for d in staged:
            await asyncio.to_thread(rem_store.persist_compaction_delta, d)
        return delta.delta_id if delta is not None else None

    def _window_start():
        return _latest(cycle_store.load_last_window_start(), _CYCLE_STATE.get("window_start"))

    def _attempt_end():
        return _latest(cycle_store.load_last_attempt_end(), _CYCLE_STATE.get("attempt_end"))

    def _persist(cycle) -> bool:
        _CYCLE_STATE["attempt_end"] = cycle.ended_at
        if cycle.status != "failed":
            _CYCLE_STATE["window_start"] = cycle.started_at
        return cycle_store.persist_cycle(cycle)

    return CycleDeps(
        load_source_rows=cycle_store.load_source_rows,
        load_idle_minutes=cycle_store.load_idle_minutes,
        load_last_window_start=_window_start,
        load_last_attempt_end=_attempt_end,
        persist_cycle=_persist,
        complete=_complete,
        rem_compaction=_rem,
    )


app = FastAPI(
    title="Orion Dream Module",
    description="Generates nightly dreams from Orion’s stored memories.",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

# Dream readout routes
app.include_router(dream_router, prefix="/dreams")

@app.post("/dreams/run", summary="Manually run the dream cycle")
async def run_dream_endpoint(mode: str = "standard"):
    """
    Publishes `dream.trigger` on the bus; cortex-orch normalizes it to `dream_cycle`.
    """
    if not settings.ORION_BUS_ENABLED:
        return {"error": "Bus disabled"}

    trigger = DreamTriggerPayload(mode=mode)
    channel = settings.CHANNEL_DREAM_TRIGGER
    catalog_entry = enforcer.entry_for(channel)
    catalog_schema_id = catalog_entry.get("schema_id") if catalog_entry else None
    logger.info(
        "Dream trigger request received channel=%s kind=%s payload_schema=%s catalog_schema=%s catalog_present=%s mode=%s",
        channel,
        "dream.trigger",
        type(trigger).__name__,
        catalog_schema_id,
        bool(catalog_entry),
        mode,
    )
    logger.debug("Dream trigger payload=%s", trigger.model_dump(mode="json"))

    # cortex-orch consumes it and normalizes to dream_cycle
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    try:
        logger.info("Connecting dream trigger bus url=%s", settings.ORION_BUS_URL)
        await bus.connect()
        env = BaseEnvelope(
            kind="dream.trigger",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
            payload=trigger.model_dump(mode="json")
        )
        logger.info(
            "Publishing dream trigger channel=%s kind=%s payload_schema=%s correlation_id=%s",
            channel,
            env.kind,
            type(trigger).__name__,
            env.correlation_id,
        )
        await bus.publish(channel, env)
        return {"status": "triggered", "mode": mode}
    except Exception as exc:
        logger.exception(
            "Dream trigger publish failed channel=%s kind=%s payload_schema=%s catalog_schema=%s bus_url=%s",
            channel,
            "dream.trigger",
            type(trigger).__name__,
            catalog_schema_id,
            settings.ORION_BUS_URL,
        )
        hint = (
            "Verify Redis connectivity plus orion/bus/channels.yaml and "
            "orion/schemas/registry.py registrations for dream.trigger."
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "dream_trigger_publish_failed",
                "channel": channel,
                "kind": "dream.trigger",
                "schema_id": type(trigger).__name__,
                "catalog_schema_id": catalog_schema_id,
                "message": str(exc),
                "hint": hint,
            },
        ) from exc
    finally:
        await bus.close()


@app.post("/dreams/rem-preview", summary="Stage a REM compaction delta (Phase F)")
async def rem_preview_endpoint():
    """Run one REM compaction pass — proposes what sleep *would* do to memory.

    Staged and hard-gated: returns `{"status": "disabled"}` unless
    ORION_DREAM_REM_ENABLED. Even when enabled it applies **nothing** — it emits a
    proposal-marked MemoryCompactionDeltaV1 for the hub "what sleep would do"
    panel and persists it to the staging table.
    """
    if not settings.ORION_DREAM_REM_ENABLED:
        return {"status": "disabled", "reason": "ORION_DREAM_REM_ENABLED is false"}

    from app.rem_compaction import run_rem_compaction_once

    bus = OrionBusAsync(settings.ORION_BUS_URL) if settings.ORION_BUS_ENABLED else None
    try:
        if bus is not None:
            await bus.connect()
        delta = await run_rem_compaction_once(bus)
    finally:
        if bus is not None:
            await bus.close()

    if delta is None:
        return {"status": "empty", "reason": "nothing settled to compact"}
    return {
        "status": "staged",
        "delta_id": delta.delta_id,
        "cards_out": delta.metrics.cards_out,
        "proposal_marked": delta.proposal_marked,
        "applied": False,
    }


@app.get("/dreams/cycle/pressure", summary="Current sleep pressure (read-only)")
async def cycle_pressure_endpoint():
    """What the sleep loop would see right now. Reads only; runs nothing."""
    from datetime import datetime

    from app.cycle import read_pressure, too_soon

    deps = build_cycle_deps()
    now = datetime.now(timezone.utc)
    last_start = await asyncio.to_thread(deps.load_last_window_start)
    last_end = await asyncio.to_thread(deps.load_last_attempt_end)
    pressure, candidates = await asyncio.to_thread(read_pressure, deps, now, last_start)
    return {
        "enabled": settings.ORION_DREAM_CYCLE_ENABLED,
        "pressure": pressure.model_dump(mode="json"),
        "is_idle": pressure.is_idle,
        "should_sleep": pressure.should_sleep,
        "too_soon": too_soon(now, last_end),
        "candidates": len(candidates),
    }


@app.post("/dreams/cycle/run", summary="Run one dream cycle v2 now")
async def cycle_run_endpoint(force: bool = False):
    """Run one sleep. `force=true` skips the pressure/idle/interval gates.

    Still refuses when ORION_DREAM_CYCLE_ENABLED is false: the flag is the
    kill switch for the whole cycle, not just the background loop.
    """
    if not settings.ORION_DREAM_CYCLE_ENABLED:
        return {"status": "disabled", "reason": "ORION_DREAM_CYCLE_ENABLED is false"}
    from app.cycle import run_cycle_once

    cycle = await run_cycle_once(build_cycle_deps(), trigger="manual", force=force)
    if cycle is None:
        return {"status": "not_due"}
    return {
        "status": cycle.status,
        "cycle_id": cycle.cycle_id,
        "pressure": cycle.pressure.pressure,
        "replay": len(cycle.replay),
        "hypotheses": [
            {"hypothesis_id": h.hypothesis_id, "claim": h.claim, "ref_a": h.ref_a, "ref_b": h.ref_b}
            for h in cycle.hypotheses
        ],
        "no_link": cycle.no_link_count,
        "unparseable": cycle.unparseable_count,
        "llm_failures": cycle.llm_failures,
    }
