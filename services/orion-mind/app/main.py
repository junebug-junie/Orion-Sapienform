from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.rpc_health_publish import RpcHealthPublisher
from orion.mind.v1 import MindRunRequestV1, MindRunResultV1
# One tracker per process: every loop instance (orion/mindloop/) yields to the same Juniper turn.
from orion.mindloop.turns import JUNIPER_TURNS

from .engine import carries_live_turn, run_mind
from .llm_client import RPC_HEALTH_SINK, MindLLMClient
from .prime_consent import PrimeConsentResponder
from .prime_loop import PrimeRuntime
from .replica_loop import ReplicaRuntime
from .settings import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("orion-mind")

heartbeat_chassis: HeartbeatOnly | None = None
rpc_health_bus: OrionBusAsync | None = None
rpc_health_publisher: RpcHealthPublisher | None = None
prime_bus: OrionBusAsync | None = None
prime_runtime: PrimeRuntime | None = None
replica_bus: OrionBusAsync | None = None
replica_runtime: ReplicaRuntime | None = None
consent_bus: OrionBusAsync | None = None
consent_responder: PrimeConsentResponder | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. Deliberately separate from the LLM-gateway RPC bus fork used
    by MIND_LLM_USE_BUS (app/llm_client.py) -- this service had no FastAPI lifespan at all
    before this patch. See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


def build_rpc_health_publisher(bus: OrionBusAsync) -> RpcHealthPublisher:
    """Publishes RpcHealthSnapshotV1 from ``bus`` (long-lived, publish-only) after draining
    RPC_HEALTH_SINK, where every per-call LLM-gateway bus folds its rpc_request() outcome
    (app/llm_client.py). Without the sink those per-call stats were discarded."""
    return RpcHealthPublisher(
        enabled=settings.RPC_HEALTH_PUBLISH_ENABLED and settings.ORION_BUS_ENABLED,
        bus_getter=lambda: bus,
        service=settings.SERVICE_NAME,
        node=settings.NODE_NAME,
        instance="main",
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        interval_sec=settings.RPC_HEALTH_PUBLISH_INTERVAL_SEC,
        include_channel_latency=settings.RPC_HEALTH_CHANNEL_LATENCY_ENABLED,
        # Dedicated publish-only bus: the task retries connect() with backoff, so a
        # mesh blip at boot does not disable publishing for the process lifetime.
        connect_bus=True,
        sinks=[RPC_HEALTH_SINK],
    )


async def _start_rpc_health() -> None:
    global rpc_health_bus, rpc_health_publisher
    if not (settings.RPC_HEALTH_PUBLISH_ENABLED and settings.ORION_BUS_ENABLED):
        return
    try:
        bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
        rpc_health_bus = bus
        rpc_health_publisher = build_rpc_health_publisher(bus)
        rpc_health_publisher.start()
    except Exception as exc:
        logger.warning("rpc_health_publish_start_failed error=%s", exc)


async def _stop_rpc_health() -> None:
    global rpc_health_bus, rpc_health_publisher
    if rpc_health_publisher is not None:
        await rpc_health_publisher.stop()
        rpc_health_publisher = None
    if rpc_health_bus is not None:
        try:
            await rpc_health_bus.close()
        except Exception as exc:
            logger.warning("rpc_health_bus_close_error error=%s", exc)
        rpc_health_bus = None


async def _start_prime() -> None:
    """Orion Prime 2A: the resident loop (app/prime_loop.py) on its own bus connection. A failed
    start is logged and leaves run_mind exactly as before."""
    global prime_bus, prime_runtime
    if not (settings.PRIME_ENABLED and settings.ORION_BUS_ENABLED):
        logger.info("prime_disabled PRIME_ENABLED=%s ORION_BUS_ENABLED=%s", settings.PRIME_ENABLED, settings.ORION_BUS_ENABLED)
        return
    try:
        bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)
        prime_bus = bus
        prime_runtime = PrimeRuntime(settings=settings, bus=bus, llm=MindLLMClient(rpc_health=False))
        await prime_runtime.start(connect=True)
    except Exception as exc:
        logger.warning("prime_start_failed error=%s", exc)
        prime_runtime = None


async def _stop_prime() -> None:
    global prime_bus, prime_runtime
    if prime_runtime is not None:
        await prime_runtime.stop()
        prime_runtime = None
    if prime_bus is not None:
        try:
            await prime_bus.close()
        except Exception as exc:
            logger.warning("prime_bus_close_error error=%s", exc)
        prime_bus = None


async def _start_replica() -> None:
    """The curiosity replica (app/replica_loop.py), loop-library instance #2, on its own bus connection.
    A failed start is logged; Hub's replica-arm runs then time out to the uncertainty order."""
    global replica_bus, replica_runtime
    if not (settings.CURIOSITY_REPLICA_ENABLED and settings.ORION_BUS_ENABLED):
        logger.info("curiosity_replica_disabled CURIOSITY_REPLICA_ENABLED=%s ORION_BUS_ENABLED=%s",
                    settings.CURIOSITY_REPLICA_ENABLED, settings.ORION_BUS_ENABLED)
        return
    try:
        bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)
        replica_bus = bus
        replica_runtime = ReplicaRuntime(settings=settings, bus=bus, llm=MindLLMClient(rpc_health=False))
        await replica_runtime.start(connect=True)
    except Exception as exc:
        logger.warning("curiosity_replica_start_failed error=%s", exc)
        replica_runtime = None


async def _stop_replica() -> None:
    global replica_bus, replica_runtime
    if replica_runtime is not None:
        await replica_runtime.stop()
        replica_runtime = None
    if replica_bus is not None:
        try:
            await replica_bus.close()
        except Exception as exc:
            logger.warning("curiosity_replica_bus_close_error error=%s", exc)
        replica_bus = None


async def _start_consent() -> None:
    """Orion's consent to a Claude tool call (app/prime_consent.py) on its own bus connection. A failed
    start is logged; every consent ask then finds no listener and the broker reads it as a no."""
    global consent_bus, consent_responder
    if not (settings.PRIME_CONSENT_ENABLED and settings.ORION_BUS_ENABLED):
        logger.info("prime_consent_disabled PRIME_CONSENT_ENABLED=%s ORION_BUS_ENABLED=%s",
                    settings.PRIME_CONSENT_ENABLED, settings.ORION_BUS_ENABLED)
        return
    try:
        bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)
        consent_bus = bus
        consent_responder = PrimeConsentResponder(settings=settings, bus=bus, llm=MindLLMClient(rpc_health=False))
        await consent_responder.start()
    except Exception as exc:
        logger.warning("prime_consent_start_failed error=%s", exc)
        consent_responder = None


async def _stop_consent() -> None:
    global consent_bus, consent_responder
    if consent_responder is not None:
        await consent_responder.stop()
        consent_responder = None
    if consent_bus is not None:
        try:
            await consent_bus.close()
        except Exception as exc:
            logger.warning("prime_consent_bus_close_error error=%s", exc)
        consent_bus = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None
    await _start_rpc_health()
    await _start_prime()
    await _start_replica()
    await _start_consent()
    try:
        yield
    finally:
        await _stop_consent()
        await _stop_replica()
        await _stop_prime()
        await _stop_rpc_health()
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None


def _log_timeout_hierarchy_warnings() -> None:
    mind_llm = float(settings.MIND_LLM_TIMEOUT_SEC)
    orch_outer = settings.ORION_MIND_TIMEOUT_SEC
    min_viable_wall_ms = mind_llm * 3.0 * 1000.0
    if float(settings.MIND_WALL_MS_DEFAULT) < min_viable_wall_ms:
        logger.warning(
            "mind_wall_hierarchy_risky MIND_WALL_MS_DEFAULT=%s min_viable_wall_ms=%s "
            "recommended MIND_WALL_MS_DEFAULT=180000",
            settings.MIND_WALL_MS_DEFAULT,
            int(min_viable_wall_ms),
        )
    if orch_outer is not None and mind_llm >= float(orch_outer):
        logger.warning(
            "mind_timeout_hierarchy_invalid MIND_LLM_TIMEOUT_SEC=%s ORION_MIND_TIMEOUT_SEC=%s "
            "recommended MIND_LLM_TIMEOUT_SEC=60 ORION_MIND_TIMEOUT_SEC=210 "
            "(inner LLM phase timeout must be less than Orch Mind HTTP timeout)",
            mind_llm,
            orch_outer,
        )
    elif orch_outer is not None and (float(orch_outer) * 1000.0) <= float(settings.MIND_WALL_MS_DEFAULT):
        logger.warning(
            "mind_timeout_hierarchy_risky ORION_MIND_TIMEOUT_SEC=%s MIND_WALL_MS_DEFAULT=%s "
            "recommended ORION_MIND_TIMEOUT_SEC=210",
            orch_outer,
            settings.MIND_WALL_MS_DEFAULT,
        )
    held = float(settings.MIND_LLM_HELD_TIMEOUT_SEC)
    if float(settings.MIND_WALL_MS_DEFAULT) < held * 3.0 * 1000.0:
        # Expected with the shipped values (150 s x 3 > 180 s): a held run's first phase may use most
        # of the wall, and later phases are then skipped (wall_budget_insufficient). Logged so the
        # trade is visible, not hidden.
        logger.info(
            "mind_held_wall_note MIND_LLM_HELD_TIMEOUT_SEC=%s MIND_WALL_MS_DEFAULT=%s: three full held "
            "phases do not fit the wall; a slow early phase leaves later phases skipped",
            held,
            settings.MIND_WALL_MS_DEFAULT,
        )
    if orch_outer is not None and held >= float(orch_outer):
        logger.warning(
            "mind_timeout_hierarchy_invalid MIND_LLM_HELD_TIMEOUT_SEC=%s ORION_MIND_TIMEOUT_SEC=%s "
            "(a held phase cap must be less than the caller's Mind HTTP timeout)",
            held,
            orch_outer,
        )
    logger.info(
        "mind_startup_config MIND_LLM_TIMEOUT_SEC=%s MIND_LLM_HELD_TIMEOUT_SEC=%s MIND_LLM_SYNTHESIS_ENABLED=%s "
        "routes semantic=%s appraisal=%s stance=%s intake=%s reply_prefix=%s",
        mind_llm,
        held,
        settings.MIND_LLM_SYNTHESIS_ENABLED,
        settings.MIND_SEMANTIC_MODEL_ROUTE,
        settings.MIND_APPRAISAL_MODEL_ROUTE,
        settings.MIND_STANCE_MODEL_ROUTE,
        settings.MIND_LLM_INTAKE_CHANNEL,
        settings.MIND_LLM_REPLY_PREFIX,
    )


_log_timeout_hierarchy_warnings()

app = FastAPI(title="Orion Mind", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
    }


@app.post("/v1/mind/run", response_model=MindRunResultV1)
async def mind_run(body: MindRunRequestV1) -> MindRunResultV1:
    router_dir: Path = settings.router_profiles_dir
    if not carries_live_turn(body):
        return run_mind(
            body,
            router_profiles_dir=router_dir,
            snapshot_max_bytes=settings.MIND_SNAPSHOT_MAX_BYTES,
            mind_settings=settings,
        )
    # Juniper's turn: Prime does not bid while it runs and discards a thought it overlapped.
    with JUNIPER_TURNS.turn():
        return run_mind(
            body,
            router_profiles_dir=router_dir,
            snapshot_max_bytes=settings.MIND_SNAPSHOT_MAX_BYTES,
            mind_settings=settings,
        )
