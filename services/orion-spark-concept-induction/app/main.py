from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.spark.concept_induction.bus_worker import ConceptWorker
from .settings import settings


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru.

    The worker (`orion.spark.concept_induction.*`, `orion.autonomy.*`) logs via
    stdlib `logging`, while this service logs via loguru. Without this bridge the
    stdlib loggers have no handler and INFO records fall through to stdlib's
    WARNING-only lastResort handler, silently dropping worker traces such as
    `substrate_policy_act`.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _install_stdlib_logging_bridge(level: str = "INFO") -> None:
    log_level = logging.getLevelName(str(level).strip().upper())
    if not isinstance(log_level, int):
        log_level = logging.INFO
    logging.basicConfig(handlers=[_InterceptHandler()], level=log_level, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "orion", "orion.autonomy"):
        std_logger = logging.getLogger(name)
        std_logger.handlers = [_InterceptHandler()]
        std_logger.propagate = False
        std_logger.setLevel(log_level)


heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from ConceptWorker's own bus
    connection (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md).
    ConceptSettings already carries a heartbeat_interval_sec field (added ahead of this wiring,
    never previously consumed by any producer -- this is that producer)."""
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
    _install_stdlib_logging_bridge(settings.log_level)
    worker = ConceptWorker(settings)
    app.state.worker = worker
    app.state.concept_worker = worker
    logger.info("concept_induction_worker_starting")
    task = asyncio.create_task(worker.start(), name="concept-induction-worker")
    worker.mark_task_created()
    task.add_done_callback(worker.record_task_exit)
    app.state.worker_task = task
    await asyncio.sleep(0)
    if task.done():
        exc = task.exception()
        if exc is not None:
            logger.exception("concept_induction_worker_startup_failed", exc_info=exc)
            raise exc
    logger.info("concept_induction_worker_started worker_task=%s", task.get_name())
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service={} interval_sec={}",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error={}", exc)
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error={}", exc)
        heartbeat_chassis = None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("concept_induction_worker_cancelled")
    finally:
        await worker.stop()
        logger.info("concept_induction_worker_stopped")


app = FastAPI(title="Orion Spark Concept Induction", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.service_name, "version": settings.service_version}


@app.get("/debug/concept-induction")
async def concept_induction_debug():
    worker = getattr(app.state, "worker", None) or getattr(app.state, "concept_worker", None)
    if worker is None:
        return {"ok": False, "error": "worker_not_initialized"}
    worker_task = getattr(app.state, "worker_task", None)
    task_state = {
        "worker_task_present": worker_task is not None,
        "worker_task_done": worker_task.done() if worker_task else None,
        "worker_task_cancelled": worker_task.cancelled() if worker_task else None,
    }
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "worker_attached": True,
        "worker_liveness": worker.worker_liveness_status(),
        "worker_task": task_state,
        "status": worker.trigger_status(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8510)
