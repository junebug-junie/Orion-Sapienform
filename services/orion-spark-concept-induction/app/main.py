from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
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
