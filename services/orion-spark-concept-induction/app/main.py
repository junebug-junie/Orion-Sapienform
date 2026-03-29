from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.spark.concept_induction.bus_worker import ConceptWorker
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = ConceptWorker(settings)
    app.state.worker = worker
    app.state.concept_worker = worker
    task = asyncio.create_task(worker.start(), name="concept-induction-worker")
    app.state.worker_task = task
    await asyncio.sleep(0)
    if task.done():
        exc = task.exception()
        if exc is not None:
            logger.exception("concept_induction_worker_startup_failed", exc_info=exc)
            raise exc
    logger.info("concept_induction_worker_startup worker_task=%s", task.get_name())
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
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "worker_attached": True,
        "status": worker.trigger_status(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8500)
