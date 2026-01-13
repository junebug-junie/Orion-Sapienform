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
    task = asyncio.create_task(worker.start(), name="concept-induction-worker")
    logger.info("Concept Induction worker booted.")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Concept Induction worker stopped.")


app = FastAPI(title="Orion Spark Concept Induction", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.service_name, "version": settings.service_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8500)
