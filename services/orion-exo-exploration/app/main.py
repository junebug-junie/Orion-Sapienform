from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.crawl.daemon import crawl_loop, retention_sweep_loop
from app.routers.crawl_runs import router as crawl_runs_router
from app.routers.finds import router as finds_router
from app.routers.health import router as health_router
from app.routers.interest_rules import router as interest_rules_router
from app.settings import settings
from app.storage.repository import ensure_tables

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-exo-exploration")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_tables()
    logger.info("orion-exo-exploration starting")
    tasks: list[asyncio.Task] = []
    if settings.exo_exploration_daemon_enabled:
        tasks.append(asyncio.create_task(crawl_loop()))
        tasks.append(asyncio.create_task(retention_sweep_loop()))
        logger.info(
            "exo_exploration_daemon_started crawl_interval_sec=%d retention_sweep_interval_sec=%d",
            settings.exo_exploration_crawl_interval_seconds,
            settings.exo_exploration_retention_sweep_interval_seconds,
        )
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="Orion Exo Exploration", version=settings.service_version, lifespan=lifespan)

app.include_router(health_router)
app.include_router(finds_router)
app.include_router(crawl_runs_router)
app.include_router(interest_rules_router)
