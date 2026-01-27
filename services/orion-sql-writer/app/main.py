from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import Base, engine
from app.settings import settings
from app.worker import build_hunter

logging.basicConfig(
    level=logging.INFO,
    format="[SQL_WRITER] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("sql-writer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure schema exists
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🛠️  Ensured DB schema is present")
    except Exception as e:
        logger.warning("Schema init warning: %s", e)

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS correlation_id TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_message_corr_id ON chat_message (correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_message_corr_role_ts ON chat_message (correlation_id, role, timestamp);"
            )
        logger.info("🧬 chat_message correlation/trace columns ensured")
    except Exception as e:
        logger.warning("chat_message migration warning: %s", e)

    task: asyncio.Task | None = None
    if settings.orion_bus_enabled:
        svc = build_hunter()
        logger.info("🚀 starting Hunter")
        logger.info("🧲 sql-writer subscribing to channels: %s", settings.effective_subscribe_channels)
        task = asyncio.create_task(svc.start())
    else:
        logger.warning("Bus disabled; writer will be idle.")

    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(Exception):
                await task


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}
