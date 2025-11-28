# services/orion-agent-council/app/main.py
from __future__ import annotations

import logging
import threading

from fastapi import FastAPI

from orion.core.bus.service import OrionBus
from .settings import settings
from .council import run_council_loop

logger = logging.getLogger("agent-council.main")


def _start_council_thread() -> None:
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        logger.error("Orion bus disabled; council thread will not start.")
        return

    t = threading.Thread(target=run_council_loop, args=(bus,), daemon=True)
    t.start()
    logger.info("Started council worker thread: %s", t.name)


app = FastAPI(title="Orion Agent Council Service")


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(
        level=logging.INFO,
        format="[AGENT-COUNCIL] %(levelname)s - %(name)s - %(message)s",
    )
    logger.info(
        "Starting Agent Council service (name=%s v=%s)",
        settings.service_name,
        settings.service_version,
    )
    _start_council_thread()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "intake_channel": settings.channel_intake,
        "llm_intake_channel": settings.llm_intake_channel,
    }
