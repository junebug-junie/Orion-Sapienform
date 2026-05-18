from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from orion.mind.v1 import MindRunRequestV1, MindRunResultV1

from .engine import run_mind
from .settings import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("orion-mind")


def _log_timeout_hierarchy_warnings() -> None:
    mind_llm = float(settings.MIND_LLM_TIMEOUT_SEC)
    orch_outer = settings.ORION_MIND_TIMEOUT_SEC
    if orch_outer is not None and mind_llm >= float(orch_outer):
        logger.warning(
            "mind_timeout_hierarchy_invalid MIND_LLM_TIMEOUT_SEC=%s ORION_MIND_TIMEOUT_SEC=%s "
            "recommended MIND_LLM_TIMEOUT_SEC=25 ORION_MIND_TIMEOUT_SEC=45 "
            "(inner LLM phase timeout must be less than Orch Mind HTTP timeout)",
            mind_llm,
            orch_outer,
        )
    elif mind_llm >= 45.0:
        logger.warning(
            "mind_timeout_hierarchy_risky MIND_LLM_TIMEOUT_SEC=%s "
            "recommended=25 with ORION_MIND_TIMEOUT_SEC=45 on Orch",
            mind_llm,
        )
    logger.info(
        "mind_startup_config MIND_LLM_TIMEOUT_SEC=%s MIND_LLM_SYNTHESIS_ENABLED=%s "
        "routes semantic=%s appraisal=%s stance=%s intake=%s reply_prefix=%s",
        mind_llm,
        settings.MIND_LLM_SYNTHESIS_ENABLED,
        settings.MIND_SEMANTIC_MODEL_ROUTE,
        settings.MIND_APPRAISAL_MODEL_ROUTE,
        settings.MIND_STANCE_MODEL_ROUTE,
        settings.MIND_LLM_INTAKE_CHANNEL,
        settings.MIND_LLM_REPLY_PREFIX,
    )


_log_timeout_hierarchy_warnings()

app = FastAPI(title="Orion Mind", version=settings.SERVICE_VERSION)


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
    return run_mind(
        body,
        router_profiles_dir=router_dir,
        snapshot_max_bytes=settings.MIND_SNAPSHOT_MAX_BYTES,
        mind_settings=settings,
    )
