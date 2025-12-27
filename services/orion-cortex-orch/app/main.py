from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress
from typing import Any, Dict, Optional

import orjson
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.core.bus.service_async import OrionBusAsync

from .bus_models import CortexOrchBusReply, CortexOrchBusRequest
from .orchestrator import OrchestrateVerbRequest, OrchestrateVerbResponse, run_cortex_verb
from .settings import get_settings


class ORJSONResponse(JSONResponse):
    """FastAPI response class using orjson."""

    media_type = "application/json"

    def render(self, content: Dict[str, Any]) -> bytes:  # type: ignore[override]
        return orjson.dumps(content, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("orion-cortex-orchestrator")

bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)

cfg = ChassisConfig(
    service_name="CortexOrchestrator",
    service_version=getattr(settings, "service_version", "0.1.0"),
    bus_url=settings.orion_bus_url,
    bus_enabled=settings.orion_bus_enabled,
)


async def _handle_orch_bus(req: CortexOrchBusRequest) -> CortexOrchBusReply:
    """Bus entrypoint: validates request, runs the verb, returns a stable reply."""
    try:
        orch_req = OrchestrateVerbRequest(
            verb_name=req.verb_name,
            origin_node=req.origin_node,
            context=req.context,
            steps=req.steps or [],
            timeout_ms=req.timeout_ms,
        )
        resp = await run_cortex_verb(bus, orch_req)
        return CortexOrchBusReply(trace_id=req.trace_id, ok=True, data=resp.model_dump(mode="json"))
    except Exception as e:
        # Chassis will also publish system.error, but we must reply so RPC callers don't hang.
        logger.exception("Bus orchestration failed (trace_id=%s)", req.trace_id)
        return CortexOrchBusReply(trace_id=req.trace_id, ok=False, error=str(e), data={})


orch_bus = Rabbit[
    CortexOrchBusRequest,
    CortexOrchBusReply,
](
    cfg,
    intake_channel=settings.cortex_orch_request_channel,
    request_model=CortexOrchBusRequest,
    handler=_handle_orch_bus,
    bus=bus,
)


app = FastAPI(
    title="Orion Cortex Orchestrator",
    version=cfg.service_version,
    default_response_class=ORJSONResponse,
)

_orch_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def _startup() -> None:
    global _orch_task

    logger.info(
        "Cortex Orchestrator starting; HTTP on %s:%s, bus_channel=%s",
        settings.api_host,
        settings.api_port,
        settings.cortex_orch_request_channel,
    )

    _orch_task = asyncio.create_task(orch_bus.run(), name="cortex-orch.rabbit")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _orch_task

    with suppress(Exception):
        await orch_bus.stop()

    if _orch_task:
        _orch_task.cancel()
        with suppress(asyncio.CancelledError):
            await _orch_task
        _orch_task = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "node_name": settings.node_name,
        "bus_enabled": bus.enabled,
        "bus_url": settings.orion_bus_url,
        "exec_request_prefix": settings.exec_request_prefix,
        "exec_result_prefix": settings.exec_result_prefix,
        "cortex_orch_request_channel": settings.cortex_orch_request_channel,
        "cortex_orch_result_prefix": settings.cortex_orch_result_prefix,
    }


@app.post("/orchestrate", response_model=OrchestrateVerbResponse)
async def orchestrate(req: OrchestrateVerbRequest) -> OrchestrateVerbResponse:
    logger.info(
        "HTTP orchestrate verb=%s origin=%s steps=%d",
        req.verb_name,
        req.origin_node,
        len(req.steps) if req.steps else 0,
    )
    return await run_cortex_verb(bus, req)
