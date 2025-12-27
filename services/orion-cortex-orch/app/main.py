from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import FastAPI

# uvicorn loads `app.main:app` — define it immediately.
app = FastAPI(title="Orion Cortex Orchestrator", version="0.0.0")
logger = logging.getLogger("orion-cortex-orchestrator")


@app.on_event("startup")
async def _startup() -> None:
    """Start the Rabbit bus worker in the background.

    Imports are delayed so the ASGI attribute `app` always exists even if
    downstream imports break; this prevents the misleading "attribute app not found"
    uvicorn error.
    """

    from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
    from orion.core.bus.service_async import OrionBusAsync

    from .bus_models import CortexOrchBusReplyV1, CortexOrchBusRequestV1
    from .orchestrator import OrchestrateVerbRequest, run_cortex_verb
    from .settings import settings

    # Logging once settings are available
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app.version = getattr(settings, "service_version", "0.1.0")

    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)

    cfg = ChassisConfig(
        service_name="CortexOrchestrator",
        service_version=app.version,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
    )

    async def handler(req: CortexOrchBusRequestV1) -> Dict[str, Any]:
        orch_req = OrchestrateVerbRequest(
            trace_id=req.trace_id,
            verb_name=req.verb_name,
            origin_node=req.origin_node,
            session_id=req.session_id,
            text=req.text,
            history=req.history,
            args=req.args,
            context=req.context,
            steps=req.steps,
            timeout_ms=req.timeout_ms,
        )

        resp = await run_cortex_verb(orch_req, bus=bus)

        data: Dict[str, Any] = {
            "trace_id": resp.trace_id,
            "ok": resp.ok,
            "verb_name": resp.verb_name,
            "exec_id": resp.exec_id,
            "elapsed_ms": resp.elapsed_ms,
            "text": resp.text,
            "step_results": [sr.model_dump(mode="json") for sr in resp.step_results],
        }

        reply = CortexOrchBusReplyV1(ok=resp.ok, data=data)
        return reply.model_dump(mode="json")

    rabbit = Rabbit[
        CortexOrchBusRequestV1,
        Dict[str, Any],
    ](
        cfg,
        intake_channel=settings.cortex_orch_request_channel,
        request_model=CortexOrchBusRequestV1,
        handler=handler,
        bus=bus,
    )

    app.state.orion_bus = bus
    app.state.rabbit = rabbit
    app.state.rabbit_task = asyncio.create_task(rabbit.run(), name="cortex-orch.rabbit")

    logger.info(
        "CortexOrch online (http + bus) request_channel=%s cortex_exec_channel=%s",
        settings.cortex_orch_request_channel,
        settings.cortex_exec_request_channel,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    rabbit = getattr(app.state, "rabbit", None)
    task: asyncio.Task | None = getattr(app.state, "rabbit_task", None)

    try:
        if rabbit is not None:
            await rabbit.stop()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        from .settings import settings

        return {
            "status": "ok",
            "service": "CortexOrchestrator",
            "version": getattr(settings, "service_version", "0.1.0"),
            "bus_url": settings.orion_bus_url,
            "orch_request_channel": settings.cortex_orch_request_channel,
            "cortex_exec_request_channel": settings.cortex_exec_request_channel,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
