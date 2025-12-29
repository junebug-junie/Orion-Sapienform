# services/orion-cortex-exec/app/bus_main.py
"""Bus-native entrypoint for cortex-exec.

This enables the canonical RPC path:
  hub -> cortex-orch (Rabbit) -> cortex-exec (Rabbit) -> reply

Containers should run this module by default.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from .models import PlanExecutionRequest
from .router import PlanRouter  # CHANGED: Was PlanRunner
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.bus")


class CortexExecRequest(BaseEnvelope):
    kind: str = Field("cortex.exec.request", frozen=True)
    payload: PlanExecutionRequest


class CortexExecResultPayload(BaseModel):
    ok: bool = True
    result: Dict[str, Any] = Field(default_factory=dict)


class CortexExecResult(BaseEnvelope):
    kind: str = Field("cortex.exec.result", frozen=True)
    payload: CortexExecResultPayload


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(settings.heartbeat_interval_sec or 10.0),
    )


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
    )


# CHANGED: Match main.py class usage
router = PlanRouter()
svc: Rabbit | None = None


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    # [DEBUG] RAW LOGGING OF INCOMING DATA (Ported from main.py)
    raw_payload = env.payload if isinstance(env.payload, dict) else {}
    has_context_key = "context" in raw_payload
    
    logger.info(f"[EXEC-BUS] Incoming correlation_id={env.correlation_id}")
    logger.debug(f"[EXEC-BUS] Raw Payload Keys: {list(raw_payload.keys())}")
    logger.debug(f"[EXEC-BUS] Has 'context'? {has_context_key}")

    try:
        req_env = CortexExecRequest.model_validate(env.model_dump(mode="json"))
    except ValidationError as ve:
        logger.error(f"Validation failed: {ve}")
        return BaseEnvelope(
            kind="cortex.exec.result",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    # [CRITICAL FIX] Ported from main.py
    # Manually extract context from raw_payload if Pydantic lost it
    payload_context = raw_payload.get("context") or req_env.payload.context or {}

    # Build the execution ctx passed down into step services
    ctx = {
        **payload_context,
        **(req_env.payload.args.extra or {}),
        "user_id": req_env.payload.args.user_id,
        "trigger_source": req_env.payload.args.trigger_source,
    }

    assert svc is not None, "Rabbit service not initialized"
    
    # CHANGED: runner -> router
    res = await router.run_plan(
        svc.bus,
        source=_source(),
        req=req_env.payload,
        correlation_id=str(env.correlation_id),
        ctx=ctx,
    )

    return CortexExecResult(
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=CortexExecResultPayload(ok=True, result=res.model_dump(mode="json")),
    )


svc = Rabbit(_cfg(), request_channel=settings.channel_exec_request, handler=handle)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Starting cortex-exec bus listener channel=%s bus=%s",
        settings.channel_exec_request,
        settings.orion_bus_url,
    )
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
