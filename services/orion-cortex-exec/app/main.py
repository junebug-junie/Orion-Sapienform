"""
Canonical Entrypoint for Cortex Exec.
This handles the RabbitMQ/Bus connection and routes requests to the PlanRouter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

# IMPORTS UPDATED: Added Envelope for generic typing
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from .router import PlanRouter
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.main")


class CortexExecRequest(BaseEnvelope):
    kind: str = Field("cortex.exec.request", frozen=True)
    payload: PlanExecutionRequest


class CortexExecResultPayload(BaseModel):
    ok: bool = True
    result: Dict[str, Any] = Field(default_factory=dict)


class CortexExecResult(BaseEnvelope):
    kind: str = Field("cortex.exec.result", frozen=True)
    payload: CortexExecResultPayload


class CognitionTraceEnvelope(Envelope[CognitionTracePayload]):
    """
    Typed contract for cognition traces aligning to Titanium Envelope[T].
    This ensures the payload is validated as a CognitionTracePayload model,
    not forced into a dict before validation.
    """
    kind: str = Field("cognition.trace", frozen=True)


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(settings.heartbeat_interval_sec),
    )


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
    )


router = PlanRouter()
svc: Rabbit | None = None


def _diagnostic_enabled(payload: PlanExecutionRequest) -> bool:
    try:
        extra = payload.args.extra or {}
        options = extra.get("options") if isinstance(extra, dict) else {}
        return bool(
            settings.diagnostic_mode
            or extra.get("diagnostic")
            or (isinstance(options, dict) and (options.get("diagnostic") or options.get("diagnostic_mode")))
        )
    except Exception:
        return settings.diagnostic_mode


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    corr_id = str(env.correlation_id)
    logger.info(f"Incoming Exec Request: correlation_id={corr_id}")

    try:
        req_env = CortexExecRequest.model_validate(env.model_dump(mode="json"))
    except ValidationError as ve:
        logger.error(f"Validation failed: {ve}")
        return BaseEnvelope(
            kind="cortex.exec.result",
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    # 1. Extract Context (Handling the Pydantic stripping issue)
    raw_payload = env.payload if isinstance(env.payload, dict) else {}
    payload_context = raw_payload.get("context") or req_env.payload.context or {}

    # 2. Merge Context
    ctx = {
        **payload_context,
        **(req_env.payload.args.extra or {}),
        "user_id": req_env.payload.args.user_id,
        "trigger_source": req_env.payload.args.trigger_source,
    }

    logger.debug(f"Context loaded with {len(ctx.get('messages', []))} history messages.")

    assert svc is not None, "Rabbit service not initialized"

    diagnostic = _diagnostic_enabled(req_env.payload)
    if diagnostic:
        logger.info("Diagnostic PlanExecutionRequest json=%s", req_env.payload.model_dump_json())
        logger.info("Diagnostic args.extra snapshot corr=%s payload=%s", env.correlation_id, req_env.payload.args.extra)
        ctx["diagnostic"] = True

    # 3. Execute Plan
    res = await router.run_plan(
        svc.bus,
        source=_source(),
        req=req_env.payload,
        correlation_id=corr_id,
        ctx=ctx,
    )

    # 4. Publish Cognition Trace
    try:
        # Attempt to extract 'packs' from args if present (typically passed in extra for agents)
        packs_used = req_env.payload.args.extra.get("packs") or []
        if isinstance(packs_used, str):
            packs_used = [packs_used]

        trace_payload = CognitionTracePayload(
            correlation_id=corr_id,
            mode=res.mode or "brain",
            verb=res.verb_name,
            packs=packs_used if isinstance(packs_used, list) else [],
            options=req_env.payload.args.extra.get("options", {}) if req_env.payload.args.extra else {},
            final_text=res.final_text,
            steps=res.steps,
            timestamp=time.time(),
            source_service=settings.service_name,
            source_node=settings.node_name,
            recall_used=res.memory_used,
            recall_debug=res.recall_debug,
            metadata={
                "request_id": res.request_id,
                "status": res.status,
            }
        )

        # UPDATED: Use the typed envelope
        trace_envelope = CognitionTraceEnvelope(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain, # Propagate causality
            payload=trace_payload
        )

        await svc.bus.publish(settings.channel_cognition_trace_pub, trace_envelope)
        logger.info(f"Published CognitionTrace to {settings.channel_cognition_trace_pub}")

    except Exception as e:
        logger.error(f"Failed to publish CognitionTrace: {e}", exc_info=True)

    return CortexExecResult(
        source=_source(),
        correlation_id=corr_id,
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
