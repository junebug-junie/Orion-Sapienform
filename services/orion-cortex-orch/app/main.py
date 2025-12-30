# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging
from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.core.bus.contracts import KINDS

from .orchestrator import call_cortex_exec
from .models import CortexOrchInput  # CHANGED: Import from models
from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch")


class CortexOrchRequest(BaseEnvelope):
    kind: str = Field(KINDS.cortex_orch_request, frozen=True)
    payload: CortexOrchInput


class CortexOrchResultPayload(BaseModel):
    ok: bool = True
    result: dict


class CortexOrchResult(BaseEnvelope):
    kind: str = Field(KINDS.cortex_orch_result, frozen=True)
    payload: CortexOrchResultPayload


def _cfg() -> ChassisConfig:
    s = get_settings()
    return ChassisConfig(
        service_name=s.service_name,
        service_version=s.service_version,
        node_name=s.node_name,
        bus_url=s.orion_bus_url,
        bus_enabled=s.orion_bus_enabled,
        heartbeat_interval_sec=float(s.heartbeat_interval_sec or 10.0),
    )


def _source() -> ServiceRef:
    s = get_settings()
    return ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name)


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    sref = _source()
    logger.info(f"Handling Orch Request kind={env.kind} corr_id={env.correlation_id}")

    # 1. Ingress Validation
    try:
        raw_payload = env.payload if isinstance(env.payload, dict) else {}
        
        if env.kind == "cortex.orch.request":
            req_env = CortexOrchRequest.model_validate(env.model_dump(mode="json"))
            inp = req_env.payload
        else:
            # Fallback for raw sends
            inp = CortexOrchInput.model_validate(raw_payload)

    except ValidationError as ve:
        logger.warning(f"Validation failed: {ve}")
        return BaseEnvelope(
            kind="cortex.orch.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    # 2. Context Extraction (Still needed due to payload nesting variations)
    final_context = raw_payload.get("context") or inp.context or {}
    logger.debug(f"Context loaded with {len(final_context.get('messages', []))} messages.")

    # 3. Execution
    try:
        s = get_settings()
        
        result_payload = await call_cortex_exec(
            svc.bus,
            source=sref,
            exec_request_channel=s.channel_exec_request,
            verb_name=inp.verb_name,
            args=inp.args,
            context=final_context,
            correlation_id=str(env.correlation_id),
            timeout_sec=float(inp.args.get("timeout_sec", 900.0)),
        )
        
        ok_flag = bool(result_payload.get("ok", True)) if isinstance(result_payload, dict) else True
        
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexOrchResultPayload(ok=ok_flag, result=result_payload),
        )

    except Exception as e:
        logger.exception(f"Execution failed for verb {inp.verb_name}")
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexOrchResultPayload(
                ok=False, 
                result={"error": str(e), "type": type(e).__name__}
            ),
        )


svc = Rabbit(_cfg(), request_channel=get_settings().channel_cortex_request, handler=handle)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Cortex Orch Service (Typed Client Version)")
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
