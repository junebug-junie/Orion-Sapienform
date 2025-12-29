# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging
import traceback
import sys

from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from .orchestrator import CortexOrchInput, call_cortex_exec
from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch")

# [DEBUG] Force stdout flushing
def debug_print(msg):
    print(f"[ORCH-DEBUG] {msg}", file=sys.stdout, flush=True)

class CortexOrchRequest(BaseEnvelope):
    kind: str = Field("cortex.orch.request", frozen=True)
    payload: CortexOrchInput


class CortexOrchResultPayload(BaseModel):
    ok: bool = True
    result: dict


class CortexOrchResult(BaseEnvelope):
    kind: str = Field("cortex.orch.result", frozen=True)
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
    debug_print(f"Handle entered for kind={env.kind}")
    sref = _source()

    try:
        if env.kind == "cortex.orch.request":
            req_env = CortexOrchRequest.model_validate(env.model_dump(mode="json"))
            inp = req_env.payload
        else:
            inp = CortexOrchInput.model_validate(env.payload if isinstance(env.payload, dict) else {})
        debug_print(f"Validation successful. Verb={inp.verb_name}")
    except ValidationError as ve:
        logger.warning("Invalid cortex-orch request: %s", ve)
        return BaseEnvelope(
            kind="cortex.orch.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    try:
        s = get_settings()
        debug_print(f"Calling call_cortex_exec on channel={s.channel_exec_request}")
        
        # [DEBUG] Check bus status
        if not svc.bus.redis:
            debug_print("CRITICAL: Bus redis connection is missing!")

        result_payload = await call_cortex_exec(
            svc.bus,
            source=sref,
            exec_request_channel=s.channel_exec_request,
            verb_name=inp.verb_name,
            args=inp.args,
            context=inp.context,
            correlation_id=str(env.correlation_id),
            timeout_sec=float(inp.args.get("timeout_sec", 900.0)),
        )
        debug_print("call_cortex_exec returned")
        ok_flag = bool(result_payload.get("ok", True)) if isinstance(result_payload, dict) else True
        
    except Exception as e:
        debug_print(f"EXCEPTION in handle: {e}")
        traceback.print_exc()
        logger.error(f"Execution failed for verb {inp.verb_name}: {e}")
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexOrchResultPayload(
                ok=False, 
                result={
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            ),
        )

    debug_print("Returning success envelope")
    return CortexOrchResult(
        source=sref,
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=CortexOrchResultPayload(ok=ok_flag, result=result_payload),
    )


cfg = _cfg()
svc = Rabbit(cfg, request_channel=get_settings().channel_cortex_request, handler=handle)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    debug_print("Starting Cortex Orch Main...")
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
