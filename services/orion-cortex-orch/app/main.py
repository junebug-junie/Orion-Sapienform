# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from .orchestrator import CortexOrchInput, call_cortex_exec
from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch")


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
    sref = _source()

    # The codec may wrap legacy dict messages into an envelope with kind=legacy.message
    # and payload containing verb_name/args/reply_channel/correlation_id.
    try:
        if env.kind == "cortex.orch.request":
            req_env = CortexOrchRequest.model_validate(env.model_dump(mode="json"))
            inp = req_env.payload
        else:
            # legacy: validate against payload dict
            inp = CortexOrchInput.model_validate(env.payload if isinstance(env.payload, dict) else {})
    except ValidationError as ve:
        logger.warning("Invalid cortex-orch request: %s", ve)
        return BaseEnvelope(
            kind="cortex.orch.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    # Delegate execution to cortex-exec (cortex-orch is a wrapper).
    s = get_settings()
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

    ok_flag = bool(result_payload.get("ok", True)) if isinstance(result_payload, dict) else True

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
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
