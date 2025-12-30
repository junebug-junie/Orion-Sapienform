# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging
import traceback

from pydantic import Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from .orchestrator import call_cortex_exec
from .settings import get_settings
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.cortex.schemas import StepExecutionResult

logger = logging.getLogger("orion.cortex.orch")


class CortexOrchRequest(BaseEnvelope):
    kind: str = Field("cortex.orch.request", frozen=True)
    payload: CortexClientRequest


class CortexOrchResult(BaseEnvelope):
    kind: str = Field("cortex.orch.result", frozen=True)
    payload: CortexClientResult


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
        req = CortexClientRequest.model_validate(raw_payload)
    except ValidationError as ve:
        logger.warning(f"Validation failed: {ve}")
        return BaseEnvelope(
            kind="cortex.orch.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexClientResult(
                ok=False,
                mode=raw_payload.get("mode") or "brain",
                verb=raw_payload.get("verb") or raw_payload.get("verb_name") or "unknown",
                status="fail",
                memory_used=False,
                recall_debug={},
                steps=[],
                error={"message": "validation_failed", "details": ve.errors()},
                correlation_id=str(env.correlation_id),
                final_text=None,
            ),
        )

    # 3. Execution
    try:
        s = get_settings()

        result_payload = await call_cortex_exec(
            svc.bus,
            source=sref,
            exec_request_channel=s.channel_exec_request,
            exec_result_prefix=s.channel_exec_result_prefix,
            client_request=req,
            correlation_id=str(env.correlation_id),
            timeout_sec=float(req.options.get("timeout_sec", 900.0)),
        )

        steps = [
            StepExecutionResult.model_validate(s) if not isinstance(s, StepExecutionResult) else s
            for s in (result_payload.get("steps") or [])
        ]

        error_payload = result_payload.get("error")
        if isinstance(error_payload, str):
            error_payload = {"message": error_payload}

        client_result = CortexClientResult(
            ok=(result_payload.get("status") == "success"),
            mode=req.mode,
            verb=req.verb,
            status=result_payload.get("status") or "fail",
            final_text=result_payload.get("final_text"),
            memory_used=bool(result_payload.get("memory_used")),
            recall_debug=result_payload.get("recall_debug") or {},
            steps=steps,
            error=error_payload,
            correlation_id=str(env.correlation_id),
        )

        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=client_result,
        )

    except Exception as e:
        logger.exception(f"Execution failed for verb {getattr(req, 'verb', 'unknown')}")
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexClientResult(
                ok=False,
                mode=getattr(req, "mode", "brain"),
                verb=getattr(req, "verb", "unknown"),
                status="fail",
                final_text=None,
                memory_used=False,
                recall_debug={},
                steps=[],
                error={"message": str(e), "type": type(e).__name__},
                correlation_id=str(env.correlation_id),
            ),
        )


svc = Rabbit(_cfg(), request_channel=get_settings().channel_cortex_request, handler=handle)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Cortex Orch Service (Typed Client Version)")
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
