# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging
import traceback
import json

from pydantic import Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit, Hunter

# REMOVED: dispatch_metacognition_tick
from .orchestrator import call_verb_runtime, dispatch_metacog_trigger
from .settings import get_settings
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.cortex.schemas import StepExecutionResult

logger = logging.getLogger("orion.cortex.orch")

# Channels
# We only listen for the trigger now. The SQL Writer handles the ticks independently.


class CortexOrchRequest(BaseEnvelope):
    kind: str = Field("cortex.orch.request", frozen=True)
    payload: CortexClientRequest


class CortexOrchResult(BaseEnvelope):
    kind: str = Field("cortex.orch.result", frozen=True)
    payload: CortexClientResult


def _is_diagnostic(raw_payload: dict | None) -> bool:
    if get_settings().diagnostic_mode:
        return True
    opts = (raw_payload or {}).get("options") if isinstance(raw_payload, dict) else {}
    return bool(isinstance(opts, dict) and (opts.get("diagnostic") or opts.get("diagnostic_mode")))


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

    if env.kind not in ("cortex.orch.request", "legacy.message"):
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=CortexClientResult(
                ok=False,
                mode="brain",
                verb="unknown",
                status="fail",
                final_text=None,
                memory_used=False,
                recall_debug={},
                steps=[],
                error={"message": f"unsupported_kind:{env.kind}"},
                correlation_id=str(env.correlation_id),
            ).model_dump(mode="json"),
        )

    # 1. Ingress Validation
    try:
        raw_payload = env.payload if isinstance(env.payload, dict) else {}
        diagnostic = _is_diagnostic(raw_payload)

        if diagnostic:
            logger.info(
                "Diagnostic ingress payload (raw) corr=%s json=%s",
                str(env.correlation_id),
                json.dumps(raw_payload, default=str),
            )
        req = CortexClientRequest.model_validate(raw_payload)

        logger.info(
            "Validated orch request: corr=%s mode=%s verb=%s packs=%s recall_enabled=%s recall_required=%s",
            str(env.correlation_id),
            req.mode,
            req.verb,
            req.packs,
            req.recall.enabled,
            req.recall.required,
        )
        if diagnostic:
            logger.info("Diagnostic CortexClientRequest json=%s", req.model_dump_json())

    except ValidationError as ve:
        trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
        logger.warning("Validation failed trace_id=%s error=%s", trace_id, ve)
        failure = CortexClientResult(
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
        )
        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=failure.model_dump(mode="json"),
         )

    # 3. Execution
    try:
        verb_result = await call_verb_runtime(
            svc.bus,
            source=sref,
            client_request=req,
            correlation_id=str(env.correlation_id),
            causality_chain=env.causality_chain,
            trace=env.trace,
            timeout_sec=float(req.options.get("timeout_sec", 900.0)),
        )

        result_payload = verb_result.output if isinstance(verb_result.output, dict) else {}
        if "result" in result_payload and isinstance(result_payload.get("result"), dict):
            result_payload = result_payload["result"]
        steps = [
            StepExecutionResult.model_validate(s) if not isinstance(s, StepExecutionResult) else s
            for s in (result_payload.get("steps") or [])
        ]

        error_payload = result_payload.get("error")
        if isinstance(error_payload, str):
            error_payload = {"message": error_payload}

        if not verb_result.ok:
            error_payload = error_payload or {"message": verb_result.error or "verb_failed"}

        # ENRICHMENT: Extract executed verbs from steps to populate metadata
        executed_verbs = []
        for s in steps:
            if s.verb_name and s.verb_name not in ["planner_react", "council_checkpoint", "agent_chain"]:
                executed_verbs.append(s.verb_name)
        
        # Determine the "primary" verb (the most specific one executed)
        primary_verb = executed_verbs[-1] if executed_verbs else req.verb

        # Merge into metadata
        final_meta = result_payload.get("metadata") or result_payload.get("spark_meta") or {}
        if isinstance(final_meta, dict):
            final_meta["trace_verb"] = primary_verb
            final_meta["executed_verbs"] = executed_verbs

        client_result = CortexClientResult(
            ok=(result_payload.get("status") == "success" and verb_result.ok),
            mode=req.mode,
            verb=req.verb,
            status=result_payload.get("status") or "fail",
            final_text=result_payload.get("final_text"),
            memory_used=bool(result_payload.get("memory_used")),
            recall_debug=result_payload.get("recall_debug") or {},
            steps=steps,
            error=error_payload,
            correlation_id=str(env.correlation_id),
            metadata=final_meta,
        )

        return CortexOrchResult(
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=client_result.model_dump(mode="json"),
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
            ).model_dump(mode="json"),
        )

svc = Rabbit(_cfg(), request_channel=get_settings().channel_cortex_request, handler=handle)
equilibrium_hunter: Hunter


async def _handle_equilibrium_envelope(env: BaseEnvelope) -> None:
    """
    Routes events from Equilibrium.
    Now only handles Triggers (for Collapse Mirror).
    Ticks are ignored here (logged directly by SQL Writer).
    """
    if env.kind == "orion.metacog.trigger.v1":
        trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
        logger.info("Metacog trigger intake kind=%s trace_id=%s", env.kind, trace_id)
        await dispatch_metacog_trigger(
            equilibrium_hunter.bus,
            source=_source(),
            env=env,
        )
        return

    # Pass on any legacy kinds or ticks
    pass

equilibrium_hunter = Hunter(
    _cfg(),
    handler=_handle_equilibrium_envelope,
    patterns=[
        get_settings().channel_metacog_trigger,
    ],
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    logger.info(
        "Starting Cortex Orch Service (Typed Client Version) "
        f"intake={s.channel_cortex_request} "
        f"exec_channel={s.channel_exec_request} "
        f"bus={s.orion_bus_url}"
    )
    await asyncio.gather(svc.start(), equilibrium_hunter.start())


if __name__ == "__main__":
    asyncio.run(main())
