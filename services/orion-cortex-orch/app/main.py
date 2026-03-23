# services/orion-cortex-orch/app/main.py
from __future__ import annotations

import asyncio
import logging
import traceback
import json

from pydantic import Field, ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit, Hunter
from orion.normalizers.agent_trace import build_agent_trace_summary

# REMOVED: dispatch_metacognition_tick
from .orchestrator import call_verb_runtime, dispatch_dream_trigger, dispatch_metacog_trigger
from .settings import get_settings
from .decision_router import DecisionRouter
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.cortex.schemas import StepExecutionResult
from orion.cognition.verb_activation import is_active, is_runtime_entry_verb

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




def _normalize_and_validate_verb(req: CortexClientRequest) -> tuple[bool, str | None]:
    mode = (req.mode or "brain").lower()
    verb = (req.verb or "").strip()

    if mode == "auto":
        req.verb = None
        return True, None

    if not verb:
        if mode == "brain":
            req.verb = "chat_general"
            verb = req.verb
        else:
            req.verb = None
            return True, None

    if mode in {"agent", "council"} and is_runtime_entry_verb(verb):
        req.verb = verb
        return True, None

    if not is_active(verb, node_name=get_settings().node_name):
        return False, verb

    req.verb = verb
    return True, None

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




def _should_auto_route(req: CortexClientRequest, env: BaseEnvelope) -> tuple[bool, str]:
    options = req.options if isinstance(req.options, dict) else {}
    route_intent = str(options.get("route_intent") or req.route_intent or "none").lower()
    requested = route_intent == "auto" or str(req.mode).lower() == "auto"
    source_name = ((env.source.name if env.source else "") or "").strip().lower()
    allowlisted = source_name in {"cortex-gateway"}

    if not requested:
        return False, "intent_none"
    if not allowlisted:
        return False, f"source_not_allowlisted:{source_name or 'unknown'}"
    return True, "intent_auto_allowlisted"
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
    route_meta: dict | None = None
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
            "orch_intake corr=%s source=%s reply=%s mode=%s verb=%s supervised=%s recall_enabled=%s recall_profile=%s packs=%s",
            str(env.correlation_id),
            (env.source.name if env.source else "unknown"),
            env.reply_to,
            req.mode,
            req.verb,
            bool((req.options or {}).get("supervised")),
            req.recall.enabled,
            req.recall.profile,
            req.packs,
        )
        if diagnostic:
            logger.info("Diagnostic CortexClientRequest json=%s", req.model_dump_json())

        should_route, route_reason = _should_auto_route(req, env)
        logger.info(
            "auto_depth_gate corr_id=%s should_auto=%s reason=%s source=%s mode=%s route_intent=%s",
            str(env.correlation_id),
            should_route,
            route_reason,
            ((env.source.name if env.source else "") or "unknown"),
            req.mode,
            (req.options.get("route_intent") if isinstance(req.options, dict) else None) or req.route_intent,
        )

        if should_route:
            router = DecisionRouter(svc.bus)
            routed = await router.route(req, correlation_id=str(env.correlation_id), source=sref)
            req = routed.request
            route_meta = routed.decision.model_dump(mode="json")
            route_meta["output_mode_decision"] = routed.output_mode_decision.model_dump()
            logger.info(
                "auto_depth_result corr_id=%s depth=%s primary_verb=%s router_source=%s confidence=%.2f",
                str(env.correlation_id),
                routed.decision.execution_depth,
                routed.decision.primary_verb,
                routed.decision.source,
                routed.decision.confidence,
            )
        elif str(req.mode).lower() == "auto":
            logger.info("auto_route_gate corr_id=%s fallback_mode=brain reason=%s", str(env.correlation_id), route_reason)
            req.mode = "brain"

        ok, bad_verb = _normalize_and_validate_verb(req)
        if not ok:
            failure = CortexClientResult(
                ok=False,
                mode=req.mode,
                verb=bad_verb or "unknown",
                status="fail",
                memory_used=False,
                recall_debug={},
                steps=[],
                error={"message": f"inactive_verb:{bad_verb}", "verb": bad_verb, "node": get_settings().node_name},
                correlation_id=str(env.correlation_id),
                final_text=f"Verb '{bad_verb}' is inactive on node {get_settings().node_name}.",
            )
            return CortexOrchResult(
                source=sref,
                correlation_id=env.correlation_id,
                causality_chain=env.causality_chain,
                payload=failure.model_dump(mode="json"),
            )

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
            router_metadata=route_meta,
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
        terminal_skip = {"planner_react", "council_checkpoint", "agent_chain", "introspect_spark"}
        for s in steps:
            if s.verb_name and s.verb_name not in terminal_skip:
                executed_verbs.append(s.verb_name)
        
        # Determine the "primary" verb (the most specific one executed)
        primary_verb = executed_verbs[-1] if executed_verbs else req.verb

        # Merge into metadata
        final_meta = result_payload.get("metadata") or result_payload.get("spark_meta") or {}
        if isinstance(final_meta, dict):
            final_meta["trace_verb"] = primary_verb
            final_meta["executed_verbs"] = executed_verbs
            if route_meta:
                final_meta["auto_route"] = route_meta

        # Answer-depth: surface agent-chain runtime_debug for live proof evidence
        for s in steps:
            if not isinstance(s.result, dict):
                continue
            ac = s.result.get("AgentChainService")
            if isinstance(ac, dict) and ac.get("runtime_debug"):
                rd = ac["runtime_debug"]
                if isinstance(final_meta, dict):
                    final_meta["answer_depth"] = {
                        "output_mode": rd.get("output_mode"),
                        "response_profile": rd.get("response_profile"),
                        "packs": rd.get("packs"),
                        "resolved_tool_ids": rd.get("resolved_tool_ids"),
                        "triage_blocked_post_step0": rd.get("triage_blocked_post_step0"),
                        "repeated_plan_action_escalation": rd.get("repeated_plan_action_escalation"),
                        "finalize_response_invoked": rd.get("finalize_response_invoked"),
                        "quality_evaluator_rewrite": rd.get("quality_evaluator_rewrite"),
                    }
                break

        agent_trace = build_agent_trace_summary(
            correlation_id=str(env.correlation_id),
            message_id=str(env.correlation_id),
            mode=req.mode,
            status=result_payload.get("status") or "fail",
            final_text=result_payload.get("final_text"),
            steps=steps,
            metadata=final_meta if isinstance(final_meta, dict) else {},
        )
        if isinstance(final_meta, dict) and agent_trace is not None:
            final_meta["agent_trace_available"] = True

        client_result = CortexClientResult(
            ok=(result_payload.get("status") == "success" and verb_result.ok),
            mode=req.mode,
            verb=req.verb or "unknown",
            status=result_payload.get("status") or "fail",
            final_text=result_payload.get("final_text"),
            memory_used=bool(result_payload.get("memory_used")),
            recall_debug=result_payload.get("recall_debug") or {},
            steps=steps,
            error=error_payload,
            correlation_id=str(env.correlation_id),
            agent_trace=agent_trace,
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
dream_hunter: Hunter


async def _handle_dream_envelope(env: BaseEnvelope) -> None:
    if env.kind != "dream.trigger":
        return
    trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
    logger.info("Dream trigger intake kind=%s trace_id=%s", env.kind, trace_id)
    await dispatch_dream_trigger(dream_hunter.bus, source=_source(), env=env)


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

dream_hunter = Hunter(
    _cfg(),
    handler=_handle_dream_envelope,
    patterns=[get_settings().channel_dream_trigger],
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
    await asyncio.gather(svc.start(), equilibrium_hunter.start(), dream_hunter.start())


if __name__ == "__main__":
    asyncio.run(main())
