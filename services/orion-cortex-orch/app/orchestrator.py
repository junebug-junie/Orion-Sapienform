# services/orion-cortex-orch/app/orchestrator.py
from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs import VerbRequestV1, VerbResultV1
from orion.cognition.plan_loader import build_plan_for_verb
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionRequest,
    PlanExecutionArgs
)
from .clients import CortexExecClient, StateServiceClient
from .settings import get_settings
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply
from orion.schemas.cortex.contracts import CortexClientRequest, RecallDirective
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.cortex.orch")

def build_agent_plan(verb_name: str | None) -> ExecutionPlan:
    """Two-step agent plan: planner-react followed by agent chain."""
    resolved_verb = verb_name or "agent_runtime"
    return ExecutionPlan(
        verb_name=resolved_verb,
        label=f"{resolved_verb}-agent",
        description="Agent chain execution via planner-react",
        category="agentic",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=resolved_verb,
                step_name="planner_react",
                description="Delegate planning to PlannerReactService",
                order=-1,
                services=["PlannerReactService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=120000,
            ),
            ExecutionStep(
                verb_name=resolved_verb,
                step_name="agent_chain",
                description="Delegate to AgentChainService (ReAct)",
                order=0,
                services=["AgentChainService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=300000,
            )
        ],
        metadata={"mode": "agent"},
    )


def build_council_plan(verb_name: str | None) -> ExecutionPlan:
    """Stub council plan; routed to CouncilService."""
    resolved_verb = verb_name or "council_runtime"
    return ExecutionPlan(
        verb_name=resolved_verb,
        label=f"{resolved_verb}-council",
        description="Council supervisor stub",
        category="council",
        priority="normal",
        interruptible=False,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=resolved_verb,
                step_name="council_supervisor",
                description="Council supervisor placeholder",
                order=0,
                services=["CouncilService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=120000,
            )
        ],
        metadata={"mode": "council"},
    )


def _build_plan_for_mode(req: CortexClientRequest) -> ExecutionPlan:
    if req.mode == "agent":
        return build_agent_plan(req.verb)
    if req.mode == "council":
        return build_council_plan(req.verb)
    return build_plan_for_verb(req.verb, mode=req.mode)


def _build_context(req: CortexClientRequest) -> Dict[str, Any]:
    return {
        "messages": [m.model_dump(mode="json") for m in req.context.messages],
        "raw_user_text": req.context.raw_user_text or None,
        "user_message": req.context.user_message or None,
        "session_id": req.context.session_id,
        "user_id": req.context.user_id,
        "trace_id": req.context.trace_id,
        "metadata": req.context.metadata,
        "packs": req.packs,
        "mode": req.mode,
        "diagnostic": _diagnostic_enabled(req),
    }


def _trace_meta(
    *,
    trace_id: str,
    event_id: str,
    parent_event_id: str | None,
    source_service: str,
) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "event_id": event_id,
        "parent_event_id": parent_event_id,
        "source_service": source_service,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def dispatch_equilibrium_snapshot(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    env: BaseEnvelope,
) -> None:
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        entry = CollapseMirrorEntryV2.model_validate(payload)
    except Exception as exc:
        logger.warning("Equilibrium snapshot payload invalid: %s", exc)
        return

    if str(entry.type or "").strip().lower() == "noop":
        logger.info("Equilibrium snapshot ignored (noop) event_id=%s", entry.event_id)
        return

    request_id = str(uuid4())
    verb_request = VerbRequestV1(
        trigger="orion.collapse.log",
        schema_id="CollapseMirrorEntryV2",
        payload=entry.model_dump(mode="json"),
        request_id=request_id,
        caller=source.name,
        meta={
            "origin": "equilibrium-service",
            "event_id": entry.event_id,
        },
    )

    envelope = BaseEnvelope(
        kind="verb.request",
        source=source,
        correlation_id=env.correlation_id,
        causality_chain=list(env.causality_chain),
        trace=dict(env.trace),
        payload=verb_request.model_dump(mode="json"),
    )

    await bus.publish("orion:verb:request", envelope)
    logger.info("Equilibrium snapshot routed to verb request event_id=%s", entry.event_id)


def _diagnostic_enabled(req: CortexClientRequest) -> bool:
    settings = get_settings()
    try:
        return bool(settings.diagnostic_mode or req.options.get("diagnostic") or req.options.get("diagnostic_mode"))
    except Exception:
        return settings.diagnostic_mode


def _plan_args(req: CortexClientRequest, correlation_id: str) -> PlanExecutionArgs:
    recall: RecallDirective = req.recall
    diagnostic = _diagnostic_enabled(req)
    return PlanExecutionArgs(
        request_id=req.context.trace_id or correlation_id,
        user_id=req.context.user_id,
        trigger_source="cortex-orch",
        extra={
            "mode": req.mode,
            "packs": req.packs,
            "recall": recall.model_dump(),
            "options": req.options,
            "trace_id": req.context.trace_id or correlation_id,
            "session_id": req.context.session_id,
            "verb": req.verb,
            "diagnostic": diagnostic,
        },
    )


async def _maybe_fetch_state(bus: OrionBusAsync, *, source: ServiceRef, correlation_id: str) -> StateLatestReply | None:
    """
    Best-effort fetch of the latest Orion state snapshot from orion-state-service.
    Never raises; returns None on failure.
    """
    settings = get_settings()
    if not settings.orion_state_enabled:
        return None

    try:
        client = StateServiceClient(
            bus,
            request_channel=settings.state_request_channel,
            result_prefix=settings.state_result_prefix,
        )
        node = (settings.state_node or "").strip() or None
        req = StateGetLatestRequest(scope=settings.state_scope, node=node)
        return await client.get_latest(
            source=source,
            req=req,
            correlation_id=correlation_id,
            timeout_sec=float(settings.state_timeout_sec),
        )
    except Exception as e:
        logger.warning(f"State service unavailable (corr={correlation_id}): {e}")
        return None


def build_plan_request(client_request: CortexClientRequest, correlation_id: str) -> PlanExecutionRequest:
    plan = _build_plan_for_mode(client_request)
    context = _build_context(client_request)

    # Attach latest Orion state (Spark) as a read-model artifact
    context.setdefault("metadata", {})["orion_state_pending"] = True

    args = _plan_args(client_request, correlation_id)
    return PlanExecutionRequest(plan=plan, args=args, context=context)


async def call_cortex_exec(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    exec_request_channel: str,
    exec_result_prefix: str,
    client_request: CortexClientRequest,
    correlation_id: str,
    timeout_sec: float = 900.0,
) -> Dict[str, Any]:
    request_object = build_plan_request(client_request, correlation_id)

    state_reply = await _maybe_fetch_state(bus, source=source, correlation_id=correlation_id)
    if state_reply is not None:
        request_object.context.setdefault("metadata", {})["orion_state"] = state_reply.model_dump(mode="json")
        request_object.context["metadata"].pop("orion_state_pending", None)
    else:
        request_object.context.setdefault("metadata", {})["orion_state"] = StateLatestReply(
            ok=False,
            status="missing",
            note="state_service_unavailable",
        ).model_dump(mode="json")
        request_object.context["metadata"].pop("orion_state_pending", None)

    diagnostic = _diagnostic_enabled(client_request)

    logger.info(
        "Dispatching exec plan",
        extra={
            "correlation_id": correlation_id,
            "mode": client_request.mode,
            "verb": client_request.verb,
            "step_count": len(request_object.plan.steps),
            "steps": [s.step_name for s in request_object.plan.steps],
        },
    )

    client = CortexExecClient(
        bus,
        request_channel=exec_request_channel,
        result_prefix=exec_result_prefix,
    )

    if diagnostic:
        logger.info("Diagnostic PlanExecutionRequest json=%s", request_object.model_dump_json())
    return await client.execute_plan(
        source=source,
        req=request_object,
        correlation_id=correlation_id,
        timeout_sec=timeout_sec,
        trace=None,
    )


def build_verb_request(
    *,
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None = None,
    trace: dict | None = None,
) -> tuple[VerbRequestV1, BaseEnvelope]:
    request_id = str(uuid4())
    verb_request = VerbRequestV1(
        trigger="legacy.plan",
        schema_id=plan_request.__class__.__name__,
        payload=plan_request.model_dump(mode="json"),
        request_id=request_id,
        caller=source.name,
        meta={
            "verb": client_request.verb,
            "mode": client_request.mode,
            "origin": "cortex-orch",
        },
    )

    envelope = BaseEnvelope(
        kind="verb.request",
        source=source,
        correlation_id=correlation_id,
        causality_chain=list(causality_chain or []),
        trace=dict(trace or {}),
        payload=verb_request.model_dump(mode="json"),
    )
    return verb_request, envelope


async def call_verb_runtime(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    client_request: CortexClientRequest,
    correlation_id: str,
    causality_chain: list | None = None,
    trace: dict | None = None,
    timeout_sec: float = 900.0,
) -> VerbResultV1:
    plan_request = build_plan_request(client_request, correlation_id)
    state_reply = await _maybe_fetch_state(bus, source=source, correlation_id=correlation_id)
    if state_reply is not None:
        plan_request.context.setdefault("metadata", {})["orion_state"] = state_reply.model_dump(mode="json")
        plan_request.context["metadata"].pop("orion_state_pending", None)
    else:
        plan_request.context.setdefault("metadata", {})["orion_state"] = StateLatestReply(
            ok=False,
            status="missing",
            note="state_service_unavailable",
        ).model_dump(mode="json")
        plan_request.context["metadata"].pop("orion_state_pending", None)

    verb_request, envelope = build_verb_request(
        client_request=client_request,
        plan_request=plan_request,
        source=source,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=trace,
    )

    async def _wait_for_result() -> VerbResultV1:
        async with bus.subscribe("orion:verb:result") as pubsub:
            await bus.publish("orion:verb:request", envelope)
            async for msg in bus.iter_messages(pubsub):
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok or decoded.envelope is None:
                    continue
                payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
                try:
                    result = VerbResultV1.model_validate(payload)
                except Exception:
                    continue
                if result.request_id == verb_request.request_id:
                    return result
        raise RuntimeError("Verb result subscription closed without a match.")

    try:
        return await asyncio.wait_for(_wait_for_result(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"RPC timeout waiting on orion:verb:result ({verb_request.request_id})") from exc


async def dispatch_metacog_trigger(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    env: BaseEnvelope,
) -> None:
    """
    Handles 'orion.metacog.trigger.v1' -> executes 'log_orion_metacognition'.
    """
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        trigger = MetacogTriggerV1.model_validate(payload)
    except Exception as exc:
        trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
        logger.warning("Metacog trigger invalid trace_id=%s error=%s", trace_id, exc)
        return

    # Use the envelope's correlation_id, since the trigger payload doesn't carry it
    # Pydantic might parse correlation_id as a UUID object, so forced cast to str()
    correlation_id = str(env.correlation_id) if env.correlation_id else str(uuid4())
    parent_event_id = (env.trace or {}).get("event_id") or str(env.id)
    trace_id = (env.trace or {}).get("trace_id") or correlation_id

    logger.info(
        "Received metacog trigger kind=%s pressure=%.2f trace_id=%s",
        trigger.trigger_kind,
        trigger.pressure,
        trace_id,
    )

    # 1. Load the "Big Mirror" Plan
    # This corresponds to your log_orion_metacognition.yaml
    verb_name = "log_orion_metacognition"
    plan = build_plan_for_verb(verb_name, mode="brain")

    # 2. Build the Request
    # We pass the trigger details in the context so the first step (ContextService) can read them
    recall_override: Dict[str, Any] = {}
    if trigger.recall_enabled is False:
        recall_override = {"recall": {"enabled": False}}
        logger.info("Metacog recall disabled for trigger %s", trigger.trigger_kind)

    req = PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source="equilibrium",
            extra={
                "trigger_kind": trigger.trigger_kind,
                "pressure": trigger.pressure,
                "trace_id": trace_id,
                "parent_event_id": parent_event_id,
                "trigger_correlation_id": correlation_id,
                "trigger_trace_id": trace_id,
                "trigger_source_service": env.source.name if env.source else None,
                "trigger_source_node": env.source.node if env.source else None,
                **recall_override,
            },
        ),
        context={
            "trigger": trigger.model_dump(mode="json"),
            "verb": verb_name,
            "trace_id": trace_id,
            "parent_event_id": parent_event_id,
            "trigger_correlation_id": correlation_id,
            "trigger_trace_id": trace_id,
        },
    )

    # 3. Execute via Cortex-Exec
    settings = get_settings()
    client = CortexExecClient(
        bus,
        request_channel=settings.channel_exec_request,
        result_prefix=settings.channel_exec_result_prefix,
    )

    rpc_timeout = float(plan.timeout_ms) / 1000.0

    await client.execute_plan(
        source=source,
        req=req,
        correlation_id=correlation_id,
        timeout_sec=rpc_timeout,
        trace=_trace_meta(
            trace_id=trace_id,
            event_id=str(uuid4()),
            parent_event_id=parent_event_id,
            source_service=source.name,
        ),
    )
    logger.info(
        "Dispatched log_orion_metacognition trace_id=%s parent_event_id=%s timeout=%.1fs",
        trace_id,
        parent_event_id,
        rpc_timeout,
    )
