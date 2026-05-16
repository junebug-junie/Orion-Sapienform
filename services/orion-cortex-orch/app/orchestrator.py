# services/orion-cortex-orch/app/orchestrator.py
from __future__ import annotations

import logging
import json
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.core.verbs import VerbRequestV1, VerbResultV1
from orion.cognition.plan_loader import build_plan_for_verb
from orion.cognition.projection_builder import build_cognitive_projection_for_context
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionRequest,
    PlanExecutionArgs
)
from .clients import CortexExecClient, StateServiceClient
from .mind_runtime import (
    _mind_enabled_exact,
    build_mind_run_request,
    call_orion_mind_http,
    fetch_substrate_telemetry_facet_for_mind,
    merge_mind_brief_into_plan_metadata,
    publish_mind_run_artifact,
)
from .execution_lanes import ExecutionLaneDecision, resolve_execution_lane
from .settings import get_settings
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective
from orion.schemas.telemetry.dream import DreamInternalTriggerV1, DreamTriggerPayload
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

from orion.cognition.output_mode_classifier import classify_output_mode
from orion.cognition.delivery_grounding import build_delivery_grounding_context
from orion.cognition.answer_contract_normalize import investigation_state_for_contract
from orion.schemas.cognition.answer_contract import AnswerContract

logger = logging.getLogger("orion.cortex.orch")

_DIRECT_VERB_TRIGGERS = {
    "actions.respond_to_juniper_collapse_mirror.v1",
    "skills.system.time_now.v1",
    "skills.gpu.nvidia_smi_snapshot.v1",
    "skills.docker.ps_status.v1",
    "skills.biometrics.snapshot.v1",
    "skills.biometrics.raw_recent.v1",
    "skills.landing_pad.metrics_snapshot.v1",
    "skills.landing_pad.last_events.v1",
    "skills.system.notify_chat_message.v1",
    "skills.chat.discussion_window.v1",
}

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
                description="Execute tool-using agent chain",
                order=0,
                services=["AgentChainService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=300000,
            ),
        ],
        metadata={"mode": "agent"},
    )


def build_council_plan(verb_name: str | None) -> ExecutionPlan:
    resolved_verb = verb_name or "council_runtime"
    return ExecutionPlan(
        verb_name=resolved_verb,
        label=f"{resolved_verb}-council",
        description="Council deliberation execution",
        category="council",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=180000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=resolved_verb,
                step_name="council_deliberate",
                description="Deliberate via CouncilService",
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
    options = req.options if isinstance(req.options, dict) else {}
    supervised = bool(options.get("supervised"))
    force_agent_chain = bool(options.get("force_agent_chain"))
    output_mode_decision = options.get("output_mode_decision") if isinstance(options.get("output_mode_decision"), dict) else None
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
            "supervised": supervised,
            "force_agent_chain": force_agent_chain,
            "output_mode_decision": output_mode_decision,
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


def _user_text_for_classifier(req: CortexClientRequest) -> str:
    """Extract user text for output mode classification."""
    raw = req.context.raw_user_text or req.context.user_message or ""
    if raw:
        return str(raw).strip()
    for m in reversed(req.context.messages or []):
        msg = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else {})
        role = msg.get("role", "")
        if str(role).lower() == "user":
            content = msg.get("content") or msg.get("text") or ""
            if content:
                return str(content)[:10000]
    return ""


def build_plan_request(
    client_request: CortexClientRequest,
    correlation_id: str,
    *,
    router_metadata: dict[str, Any] | None = None,
) -> PlanExecutionRequest:
    plan = _build_plan_for_mode(client_request)
    context = _build_context(client_request)

    # Output mode classification (from options when auto-routed, else classify here)
    options = client_request.options if isinstance(client_request.options, dict) else {}
    output_mode = options.get("output_mode")
    response_profile = options.get("response_profile")
    output_mode_decision = options.get("output_mode_decision")
    if not output_mode or not response_profile:
        omd = classify_output_mode(_user_text_for_classifier(client_request))
        output_mode = output_mode or omd.output_mode
        response_profile = response_profile or omd.response_profile
        output_mode_decision = output_mode_decision or omd.model_dump()
    context["output_mode"] = output_mode
    context["response_profile"] = response_profile
    context.update(build_delivery_grounding_context(user_text=_user_text_for_classifier(client_request), output_mode=output_mode))
    if isinstance(output_mode_decision, dict):
        context.setdefault("metadata", {})["output_mode_decision"] = output_mode_decision

    # Ensure delivery_pack for all delivery-oriented modes (shared merge logic)
    from orion.cognition.runtime_pack_merge import ensure_delivery_pack_in_packs

    context["packs"] = ensure_delivery_pack_in_packs(
        context.get("packs"),
        output_mode=output_mode,
        user_text=_user_text_for_classifier(client_request),
    )
    args = _plan_args(client_request, correlation_id)
    if isinstance(args.extra, dict):
        args.extra["packs"] = list(context.get("packs") or [])
    ut_preview = _user_text_for_classifier(client_request)
    logger.info(
        "orch_plan_wiring corr=%s output_mode=%s profile=%s packs=%s supervised=%s force_agent_chain=%s output_mode_decision=%s",
        correlation_id,
        output_mode,
        response_profile,
        context.get("packs"),
        bool(args.extra.get("supervised")) if isinstance(args.extra, dict) else False,
        bool(args.extra.get("force_agent_chain")) if isinstance(args.extra, dict) else False,
        bool(context.get("metadata", {}).get("output_mode_decision")) if isinstance(context.get("metadata"), dict) else False,
    )
    logger.info(
        "orch_plan_user_text_preview corr=%s verb=%s user_text_len=%s head=%r",
        correlation_id,
        client_request.verb,
        len(ut_preview),
        ut_preview[:220],
    )

    # Attach latest Orion state (Spark) as a read-model artifact
    context.setdefault("metadata", {})["orion_state_pending"] = True

    execution_depth = None
    if isinstance(client_request.options, dict):
        execution_depth = client_request.options.get("execution_depth")
    if execution_depth is not None:
        normalized_depth = str(int(execution_depth))
        plan.metadata["execution_depth"] = normalized_depth
        context.setdefault("metadata", {})["execution_depth"] = int(execution_depth)
    if router_metadata:
        context.setdefault("metadata", {})["auto_route"] = router_metadata
        plan.metadata["auto_route"] = json.dumps(router_metadata, default=str)
        if isinstance(router_metadata, dict) and router_metadata.get("execution_depth") is not None:
            normalized_depth = str(int(router_metadata.get("execution_depth")))
            plan.metadata["execution_depth"] = normalized_depth
            context.setdefault("metadata", {})["execution_depth"] = int(router_metadata.get("execution_depth"))
    ac_raw = options.get("answer_contract")
    if isinstance(ac_raw, dict):
        try:
            ac = AnswerContract.model_validate(ac_raw)
            context.setdefault("metadata", {})["answer_contract"] = ac.model_dump(mode="json")
            context.setdefault("metadata", {})["investigation_state"] = investigation_state_for_contract(ac)
            context.setdefault("metadata", {})["evidence_first_investigation"] = bool(
                ac.requires_repo_grounding or ac.requires_runtime_grounding
            )
            if ac.requires_repo_grounding or ac.requires_runtime_grounding:
                plan.metadata["evidence_first_investigation"] = "1"
        except Exception as exc:
            logger.warning("orch_answer_contract_attach_failed corr=%s err=%s", correlation_id, exc)
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
    trace: dict | None = None,
    plan_request: PlanExecutionRequest | None = None,
) -> Dict[str, Any]:
    """Dispatch PlanExecution to cortex-exec.

    When ``plan_request`` is provided (direct exec lane path), it must already include
    state, mind, router, and lane metadata merged by :func:`call_verb_runtime`.
    """
    if plan_request is not None:
        request_object = plan_request
    else:
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
    return await client.execute(
        source=source,
        req=request_object,
        correlation_id=correlation_id,
        timeout_sec=timeout_sec,
        diagnostic=diagnostic,
    )


def build_verb_request(
    *,
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None = None,
    trace: dict | None = None,
    lane_decision: ExecutionLaneDecision | None = None,
    verb_runtime_request_id: str | None = None,
) -> tuple[VerbRequestV1, BaseEnvelope]:
    request_id = verb_runtime_request_id or str(uuid4())
    lane = lane_decision or resolve_execution_lane(client_request)
    verb_request = VerbRequestV1(
        trigger="cortex.client.request",
        schema_id="PlanExecutionRequest",
        payload=plan_request.model_dump(mode="json"),
        request_id=request_id,
        caller=source.name,
        meta={
            "mode": client_request.mode,
            "route_intent": client_request.route_intent,
            "user_id": client_request.context.user_id,
            "session_id": client_request.context.session_id,
            "trace_id": client_request.context.trace_id,
            "exec_lane": lane.lane,
            "exec_lane_reason": lane.reason,
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
    envelope.reply_to = f"orion:verb:result:{request_id}"
    return verb_request, envelope


async def call_verb_runtime(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    client_request: CortexClientRequest,
    correlation_id: str,
    timeout_sec: float = 900.0,
    causality_chain: list | None = None,
    trace: dict | None = None,
    router_metadata: dict[str, Any] | None = None,
) -> VerbResultV1:
    settings = get_settings()
    base_lane_decision = resolve_execution_lane(client_request)
    lane_decision = (
        base_lane_decision
        if settings.exec_lane_routing_enabled
        else replace(base_lane_decision, reason="lane_routing_disabled")
    )

    plan_request = build_plan_request(client_request, correlation_id, router_metadata=router_metadata)
    trace_id = (trace or {}).get("trace_id") or getattr(client_request.context, "trace_id", None) or correlation_id
    mode_s = str(client_request.mode or "")
    verb_s = str(client_request.verb or plan_request.plan.verb_name or "")

    logger.info(
        "orch_execution_lane_decision corr=%s trace_id=%s mode=%s verb=%s lane=%s reason=%s explicit=%s requested=%s",
        correlation_id,
        trace_id,
        mode_s,
        verb_s,
        lane_decision.lane,
        lane_decision.reason,
        lane_decision.explicit,
        lane_decision.requested,
    )

    plan_request.context.setdefault("metadata", {})["execution_lane"] = lane_decision.lane
    plan_request.context["metadata"]["execution_lane_reason"] = lane_decision.reason

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

    cr_meta = client_request.context.metadata if isinstance(client_request.context.metadata, dict) else {}
    if _mind_enabled_exact(cr_meta):
        substrate_facet: dict[str, Any] | None = None
        inline = cr_meta.get("substrate_telemetry_facet")
        if isinstance(inline, dict):
            substrate_facet = inline
        else:
            substrate_facet = await fetch_substrate_telemetry_facet_for_mind(correlation_id)
        mind_req = build_mind_run_request(
            client_request,
            plan_request,
            correlation_id,
            substrate_telemetry_facet=substrate_facet,
        )
        try:
            mind_res = await call_orion_mind_http(mind_req)
            merge_mind_brief_into_plan_metadata(plan_request, mind_res)
            try:
                await publish_mind_run_artifact(
                    bus,
                    source=source,
                    correlation_id=correlation_id,
                    causality_chain=causality_chain,
                    trace=trace,
                    client_request=client_request,
                    mind_req=mind_req,
                    mind_res=mind_res,
                )
            except Exception as pub_exc:
                logger.warning(
                    "mind_artifact_publish_failed corr=%s err=%s",
                    correlation_id,
                    pub_exc,
                )
                plan_request.context.setdefault("metadata", {})["mind_artifact_persist_failed"] = True
        except Exception as mind_exc:
            logger.warning("mind_http_failed corr=%s err=%s", correlation_id, mind_exc)
            plan_request.context.setdefault("metadata", {})["mind_invocation_failed"] = str(mind_exc)

    use_direct_exec = settings.exec_lane_routing_enabled and lane_decision.lane != "chat"
    # Same id allocation as the verb envelope path so VerbResultV1.request_id is stable per invocation
    # whether orch uses direct PlanExecution RPC or orion:verb:request.
    verb_runtime_request_id = str(uuid4())

    if use_direct_exec:
        exec_channel = settings.exec_request_channel_for_lane(lane_decision.lane)
        logger.info(
            "orch_publish_exec_lane corr=%s trace_id=%s lane=%s channel=%s verb=%s",
            correlation_id,
            trace_id,
            lane_decision.lane,
            exec_channel,
            verb_s,
        )
        exec_trace = dict(trace or {})
        if trace_id and "trace_id" not in exec_trace:
            exec_trace["trace_id"] = trace_id
        raw = await call_cortex_exec(
            bus,
            source=source,
            exec_request_channel=exec_channel,
            exec_result_prefix=settings.channel_exec_result_prefix,
            client_request=client_request,
            correlation_id=correlation_id,
            timeout_sec=timeout_sec,
            trace=exec_trace or None,
            plan_request=plan_request,
        )
        ok = isinstance(raw, dict) and str(raw.get("status") or "").lower() == "success"
        return VerbResultV1(
            verb="legacy.plan",
            ok=ok,
            output={"result": raw} if isinstance(raw, dict) else {"result": {}},
            request_id=verb_runtime_request_id,
        )

    verb_request, envelope = build_verb_request(
        client_request=client_request,
        plan_request=plan_request,
        source=source,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=trace,
        lane_decision=lane_decision,
        verb_runtime_request_id=verb_runtime_request_id,
    )
    request_summary = {
        "corr_id": correlation_id,
        "source_service": source.name,
        "reply_channel": envelope.reply_to,
        "mode": client_request.mode,
        "verb": client_request.verb or plan_request.plan.verb_name,
        "supervised": bool((client_request.options or {}).get("supervised")),
        "recall_enabled": bool(client_request.recall.enabled),
        "recall_profile": client_request.recall.profile,
        "packs": list(plan_request.context.get("packs") or []),
        "output_mode": plan_request.context.get("output_mode"),
        "response_profile": plan_request.context.get("response_profile"),
    }
    logger.info("orch_publish_verb_runtime %s", json.dumps(request_summary, sort_keys=True, default=str))
    logger.info(
        "orch_publish_verb_intake corr=%s trace_id=%s lane=%s channel=%s verb=%s",
        correlation_id,
        trace_id,
        lane_decision.lane,
        "orion:verb:request",
        verb_s,
    )

    async def _wait_for_result() -> VerbResultV1:
        reply_channel = str(envelope.reply_to or "orion:verb:result")
        logger.info(
            "orch_wait_verb_runtime corr=%s reply=%s request_id=%s",
            correlation_id,
            reply_channel,
            verb_request.request_id,
        )
        async with bus.subscribe(reply_channel) as pubsub:
            await bus.publish("orion:verb:request", envelope)
            async for msg in bus.iter_messages(pubsub):
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok or decoded.envelope is None:
                    logger.warning(
                        "orch_wait_verb_runtime_decode_failed corr=%s reply=%s error=%s",
                        correlation_id,
                        reply_channel,
                        decoded.error,
                    )
                    continue
                payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
                try:
                    result = VerbResultV1.model_validate(payload)
                except Exception:
                    logger.warning(
                        "orch_wait_verb_runtime_invalid_payload corr=%s reply=%s kind=%s",
                        correlation_id,
                        reply_channel,
                        decoded.envelope.kind,
                    )
                    continue
                if result.request_id == verb_request.request_id:
                    logger.info(
                        "orch_verb_runtime_result corr=%s reply=%s request_id=%s ok=%s",
                        correlation_id,
                        reply_channel,
                        result.request_id,
                        result.ok,
                    )
                    return result
                logger.info(
                    "orch_verb_runtime_skip corr=%s reply=%s expected_request_id=%s got_request_id=%s",
                    correlation_id,
                    reply_channel,
                    verb_request.request_id,
                )
        raise RuntimeError("Verb result subscription closed without a match.")

    try:
        return await asyncio.wait_for(_wait_for_result(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"RPC timeout waiting on {envelope.reply_to or 'orion:verb:result'} ({verb_request.request_id})"
        ) from exc


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
