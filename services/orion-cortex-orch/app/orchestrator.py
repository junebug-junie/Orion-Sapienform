# services/orion-cortex-orch/app/orchestrator.py
from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

import orion  # used to locate installed package path for cognition/verbs

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs import VerbRequestV1, VerbResultV1
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
from orion.schemas.telemetry.metacognition import MetacognitionTickV1, MetacognitionEnrichedV1
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest, PlanExecutionArgs


logger = logging.getLogger("orion.cortex.orch")

# Locate cognition directories
ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"
PROMPTS_DIR = ORION_PKG_DIR / "cognition" / "prompts"


def _load_verb_yaml(verb_name: str) -> dict:
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_prompt_content(template_ref: Optional[str]) -> Optional[str]:
    """
    If template_ref looks like a file (ends in .j2), load its content.
    Otherwise return it as-is (assuming it's a raw string or None).
    """
    if not template_ref:
        return None

    if template_ref.strip().endswith(".j2"):
        prompt_path = PROMPTS_DIR / template_ref.strip()
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Prompt template file not found: {prompt_path}")
            # Fallback: return the filename so at least something happens
            return template_ref

    return template_ref


def build_plan_for_verb(verb_name: str, *, mode: str = "brain") -> ExecutionPlan:
    data = _load_verb_yaml(verb_name)

    # Defaults
    timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
    default_services = list(data.get("services") or [])

    # Load the raw content if it's a file reference
    raw_template_ref = str(data.get("prompt_template") or "")
    default_prompt = _load_prompt_content(raw_template_ref)

    steps: List[ExecutionStep] = []
    raw_steps = data.get("steps") or data.get("plan")  # handle 'plan' alias in yaml

    if isinstance(raw_steps, list) and raw_steps:
        for i, s in enumerate(raw_steps):
            # Resolve step-level prompt if provided, else use default
            step_template_ref = str(s.get("prompt_template") or "")
            step_prompt = _load_prompt_content(step_template_ref) if step_template_ref else default_prompt

            steps.append(
                ExecutionStep(
                    verb_name=verb_name,
                    step_name=str(s.get("name") or f"step_{i}"),
                    description=str(s.get("description") or ""),
                    order=int(s.get("order", i)),
                    services=list(s.get("services") or default_services),
                    prompt_template=step_prompt,
                    requires_gpu=bool(s.get("requires_gpu", False)),
                    requires_memory=bool(s.get("requires_memory", False)),
                    timeout_ms=int(s.get("timeout_ms", timeout_ms) or timeout_ms),
                )
            )
    else:
        # Single-step inference
        steps.append(
            ExecutionStep(
                verb_name=verb_name,
                step_name=verb_name,
                description=str(data.get("description") or ""),
                order=0,
                services=default_services,
                prompt_template=default_prompt,
                requires_gpu=bool(data.get("requires_gpu", False)),
                requires_memory=bool(data.get("requires_memory", False)),
                timeout_ms=timeout_ms,
            )
        )

    return ExecutionPlan(
        verb_name=verb_name,
        label=str(data.get("label") or verb_name),
        description=str(data.get("description") or ""),
        category=str(data.get("category") or "general"),
        priority=str(data.get("priority") or "normal"),
        interruptible=bool(data.get("interruptible", True)),
        can_interrupt_others=bool(data.get("can_interrupt_others", False)),
        timeout_ms=timeout_ms,
        max_recursion_depth=int(data.get("max_recursion_depth", 2) or 2),
        steps=steps,
        metadata={"verb_yaml": f"{verb_name}.yaml", "mode": mode},
    )


def build_agent_plan(verb_name: str) -> ExecutionPlan:
    """Two-step agent plan: planner-react followed by agent chain."""
    return ExecutionPlan(
        verb_name=verb_name,
        label=f"{verb_name}-agent",
        description="Agent chain execution via planner-react",
        category="agentic",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=verb_name,
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
                verb_name=verb_name,
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


def build_council_plan(verb_name: str) -> ExecutionPlan:
    """Stub council plan; routed to CouncilService."""
    return ExecutionPlan(
        verb_name=verb_name,
        label=f"{verb_name}-council",
        description="Council supervisor stub",
        category="council",
        priority="normal",
        interruptible=False,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=verb_name,
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


async def dispatch_metacognition_tick(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    env: BaseEnvelope,
) -> None:
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        tick = MetacognitionTickV1.model_validate(payload)
    except Exception as exc:
        logger.warning("Metacognition tick invalid: %s", exc)
        return

    # 1-step plan: MetaTagsService (LLM can be inserted later)
    plan = ExecutionPlan(
        verb_name="metacognition.enrich",
        label="metacognition.enrich",
        description="Enrich metacognition tick via MetaTagsService (LLM later)",
        category="telemetry",
        priority="low",
        steps=[
            ExecutionStep(
                verb_name="metacognition.enrich",
                step_name="meta_tags",
                description="Extract tags/entities from metacognition tick",
                order=0,
                services=["MetaTagsService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=False,
                timeout_ms=60000,
            )
        ],
    )

    req = PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=tick.tick_id,
            trigger_source="cortex-orch",
            extra={"tick_id": tick.tick_id},
        ),
        context={
            "verb": "metacognition.enrich",
            "metacognition_tick": tick.model_dump(mode="json"),
        },
    )

    # Send to cortex-exec
    settings = get_settings()
    client = CortexExecClient(
        bus,
        request_channel=settings.channel_exec_request,
        result_prefix=settings.channel_exec_result_prefix,
    )
    result = await client.execute_plan(
        source=source,
        req=req,
        correlation_id=tick.tick_id,
        timeout_sec=90.0,
    )

    # Expect MetaTagsService to put tags/entities in step[0].result["MetaTagsService"] or similar
    tags: list[str] = []
    entities: list[str] = []
    try:
        steps = result.get("steps") or []
        if steps and isinstance(steps[0], dict):
            step_result = steps[0].get("result") or {}
            mt = step_result.get("MetaTagsService") or step_result.get("meta_tags") or step_result
            if isinstance(mt, dict):
                tags = list(mt.get("tags") or [])
                entities = list(mt.get("entities") or [])
    except Exception:
        pass

    enriched = MetacognitionEnrichedV1(
        tick_id=tick.tick_id,
        generated_at=tick.generated_at,
        source_service=tick.source_service,
        source_node=tick.source_node,
        distress_score=tick.distress_score,
        zen_score=tick.zen_score,
        services_tracked=tick.services_tracked,
        tags=tags,
        entities=entities,
        raw_tick=tick.model_dump(mode="json"),
    )

    out_env = BaseEnvelope(
        kind="metacognition.enriched.v1",
        source=source,
        correlation_id=tick.tick_id,
        payload=enriched.model_dump(mode="json"),
    )

    await bus.publish("orion:metacognition:enriched", out_env)
    logger.info("Published metacognition enriched tick_id=%s tags=%d", tick.tick_id, len(tags))



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


async def dispatch_metacognition_tick(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    env: BaseEnvelope,
) -> None:
    """
    Model A: metacognition is telemetry, not collapse mirror.

    Flow:
      equilibrium -> orion:metacognition:tick
      cortex-orch -> cortex-exec plan (MetaTagsService step)
      cortex-orch -> orion:metacognition:enriched
    """
    settings = get_settings()

    payload = env.payload if isinstance(env.payload, dict) else {}
    if not isinstance(payload, dict):
        logger.warning("Metacognition tick payload not a dict; skipping")
        return

    # Stable id for joins across services
    tick_id = (
        payload.get("tick_id")
        or payload.get("event_id")
        or (str(env.correlation_id) if env.correlation_id else None)
        or str(uuid4())
    )

    # Minimal 1-step plan (LLM step can be inserted later)
    plan = ExecutionPlan(
        verb_name="metacognition.enrich",
        label="metacognition.enrich",
        description="Enrich metacognition tick via MetaTagsService (LLM later)",
        category="telemetry",
        priority="low",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=60000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name="metacognition.enrich",
                step_name="meta_tags",
                description="Extract tags/entities from metacognition tick",
                order=0,
                services=["MetaTagsService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=False,
                timeout_ms=60000,
            )
        ],
        metadata={"origin": "cortex-orch"},
    )

    req = PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=tick_id,
            user_id=None,
            trigger_source="cortex-orch",
            extra={
                # IMPORTANT: disable recall for telemetry ticks
                "recall": {"enabled": False, "required": False},
                "mode": "brain",
                "verb": "metacognition.enrich",
                "metacognition_tick_id": tick_id,
            },
        ),
        context={
            # This is what cortex-exec step services should read
            "metacognition_tick": payload,
        },
    )

    client = CortexExecClient(
        bus,
        request_channel=settings.channel_exec_request,
        result_prefix=settings.channel_exec_result_prefix,
    )

    try:
        exec_res = await client.execute_plan(
            source=source,
            req=req,
            correlation_id=tick_id,
            timeout_sec=90.0,
        )
    except Exception as exc:
        logger.warning("Metacognition exec plan failed tick_id=%s err=%s", tick_id, exc)
        return

    # Extract tags/entities from exec result
    tags: list[str] = []
    entities: list[str] = []
    try:
        steps = exec_res.get("steps") or []
        for st in steps:
            if not isinstance(st, dict):
                continue
            r = st.get("result") or {}
            if isinstance(r, dict):
                mt = r.get("MetaTagsService")
                if isinstance(mt, dict):
                    tags = list(mt.get("tags") or [])
                    entities = list(mt.get("entities") or [])
                    break
    except Exception:
        pass

    enriched_payload = {
        "tick_id": tick_id,
        "generated_at": payload.get("generated_at") or payload.get("timestamp"),
        "source_service": (payload.get("source_service") or (env.source.name if env.source else None)),
        "source_node": (payload.get("source_node") or (env.source.node if env.source else None)),
        "distress_score": payload.get("distress_score"),
        "zen_score": payload.get("zen_score"),
        "services_tracked": payload.get("services_tracked"),
        "tags": tags,
        "entities": entities,
        "raw_tick": payload,
    }

    out_env = BaseEnvelope(
        kind="metacognition.enriched.v1",
        source=source,
        correlation_id=tick_id,
        causality_chain=list(env.causality_chain),
        trace=dict(env.trace),
        payload=enriched_payload,
    )

    await bus.publish("orion:metacognition:enriched", out_env)
    logger.info("Published metacognition enriched tick_id=%s tags=%d", tick_id, len(tags))



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
