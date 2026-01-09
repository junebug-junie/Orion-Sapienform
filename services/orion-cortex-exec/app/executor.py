from __future__ import annotations

"""
Core execution engine for cortex-exec.
Handles recall, planner-react, agent-chain, and LLM Gateway hops over the bus.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.core.contracts.recall import RecallQueryV1

from orion.schemas.agents.schemas import AgentChainRequest, DeliberationRequest
from orion.core.verbs import VerbResultV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, find_collapse_entry
from orion.schemas.cortex.schemas import ExecutionStep, StepExecutionResult

from .settings import settings
from .clients import AgentChainClient, LLMGatewayClient, RecallClient, PlannerReactClient

logger = logging.getLogger("orion.cortex.exec")


def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_str or "")
    return tmpl.render(**ctx)


def _last_user_message(ctx: Dict[str, Any]) -> str:
    msgs = ctx.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
            if isinstance(m, LLMMessage) and getattr(m, "role", None) == "user":
                return str(getattr(m, "content", "") or "")
            if hasattr(m, "model_dump"):
                try:
                    d = m.model_dump(mode="json")
                    if isinstance(d, dict) and d.get("role") == "user":
                        return str(d.get("content") or "")
                except Exception:
                    pass
    return str(ctx.get("user_message") or "")


def _json_sanitize(obj: Any, *, _seen: set[int] | None = None, _depth: int = 0, _max_depth: int = 10) -> Any:
    """
    Best-effort JSON-safe conversion with circular reference protection.

    # KEEP / IMPORTANT:
    This is NOT for "pretty serialization". It's for safety when objects or
    accidental backrefs leak into payloads. If we see a cycle, we replace it.
    """
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    oid = id(obj)
    if oid in _seen:
        return "<circular_ref>"
    if _depth >= _max_depth:
        return "<max_depth>"

    if isinstance(obj, dict):
        _seen.add(oid)
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        _seen.remove(oid)
        return out

    if isinstance(obj, (list, tuple, set)):
        _seen.add(oid)
        out_list = [_json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
        _seen.remove(oid)
        return out_list

    if isinstance(obj, LLMMessage):
        return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)

    if hasattr(obj, "model_dump"):
        try:
            return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            return str(obj)

    return str(obj)


def _build_hop_messages(
    *,
    prompt: str,
    ctx_messages: Any,
) -> List[Dict[str, Any]]:
    """
    Build the message list for downstream LLM-like services.

    # KEEP / IMPORTANT:
    Step execution (prompt_template) is the lynchpin of Orion's architecture.
    The rendered step prompt MUST be injected as a SYSTEM message on every hop.

    # KEEP / IMPORTANT (DUAL PATHWAY):
    DO NOT mutate ctx["messages"].
    DO NOT prepend the system prompt into ctx.
    This function returns a NEW list of NEW dicts for the outbound hop only.
    This preserves the race-safe/persistence-safe pathway used by Spark + SQL-writer.
    """
    normalized: List[Dict[str, Any]] = []

    raw_msgs = ctx_messages or []
    if isinstance(raw_msgs, list):
        for m in raw_msgs:
            if isinstance(m, dict):
                normalized.append(_json_sanitize(m))
            elif isinstance(m, LLMMessage):
                normalized.append(_json_sanitize(m.model_dump(mode="json")))
            elif hasattr(m, "model_dump"):
                try:
                    normalized.append(_json_sanitize(m.model_dump(mode="json")))
                except Exception:
                    pass

    if prompt and str(prompt).strip():
        sys_msg = {"role": "system", "content": str(prompt)}

        # Replace upstream system rather than stacking multiple system messages.
        if normalized and isinstance(normalized[0], dict) and normalized[0].get("role") == "system":
            normalized[0] = sys_msg
        else:
            normalized = [sys_msg] + normalized

    if not normalized:
        content = (prompt or " ").strip() or " "
        normalized = [{"role": "user", "content": content}]

    return normalized



async def run_recall_step(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    recall_profile: str | None = None,
    step_name: str = "recall",
    step_order: int = -1,
    diagnostic: bool = False,
) -> Tuple[StepExecutionResult, Dict[str, Any], str]:
    t0 = time.time()
    recall_client = RecallClient(bus)
    reply_channel = f"orion:exec:result:RecallService:{uuid4()}"

    # FIX: Define the timeout variable that was missing
    recall_timeout = float(settings.step_timeout_ms) / 1000.0

    fragment = _last_user_message(ctx) or ""
    trace_val = ctx.get("trace_id") or recall_cfg.get("trace_id") or correlation_id
    req = RecallQueryV1(
        fragment=fragment,
        verb=str(ctx.get("verb") or recall_cfg.get("verb") or "unknown"),
        intent=ctx.get("intent"),
        session_id=ctx.get("session_id"),
        node_id=ctx.get("node_id"),
        profile=recall_profile or recall_cfg.get("profile") or "reflect.v1",
        reply_to=reply_channel,
    )

    logs: List[str] = [f"rpc -> RecallService (profile={req.profile})"]
    debug: Dict[str, Any] = {}
    try:
        res = await recall_client.query(
            source=source,
            req=req,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=recall_timeout,
        )
        bundle = res.bundle
        debug = {
            "count": len(bundle.items),
            "profile": req.profile,
            "error": None,
        }
        memory_digest = bundle.rendered if hasattr(bundle, "rendered") else ""
        ctx["memory_digest"] = memory_digest
        ctx["memory_bundle"] = bundle.model_dump(mode="json")
        ctx["memory_used"] = True
        ctx["recall_fragments"] = [i.model_dump(mode="json") for i in bundle.items]
        logs.append(f"ok <- RecallService ({len(bundle.items)} items)")

        return (
            StepExecutionResult(
                status="success",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name=step_name,
                order=step_order,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
            ),
            debug,
            memory_digest,
        )
    except Exception as e:
        logs.append(f"exception <- RecallService: {e}")
        debug["error"] = str(e)
        return (
            StepExecutionResult(
                status="fail",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name=step_name,
                order=step_order,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=str(e),
            ),
            debug,
            "",
        )


async def call_step_services(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    step: ExecutionStep,
    ctx: Dict[str, Any],
    correlation_id: str,
    diagnostic: bool = False,
) -> StepExecutionResult:
    t0 = time.time()
    logs: List[str] = []
    merged_result: Dict[str, Any] = {}
    # Optional embedding vector returned by some LLM backends (e.g. the
    # llama.cpp neural host). Keep it at the step level so downstream
    # consumers (Spark introspector, vector writers, etc.) don't need to
    # know how each service nests its result payload.
    spark_vector: list[float] | None = None

    # DEBUG: Log Context Keys to prove data is present
    logger.info(f"--- EXEC STEP '{step.step_name}' START ---")
    logger.info(f"Context Keys available: {list(ctx.keys())}")

    prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""

    # DEBUG: Log Rendered Prompt (Truncated)
    debug_prompt = (prompt[:200] + "...") if len(prompt) > 200 else prompt
    logger.info(f"Rendered Prompt: {debug_prompt!r}")

    # Calculate Timeout from Step Definition (default to 60s if missing)
    # The YAML says 60000ms, so we convert to 60.0s
    step_timeout_sec = (step.timeout_ms or 60000) / 1000.0
    
    # FIX: Define effective_timeout so the loop below can use it
    effective_timeout = step_timeout_sec

    # Instantiate Clients
    llm_client = LLMGatewayClient(bus)
    planner_client = PlannerReactClient(bus)
    agent_client = AgentChainClient(bus)

    for service in step.services:
        reply_channel = f"orion:exec:result:{service}:{uuid4()}"

        try:
            if service == "RecallService":
                logs.append(f"rpc -> RecallService (reply={reply_channel}, profile={step.recall_profile})")
                recall_step, recall_debug, memory_digest = await run_recall_step(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=ctx.get("recall") or {},
                    recall_profile=step.recall_profile,
                    step_name=step.step_name,
                    step_order=step.order,
                    diagnostic=diagnostic,
                )
                merged_result["RecallService"] = recall_debug
                logs.extend(recall_step.logs)
                return recall_step

            if service == "LLMGatewayService":
                # --- STRICT PATH ---
                # 1. Build Pydantic Model
                req_model = ctx.get("model") or ctx.get("llm_model") or None

                # KEEP / IMPORTANT:
                # Step prompt_template MUST govern this hop even if ctx["messages"] exists.
                messages_payload = _build_hop_messages(
                    prompt=prompt,
                    ctx_messages=ctx.get("messages"),
                )

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    raw_user_text=ctx.get("raw_user_text") or _last_user_message(ctx),
                    options={
                        "temperature": float(ctx.get("temperature", 0.7)),
                        "max_tokens": int(ctx.get("max_tokens", 512)),
                        "stream": False,  # Keep this fix!
                    }
                )

                # 2. Delegate to Client WITH TIMEOUT
                logs.append(f"rpc -> LLMGateway via client (timeout={effective_timeout}s)")
                result_object = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )

                # Explicitly dump to dict for storage in result payload
                merged_result[service] = result_object.model_dump(mode="json")

                # Capture optional embedding vector if present. Some backends
                # don't return it ("vec=no"), others do ("vec=yes").
                if spark_vector is None:
                    try:
                        _sv = getattr(result_object, "spark_vector", None)
                    except Exception:
                        _sv = None
                    if _sv:
                        spark_vector = _sv
                logs.append(f"ok <- {service}")

            elif service == "AgentChainService":
                # KEEP / IMPORTANT:
                # Agent chain must also receive the step prompt_template as a system message.
                hop_msgs = _build_hop_messages(
                    prompt=prompt,
                    ctx_messages=ctx.get("messages"),
                )

                agent_req = AgentChainRequest(
                    text=_last_user_message(ctx),
                    mode=ctx.get("mode") or "agent",
                    session_id=ctx.get("session_id"),
                    user_id=ctx.get("user_id"),
                    messages=[
                        LLMMessage(**m) if not isinstance(m, LLMMessage) else m
                        for m in (hop_msgs or [])
                    ],
                    packs=ctx.get("packs") or [],
                )
                logs.append(f"rpc -> AgentChainService (reply={reply_channel}, timeout={effective_timeout}s)")
                agent_res = await agent_client.run_chain(
                    source=source,
                    req=agent_req,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )
                merged_result[service] = agent_res.model_dump(mode="json")
                logs.append("ok <- AgentChainService")

            elif service == "PlannerReactService":
                planner_req = PlannerRequest(
                    request_id=str(correlation_id),
                    caller="cortex-exec",
                    goal=Goal(description=_last_user_message(ctx), metadata={"verb": step.verb_name}),
                    context=ContextBlock(conversation_history=[LLMMessage(**m) for m in (ctx.get("messages") or [])]),
                    toolset=[],
                )
                logs.append(f"rpc -> PlannerReactService (timeout={effective_timeout}s)")
                planner_res = await planner_client.plan(
                    source=source,
                    req=planner_req,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )
                merged_result[service] = planner_res.model_dump(mode="json")
                logs.append("ok <- PlannerReactService")
                # expose planner trace to downstream agent chain calls
                ctx.setdefault("planner_trace", planner_res.model_dump(mode="json"))


            elif service == "CouncilService":
                council_req = DeliberationRequest(
                    prompt=_last_user_message(ctx),
                    history=ctx.get("messages") or [],
                    tags=ctx.get("packs") or [],
                    universe=ctx.get("mode") or "agent",
                    response_channel=reply_channel,
                )

                env = BaseEnvelope(
                    kind="council.request",
                    source=source,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    payload=council_req.model_dump(mode="json"),
                )

                logs.append(f"rpc -> CouncilService reply={reply_channel}")
                msg = await bus.rpc_request(
                    settings.channel_council_intake,
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=step_timeout_sec,
                )
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    raise RuntimeError(f"CouncilService decode failed: {decoded.error}")

                payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
                merged_result[service] = payload
                logs.append("ok <- CouncilService")

            elif service == "VerbRequestService":
                candidate = None
                if isinstance(ctx.get("collapse_entry"), dict):
                    candidate = ctx.get("collapse_entry")
                elif isinstance(ctx.get("collapse_json"), str):
                    try:
                        import json

                        candidate = json.loads(ctx.get("collapse_json"))
                    except Exception:
                        candidate = None

                if not candidate:
                    candidate = find_collapse_entry(ctx.get("prior_step_results"))

                if not isinstance(candidate, dict):
                    merged_result[service] = {"ok": False, "reason": "no_collapse_entry"}
                    logs.append("skip <- VerbRequestService (no entry)")
                elif str(candidate.get("type", "")).strip().lower() == "noop":
                    merged_result[service] = {"ok": True, "reason": "noop"}
                    logs.append("skip <- VerbRequestService (noop)")
                else:
                    try:
                        entry = CollapseMirrorEntryV2.model_validate(candidate)
                    except Exception as exc:
                        merged_result[service] = {"ok": False, "reason": "invalid_collapse_entry", "error": str(exc)}
                        logs.append("skip <- VerbRequestService (invalid entry)")
                    else:
                        envelope = BaseEnvelope(
                            kind="collapse.mirror.intake",
                            source=source,
                            correlation_id=correlation_id,
                            payload=entry.model_dump(mode="json"),
                        )
                        await bus.publish("orion:collapse:intake", envelope)
                        merged_result[service] = {
                            "ok": True,
                            "published": True,
                            "channel": "orion:collapse:intake",
                            "event_id": entry.event_id,
                        }
                        logs.append("publish -> orion:collapse:intake")

            else:
                logs.append(f"skip <- {service} (generic path not implemented in example)")

        except Exception as e:
            logs.append(f"exception <- {service}: {e}")
            logger.error(f"Service {service} failed: {e}")
            return StepExecutionResult(
                status="fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result=merged_result,
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=f"{service}: {e}",
            )

    return StepExecutionResult(
        status="success",
        verb_name=step.verb_name,
        step_name=step.step_name,
        order=step.order,
        result=merged_result,
        spark_vector=spark_vector,
        latency_ms=int((time.time() - t0) * 1000),
        node=settings.node_name,
        logs=logs,
    )
