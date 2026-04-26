"""
Canonical Entrypoint for Cortex Exec.
This handles the RabbitMQ/Bus connection and routes requests to the PlanRouter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict
from uuid import NAMESPACE_URL, UUID, uuid5

import requests
from pydantic import Field, ValidationError

# IMPORTS UPDATED: Added Envelope for generic typing
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef, CausalityLink
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit, Hunter
from orion.core.verbs import VerbRequestV1, VerbResultV1, VerbEffectV1, VerbRuntime

from orion.schemas.cortex.exec import CortexExecResultPayload
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult
from orion.schemas.platform import CoreEventV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.metacognitive_trace import MetacognitiveTraceEnvelope, MetacognitiveTraceV1
from .router import PlanRouter
from .settings import settings
from .dream_publish import build_dream_publish_envelope
from .chat_stance import resolve_autonomy_graphdb_config
from .core_event_cache import get_core_event_cache
from .trace_cache import get_trace_cache
from .verb_adapters import LegacyPlanVerb, RespondToJuniperCollapseMirrorVerb  # noqa: F401 - register verb adapter
from .collapse_verbs import (  # noqa: F401 - register collapse verbs
    LogCollapseMirrorVerb,
    EnrichCollapseMirrorVerb,
    ScoreCausalDensityVerb,
)

logger = logging.getLogger("orion.cortex.exec.main")


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _uuid_from_correlation_id(value: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return uuid5(NAMESPACE_URL, str(value))


class CortexExecRequest(BaseEnvelope):
    kind: str = Field("cortex.exec.request", frozen=True)
    payload: PlanExecutionRequest


class CortexExecResult(BaseEnvelope):
    kind: str = Field("cortex.exec.result", frozen=True)
    payload: CortexExecResultPayload


class CognitionTraceEnvelope(Envelope[CognitionTracePayload]):
    """
    Typed contract for cognition traces aligning to Titanium Envelope[T].
    This ensures the payload is validated as a CognitionTracePayload model,
    not forced into a dict before validation.
    """
    kind: str = Field("cognition.trace", frozen=True)


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(settings.heartbeat_interval_sec),
    )


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        version=settings.service_version,
        node=settings.node_name,
    )


router = PlanRouter()
svc: Rabbit | None = None
verb_listener: Hunter | None = None
trace_listener: Hunter | None = None
core_event_listener: Hunter | None = None
verb_runtime: VerbRuntime | None = None


def _diagnostic_enabled(payload: PlanExecutionRequest) -> bool:
    try:
        extra = payload.args.extra or {}
        options = extra.get("options") if isinstance(extra, dict) else {}
        return bool(
            settings.diagnostic_mode
            or extra.get("diagnostic")
            or (isinstance(options, dict) and (options.get("diagnostic") or options.get("diagnostic_mode")))
        )
    except Exception:
        return settings.diagnostic_mode


def _resolve_autonomy_backend() -> str:
    backend = (os.getenv("AUTONOMY_REPOSITORY_BACKEND") or "graph").strip().lower()
    return backend if backend in {"graph", "local", "shadow"} else "graph"


def _run_autonomy_graph_probe() -> None:
    backend = _resolve_autonomy_backend()
    if backend != "graph":
        return

    cfg = resolve_autonomy_graphdb_config()
    endpoint = str(cfg.get("endpoint") or "")
    repo = str(cfg.get("repo") or "collapse")
    user = cfg.get("user")
    password = cfg.get("password")
    auth_present = bool(user and password)
    source = str(cfg.get("source") or "unconfigured")
    endpoint_present = bool(endpoint)

    logger.info(
        "autonomy_graph_probe backend=%s endpoint_present=%s repo=%s auth_present=%s source=%s",
        backend,
        "yes" if endpoint_present else "no",
        repo,
        "yes" if auth_present else "no",
        source,
    )
    if not endpoint_present:
        logger.warning(
            "autonomy_graph_probe result=fail reason=graph_not_configured endpoint=graphdb:unconfigured repo=%s",
            repo,
        )
        return

    timeout_sec = min(max(float(os.getenv("GRAPHDB_PROBE_TIMEOUT_SEC", "3.0")), 1.0), 4.0)
    deep_probe = str(os.getenv("AUTONOMY_GRAPH_PROBE_DEEP", "false")).strip().lower() in {"1", "true", "yes", "on"}
    auth = (str(user), str(password)) if auth_present else None
    ask_query = "ASK { ?s ?p ?o }"
    headers = {
        "Accept": "application/sparql-results+json",
    }
    try:
        response = requests.post(
            endpoint,
            data={"query": ask_query},
            headers=headers,
            timeout=timeout_sec,
            auth=auth,
        )
        if response.status_code == 200:
            parsed_ok = False
            try:
                payload = response.json()
                parsed_ok = isinstance(payload, dict) and isinstance(payload.get("boolean"), bool)
            except Exception:
                body = (response.text or "").strip().lower()
                parsed_ok = body in {"true", "false"} or "<boolean>true</boolean>" in body or "<boolean>false</boolean>" in body
            if parsed_ok:
                logger.info("autonomy_graph_probe result=ok endpoint=%s repo=%s query=ASK", endpoint, repo)
                if not deep_probe:
                    return
                deep_query = "ASK { GRAPH <http://conjourney.net/graph/autonomy/identity> { ?s ?p ?o } }"
                deep_response = requests.post(
                    endpoint,
                    data={"query": deep_query},
                    headers=headers,
                    timeout=timeout_sec,
                    auth=auth,
                )
                if deep_response.status_code == 200:
                    logger.info(
                        "autonomy_graph_probe_deep result=ok endpoint=%s repo=%s graph=autonomy_identity",
                        endpoint,
                        repo,
                    )
                else:
                    logger.warning(
                        "autonomy_graph_probe_deep result=fail reason=http_%s endpoint=%s repo=%s",
                        deep_response.status_code,
                        endpoint,
                        repo,
                    )
                return
            logger.warning(
                "autonomy_graph_probe result=fail reason=parse_error endpoint=%s repo=%s response_snippet=%r",
                endpoint,
                repo,
                _debug_snippet(response.text, max_len=160),
            )
            return
        logger.warning(
            "autonomy_graph_probe result=fail reason=http_%s endpoint=%s repo=%s response_snippet=%r",
            response.status_code,
            endpoint,
            repo,
            _debug_snippet(response.text, max_len=160),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "autonomy_graph_probe result=fail reason=%s endpoint=%s repo=%s",
            exc.__class__.__name__,
            endpoint,
            repo,
        )


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    corr_id = str(env.correlation_id)
    logger.info(f"Incoming Exec Request: correlation_id={corr_id}")
    trace_id = (env.trace or {}).get("trace_id") or corr_id
    parent_event_id = (env.trace or {}).get("event_id") or (env.trace or {}).get("parent_event_id")

    try:
        req_env = CortexExecRequest.model_validate(env.model_dump(mode="json"))
    except ValidationError as ve:
        logger.error("Validation failed trace_id=%s error=%s", trace_id, ve)
        return BaseEnvelope(
            kind="cortex.exec.result",
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload={"ok": False, "error": "validation_failed", "details": ve.errors()},
        )

    # 1. Extract Context (Handling the Pydantic stripping issue)
    raw_payload = env.payload if isinstance(env.payload, dict) else {}
    payload_context = raw_payload.get("context") or req_env.payload.context or {}

    # 2. Merge Context
    plan_metadata = req_env.payload.plan.metadata if isinstance(req_env.payload.plan.metadata, dict) else {}
    ctx = {
        **payload_context,
        **(req_env.payload.args.extra or {}),
        "user_id": req_env.payload.args.user_id,
        "trigger_source": req_env.payload.args.trigger_source,
        "trace_id": trace_id,
        "parent_event_id": parent_event_id,
        "correlation_id": corr_id,
        "plan_metadata": plan_metadata,
    }
    if "personality_file" in plan_metadata:
        # Preserve declaration state (including empty string) for precise identity fallback diagnostics.
        ctx["personality_file"] = plan_metadata.get("personality_file")
    ctx.setdefault("trigger_correlation_id", ctx.get("chat_correlation_id") or corr_id)
    ctx.setdefault("trigger_trace_id", trace_id)

    logger.debug(f"Context loaded with {len(ctx.get('messages', []))} history messages.")

    if req_env.payload.plan.metadata and isinstance(req_env.payload.plan.metadata, dict):
        auto_route_meta = req_env.payload.plan.metadata.get("auto_route")
        if isinstance(auto_route_meta, dict):
            logger.info("Exec received auto_route metadata corr=%s mode=%s verb=%s source=%s", corr_id, auto_route_meta.get("route_mode"), auto_route_meta.get("verb"), auto_route_meta.get("source"))

    if str(ctx.get("mode") or "").lower() == "auto":
        logger.warning("Exec received unexpected auto mode corr=%s; executing plan deterministically as provided", corr_id)

    assert svc is not None, "Rabbit service not initialized"

    diagnostic = _diagnostic_enabled(req_env.payload)
    if diagnostic:
        logger.info("Diagnostic PlanExecutionRequest json=%s", req_env.payload.model_dump_json())
        logger.info("Diagnostic args.extra snapshot corr=%s payload=%s", env.correlation_id, req_env.payload.args.extra)
        ctx["diagnostic"] = True

    # 3. Execute Plan
    res = await router.run_plan(
        svc.bus,
        source=_source(),
        req=req_env.payload,
        correlation_id=corr_id,
        ctx=ctx,
    )

    # 4. Publish Cognition Trace
    try:
        # Attempt to extract 'packs' from args if present (typically passed in extra for agents)
        packs_used = req_env.payload.args.extra.get("packs") or []
        if isinstance(packs_used, str):
            packs_used = [packs_used]

        trace_payload = CognitionTracePayload(
            correlation_id=corr_id,
            mode=res.mode or "brain",
            verb=res.verb_name,
            packs=packs_used if isinstance(packs_used, list) else [],
            options=req_env.payload.args.extra.get("options", {}) if req_env.payload.args.extra else {},
            final_text=res.final_text,
            steps=res.steps,
            timestamp=time.time(),
            source_service=settings.service_name,
            source_node=settings.node_name,
            recall_used=res.memory_used,
            recall_debug=res.recall_debug,
            metadata={
                "request_id": res.request_id,
                "status": res.status,
            }
        )

        # Use the typed envelope
        trace_envelope = CognitionTraceEnvelope(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload=trace_payload
        )

        await svc.bus.publish(settings.channel_cognition_trace_pub, trace_envelope)
        logger.info(f"Published CognitionTrace to {settings.channel_cognition_trace_pub}")

        reasoning_trace = next(
            (
                trace
                for trace in (res.metacog_traces or [])
                if isinstance(trace, MetacognitiveTraceV1) and str(trace.content or "").strip()
            ),
            None,
        )
        extra = req_env.payload.args.extra if req_env.payload.args else {}
        session_id = None
        message_id = None
        if isinstance(extra, dict):
            session_id = extra.get("session_id")
            message_id = extra.get("message_id")
        if not session_id:
            session_id = ctx.get("session_id")
        if not message_id:
            message_id = ctx.get("message_id")

        metacog_payload = MetacognitiveTraceV1(
            correlation_id=corr_id,
            session_id=str(session_id) if session_id is not None else None,
            message_id=str(message_id) if message_id is not None else None,
            trace_role="reasoning",
            trace_stage="pre_answer",
            content=(reasoning_trace.content if reasoning_trace is not None else (res.final_text or "")).strip(),
            model=(reasoning_trace.model if reasoning_trace is not None else "unknown"),
            token_count=reasoning_trace.token_count if reasoning_trace is not None else None,
            confidence=reasoning_trace.confidence if reasoning_trace is not None else None,
            metadata={
                **((reasoning_trace.metadata or {}) if reasoning_trace is not None else {}),
                "request_id": res.request_id,
                "status": res.status,
                "fallback_from_final_text": reasoning_trace is None,
            },
        )
        if _thought_debug_enabled():
            logger.info(
                "THOUGHT_DEBUG_METACOG_PUB stage=prepare corr=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r fallback_from_final_text=%s",
                corr_id,
                metacog_payload.trace_role,
                metacog_payload.trace_stage,
                metacog_payload.model,
                _debug_len(metacog_payload.content),
                _debug_snippet(metacog_payload.content),
                reasoning_trace is None,
            )
        if metacog_payload.content:
            metacog_envelope = MetacognitiveTraceEnvelope(
                source=_source(),
                correlation_id=_uuid_from_correlation_id(corr_id),
                causality_chain=env.causality_chain,
                payload=metacog_payload,
            )
            await svc.bus.publish(settings.channel_metacog_trace_pub, metacog_envelope)
            logger.info(
                "Published MetacognitiveTrace to %s correlation_id=%s fallback=%s",
                settings.channel_metacog_trace_pub,
                corr_id,
                reasoning_trace is None,
            )
            if _thought_debug_enabled():
                logger.info(
                    "THOUGHT_DEBUG_METACOG_PUB stage=published corr=%s channel=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r",
                    corr_id,
                    settings.channel_metacog_trace_pub,
                    metacog_payload.trace_role,
                    metacog_payload.trace_stage,
                    metacog_payload.model,
                    _debug_len(metacog_payload.content),
                    _debug_snippet(metacog_payload.content),
                )
        else:
            logger.info(
                "Skipped MetacognitiveTrace publish due to empty content correlation_id=%s",
                corr_id,
            )
            if _thought_debug_enabled():
                logger.info(
                    "THOUGHT_DEBUG_METACOG_PUB stage=skipped corr=%s reason=empty_content fallback_from_final_text=%s final_text_len=%s",
                    corr_id,
                    reasoning_trace is None,
                    _debug_len(res.final_text),
                )

    except Exception as e:
        logger.error(f"Failed to publish CognitionTrace: {e}", exc_info=True)
        return CortexExecResult(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload=CortexExecResultPayload(ok=False, error=str(e)),
        )

    try:
        dream_env = build_dream_publish_envelope(
            source=_source(),
            causality_chain=list(env.causality_chain or []),
            correlation_id=corr_id,
            res=res,
            context=ctx,
            extra=req_env.payload.args.extra if req_env.payload.args else None,
        )
        if dream_env is not None:
            await svc.bus.publish(settings.channel_dream_log, dream_env)
            logger.info("Published dream.result.v1 to %s", settings.channel_dream_log)
    except Exception as exc:
        logger.warning("dream.result.v1 publish skipped/failed corr=%s err=%s", corr_id, exc)

    if env.reply_to:
        res_payload = res.model_dump(mode="json")
        reasoning_content = res_payload.get("reasoning_content") if isinstance(res_payload, dict) else None
        reasoning_trace = res_payload.get("reasoning_trace") if isinstance(res_payload, dict) else None
        trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
        preview_text = repr(str((reasoning_content if isinstance(reasoning_content, str) else None) or trace_content or "")[:220])
        print(
            "===THINK_HOP=== hop=exec_out "
            f"corr={corr_id} "
            f"payload_keys={sorted(res_payload.keys()) if isinstance(res_payload, dict) else []} "
            f"reasoning_len={len(reasoning_content) if isinstance(reasoning_content, str) else 0} "
            f"trace_len={len(trace_content) if isinstance(trace_content, str) else 0} "
            f"metacog_count={len(res_payload.get('metacog_traces')) if isinstance(res_payload.get('metacog_traces'), list) else 0} "
            f"preview={preview_text}",
            flush=True,
        )
        manual_result = CortexExecResult(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload=CortexExecResultPayload(ok=True, result=res_payload),
        )
        publish_started = time.perf_counter()
        try:
            await svc.bus.publish(env.reply_to, manual_result)
        except Exception as exc:
            elapsed = time.perf_counter() - publish_started
            logger.warning(
                "Exec result publish failed corr=%s reply=%s elapsed=%.2fs error=%s",
                corr_id,
                env.reply_to,
                elapsed,
                exc,
            )
        else:
            elapsed = time.perf_counter() - publish_started
            logger.info(
                "Exec result published corr=%s reply=%s elapsed=%.2fs",
                corr_id,
                env.reply_to,
                elapsed,
            )
    else:
        logger.warning("Exec result missing reply_to corr=%s", corr_id)

    return CortexExecResult(
        source=_source(),
        correlation_id=corr_id,
        causality_chain=env.causality_chain,
        payload=CortexExecResultPayload(ok=True, result=res.model_dump(mode="json")),
    )

def _derive_envelope(env: BaseEnvelope, *, kind: str, payload: dict) -> BaseEnvelope:
    parent_link = CausalityLink(
        correlation_id=env.correlation_id,
        kind=env.kind,
        source=env.source,
        created_at=env.created_at,
    )
    return BaseEnvelope(
        kind=kind,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=[*env.causality_chain, parent_link],
        trace=dict(env.trace),
        payload=payload,
    )


async def handle_trace(env: BaseEnvelope) -> None:
    # Cache recent traces for metacognition context
    try:
        raw = env.payload if isinstance(env.payload, dict) else {}
        # Best effort parsing
        try:
            trace = CognitionTracePayload.model_validate(raw)
        except Exception:
            # Fallback for untyped or partial payloads
            trace = CognitionTracePayload(
                correlation_id=str(env.correlation_id),
                mode="unknown",
                verb="unknown",
                final_text=str(raw.get("final_text") or ""),
                steps=[],
                timestamp=time.time(),
                source_service=env.source.name,
                source_node=env.source.node,
            )
        get_trace_cache().append(trace)
    except Exception as e:
        logger.warning(f"Failed to cache trace: {e}")


async def handle_core_event(env: BaseEnvelope) -> None:
    try:
        raw = env.payload if isinstance(env.payload, dict) else {}
        try:
            event = CoreEventV1.model_validate(raw)
            event_data = event.model_dump(mode="json")
        except Exception:
            event_data = {"event": raw.get("event"), "payload": raw.get("payload"), "meta": raw.get("meta")}
        get_core_event_cache().append(event_data)
    except Exception as e:
        logger.warning("Failed to cache core event: %s", e)


async def handle_verb_request(env: BaseEnvelope) -> None:
    assert svc is not None, "Rabbit service not initialized"
    assert verb_runtime is not None, "Verb runtime not initialized"

    raw_payload = env.payload if isinstance(env.payload, dict) else {}
    corr_id = str(env.correlation_id or raw_payload.get("request_id") or "unknown")
    reply_channel = str(env.reply_to or "orion:verb:result")
    try:
        req = VerbRequestV1.model_validate(raw_payload)
    except ValidationError as ve:
        logger.warning(
            "verb_runtime_validation_failed corr=%s reply=%s request_id=%s error=%s",
            corr_id,
            reply_channel,
            raw_payload.get("request_id"),
            ve,
        )
        error_result = VerbResultV1(
            verb=str(raw_payload.get("trigger") or raw_payload.get("verb") or "unknown"),
            ok=False,
            error=f"invalid_request:{ve}",
            request_id=raw_payload.get("request_id"),
        )
        result_env = _derive_envelope(env, kind="verb.result", payload=error_result.model_dump(mode="json"))
        await svc.bus.publish(reply_channel, result_env)
        if reply_channel != "orion:verb:result":
            await svc.bus.publish("orion:verb:result", result_env)
        return

    logger.info(
        "verb_runtime_intake corr=%s reply=%s request_id=%s trigger=%s",
        corr_id,
        reply_channel,
        req.request_id,
        req.trigger,
    )
    result = await verb_runtime.handle_request(
        req,
        extra_meta={
            "bus": svc.bus,
            "source": _source(),
            "correlation_id": corr_id,
        },
    )

    result_env = _derive_envelope(env, kind="verb.result", payload=result.model_dump(mode="json"))
    await svc.bus.publish(reply_channel, result_env)
    logger.info(
        "verb_runtime_result corr=%s reply=%s request_id=%s ok=%s",
        corr_id,
        reply_channel,
        result.request_id,
        result.ok,
    )
    if reply_channel != "orion:verb:result":
        await svc.bus.publish("orion:verb:result", result_env)
        logger.info(
            "verb_runtime_result_legacy_mirror corr=%s reply=%s legacy_channel=orion:verb:result request_id=%s",
            corr_id,
            reply_channel,
            result.request_id,
        )

    try:
        if req.trigger == "legacy.plan":
            plan_req = PlanExecutionRequest.model_validate(req.payload)
            result_payload = result.output if isinstance(result.output, dict) else {}
            plan_result_payload = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else result_payload
            plan_result = PlanExecutionResult.model_validate(plan_result_payload)
            dream_env = build_dream_publish_envelope(
                source=_source(),
                causality_chain=list(env.causality_chain or []),
                correlation_id=corr_id,
                res=plan_result,
                context=plan_req.context if isinstance(plan_req.context, dict) else {},
                extra=plan_req.args.extra if plan_req.args else None,
            )
            if dream_env is not None:
                await svc.bus.publish(settings.channel_dream_log, dream_env)
                logger.info("Published dream.result.v1 to %s via verb runtime", settings.channel_dream_log)
    except Exception as exc:
        logger.warning("dream.result.v1 legacy-plan publish skipped/failed corr=%s err=%s", corr_id, exc)

    for effect in result.effects:
        effect_model = effect if isinstance(effect, VerbEffectV1) else VerbEffectV1.model_validate(effect)
        effect_env = _derive_envelope(env, kind="verb.effect", payload=effect_model.model_dump(mode="json"))
        await svc.bus.publish(f"orion:effect:{effect_model.kind}", effect_env)


svc = Rabbit(_cfg(), request_channel=settings.channel_exec_request, handler=handle)
verb_runtime = VerbRuntime(
    service_name=settings.service_name,
    instance_id=settings.node_name,
    bus=svc.bus,
    logger=logger,
    allow_backdoor=settings.orion_verb_backdoor_enabled,
)
verb_listener = Hunter(
    _cfg(),
    handler=handle_verb_request,
    patterns=["orion:verb:request"],
    concurrent_handlers=True,
)
trace_listener = Hunter(_cfg(), handler=handle_trace, patterns=["orion:cognition:trace"])
core_event_listener = Hunter(_cfg(), handler=handle_core_event, patterns=[settings.channel_core_events])


async def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    pageindex_base_url = str(settings.journal_pageindex_service_url or "").strip()
    pageindex_client_enabled = bool(pageindex_base_url)
    logger.info(
        "startup journal_pageindex_service_base_url=%s journal_pageindex_client_enabled=%s",
        pageindex_base_url,
        pageindex_client_enabled,
    )
    logger.info(
        f"Starting cortex-exec bus listener channel={settings.channel_exec_request} "
        f"bus={settings.orion_bus_url}"
    )
    _run_autonomy_graph_probe()
    assert verb_listener is not None, "Verb listener not initialized"
    assert trace_listener is not None, "Trace listener not initialized"
    assert core_event_listener is not None, "Core event listener not initialized"
    await verb_listener.start_background()
    await trace_listener.start_background()
    await core_event_listener.start_background()
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
