"""
Canonical Entrypoint for Cortex Exec.
This handles the RabbitMQ/Bus connection and routes requests to the PlanRouter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from pydantic import Field, ValidationError

# IMPORTS UPDATED: Added Envelope for generic typing
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef, CausalityLink
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit, Hunter
from orion.core.verbs import VerbRequestV1, VerbResultV1, VerbEffectV1, VerbRuntime

from orion.schemas.cortex.exec import CortexExecResultPayload
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.platform import CoreEventV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from .router import PlanRouter
from .settings import settings
from .core_event_cache import get_core_event_cache
from .trace_cache import get_trace_cache
from .verb_adapters import LegacyPlanVerb  # noqa: F401 - register verb adapter
from .collapse_verbs import (  # noqa: F401 - register collapse verbs
    LogCollapseMirrorVerb,
    EnrichCollapseMirrorVerb,
    ScoreCausalDensityVerb,
)

logger = logging.getLogger("orion.cortex.exec.main")


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
    ctx = {
        **payload_context,
        **(req_env.payload.args.extra or {}),
        "user_id": req_env.payload.args.user_id,
        "trigger_source": req_env.payload.args.trigger_source,
        "trace_id": trace_id,
        "parent_event_id": parent_event_id,
        "correlation_id": corr_id,
    }
    ctx.setdefault("trigger_correlation_id", ctx.get("chat_correlation_id") or corr_id)
    ctx.setdefault("trigger_trace_id", trace_id)

    logger.debug(f"Context loaded with {len(ctx.get('messages', []))} history messages.")

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

    except Exception as e:
        logger.error(f"Failed to publish CognitionTrace: {e}", exc_info=True)
        return CortexExecResult(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload=CortexExecResultPayload(ok=False, error=str(e)),
        )


    if env.reply_to:
        manual_result = CortexExecResult(
            source=_source(),
            correlation_id=corr_id,
            causality_chain=env.causality_chain,
            payload=CortexExecResultPayload(ok=True, result=res.model_dump(mode="json")),
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
    try:
        req = VerbRequestV1.model_validate(raw_payload)
    except ValidationError as ve:
        error_result = VerbResultV1(
            verb=str(raw_payload.get("trigger") or raw_payload.get("verb") or "unknown"),
            ok=False,
            error=f"invalid_request:{ve}",
            request_id=raw_payload.get("request_id"),
        )
        result_env = _derive_envelope(env, kind="verb.result", payload=error_result.model_dump(mode="json"))
        await svc.bus.publish("orion:verb:result", result_env)
        return

    result = await verb_runtime.handle_request(
        req,
        extra_meta={
            "bus": svc.bus,
            "source": _source(),
            "correlation_id": str(env.correlation_id),
        },
    )

    result_env = _derive_envelope(env, kind="verb.result", payload=result.model_dump(mode="json"))
    await svc.bus.publish("orion:verb:result", result_env)

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
verb_listener = Hunter(_cfg(), handler=handle_verb_request, patterns=["orion:verb:request"])
trace_listener = Hunter(_cfg(), handler=handle_trace, patterns=["orion:cognition:trace"])
core_event_listener = Hunter(_cfg(), handler=handle_core_event, patterns=[settings.channel_core_events])


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"Starting cortex-exec bus listener channel={settings.channel_exec_request} "
        f"bus={settings.orion_bus_url}"
    )
    assert verb_listener is not None, "Verb listener not initialized"
    assert trace_listener is not None, "Trace listener not initialized"
    assert core_event_listener is not None, "Core event listener not initialized"
    await verb_listener.start_background()
    await trace_listener.start_background()
    await core_event_listener.start_background()
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
