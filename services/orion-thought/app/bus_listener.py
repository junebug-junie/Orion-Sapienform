from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.thought import GroundingCapsuleV1, StanceReactRequestV1, ThoughtEventV1
from orion.thought.stance_react import (
    apply_stance_react_pipeline,
    build_stance_react_failure_thought,
    parse_stance_react_payload,
    slim_association_for_prompt,
    slim_repair_bundle_for_prompt,
)

from .cortex_client import CortexExecClient
from .mind_enrichment import (
    build_light_mind_request,
    publish_mind_run_artifact_for_thought,
    run_mind_for_thought,
    select_mind_coloring,
)
from .settings import settings

logger = logging.getLogger("orion-thought.bus")

# Poll get_message with a 1s timeout; after this many idle polls verify Redis still
# lists us as a subscriber. A silent pubsub disconnect leaves health checks green
# but PUBSUB NUMSUB returns 0 — hub RPC then hangs until timeout and the message
# is lost (pubsub is not durable).
_PUBSUB_IDLE_POLLS_BEFORE_HEALTH = 30


async def _thought_channel_subscribers(bus: OrionBusAsync, channel: str) -> int:
    """Return subscriber count for channel, or -1 when the probe itself fails."""
    try:
        pairs = await bus.redis.pubsub_numsub(channel)
    except Exception as exc:  # noqa: BLE001 — probe must not take down the worker
        logger.warning("pubsub health probe failed channel=%s err=%s", channel, exc)
        return -1
    for name, count in pairs:
        key = name.decode() if isinstance(name, bytes) else str(name)
        if key == channel:
            return int(count)
    return 0


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


def _envelope_correlation_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(str(raw))
        except ValueError:
            pass
    return uuid4()


def _coalition_projection(request: StanceReactRequestV1) -> dict[str, Any] | None:
    broadcast = request.association.broadcast
    if broadcast is None:
        return None
    return {
        "attended_node_ids": list(broadcast.attended_node_ids),
        "open_loop_ids": [loop.id for loop in broadcast.frame.open_loops],
        "broadcast_stale": request.association.broadcast_stale,
    }


def build_stance_react_context(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "user_message": request.user_message,
        "stance_inputs": {"user_message": request.user_message},
        "association": slim_association_for_prompt(request.association),
        "repair_bundle": slim_repair_bundle_for_prompt(request.repair_bundle),
        "coalition_projection": _coalition_projection(request),
        "metadata": {
            "correlation_id": request.correlation_id,
            "session_id": request.session_id,
            "llm_profile": request.llm_profile,
            "mode": "brain",
        },
    }
    if mind_coloring is not None:
        context["mind_coloring"] = mind_coloring
    return context


def build_stance_react_plan_request(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
) -> PlanExecutionRequest:
    plan = build_plan_for_verb("stance_react", mode="brain")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=request.correlation_id,
            trigger_source=settings.service_name,
            extra={
                "llm_profile": request.llm_profile,
                "mode": "brain",
            },
        ),
        context=build_stance_react_context(request, mind_coloring=mind_coloring),
    )


def extract_stance_react_payload(result: dict[str, Any]) -> dict[str, Any] | str:
    final_text = result.get("final_text")
    if isinstance(final_text, str) and final_text.strip():
        return final_text

    steps = result.get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_result = step.get("result")
        if not isinstance(step_result, dict):
            continue
        for key in ("structured", "json", "payload", "final_text", "text", "content"):
            value = step_result.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value

    raise ValueError("stance_react exec result missing thought payload")


def _extract_grounding_capsule(exec_result: dict[str, Any]) -> GroundingCapsuleV1 | None:
    metadata = exec_result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("grounding_capsule")
    if not isinstance(raw, dict):
        return None
    try:
        return GroundingCapsuleV1.model_validate(raw)
    except Exception:
        logger.warning("grounding_capsule_parse_failed corr=%s", exec_result.get("request_id"))
        return None


async def _maybe_build_mind_coloring(
    request: StanceReactRequestV1,
    *,
    bus: OrionBusAsync | None,
) -> dict[str, Any] | None:
    """Run Mind and select advisory coloring. Fail-open: any error/None short-circuits."""
    if not settings.mind_enrichment_enabled:
        return None
    try:
        mind_req = build_light_mind_request(
            request,
            wall_time_ms=settings.mind_wall_ms,
            router_profile=settings.mind_router_profile,
        )
        result = await run_mind_for_thought(
            mind_req,
            settings=settings,
            correlation_id=request.correlation_id,
        )
        if result is None:
            return None
        coloring = select_mind_coloring(result, max_items=settings.mind_coloring_max_items)
        if settings.mind_artifact_publish_enabled and bus is not None:
            await publish_mind_run_artifact_for_thought(
                bus,
                source=_source(),
                request=request,
                mind_req=mind_req,
                mind_res=result,
                channel=settings.channel_mind_artifact,
            )
        logger.info(
            "mind_enrichment corr=%s mind_run_id=%s quality=%s coloring=%s",
            request.correlation_id,
            result.mind_run_id,
            result.brief.mind_quality,
            "fired" if coloring else "skipped",
        )
        return coloring
    except Exception as exc:  # noqa: BLE001 — enrichment must never fail the turn
        logger.warning(
            "mind_enrichment_failed corr=%s reason=%s err=%s",
            request.correlation_id,
            type(exc).__name__,
            exc,
        )
        return None


async def run_stance_react(
    request: StanceReactRequestV1,
    *,
    bus: OrionBusAsync,
    cortex_client: CortexExecClient | None = None,
) -> ThoughtEventV1:
    client = cortex_client or CortexExecClient(bus)
    mind_coloring = await _maybe_build_mind_coloring(request, bus=bus)
    plan_request = build_stance_react_plan_request(request, mind_coloring=mind_coloring)
    exec_result = await client.execute_plan(
        source=_source(),
        req=plan_request,
        correlation_id=request.correlation_id,
        timeout_sec=settings.stance_react_timeout_sec,
    )
    raw_payload = extract_stance_react_payload(exec_result)
    thought = parse_stance_react_payload(
        raw_payload,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
    )
    enriched = apply_stance_react_pipeline(thought, request)
    capsule = _extract_grounding_capsule(exec_result)
    if capsule is not None:
        enriched = enriched.model_copy(update={"grounding_capsule": capsule})
    return enriched


async def handle_stance_react_request(
    bus: OrionBusAsync,
    request: StanceReactRequestV1,
    *,
    reply_to: str,
    correlation_id: str | None = None,
    causality_chain: list[str] | None = None,
) -> ThoughtEventV1:
    corr = correlation_id or request.correlation_id or str(uuid4())
    causality = list(causality_chain or [])
    thought = await run_stance_react(request, bus=bus)
    payload = thought.model_dump(mode="json")
    envelope = BaseEnvelope(
        kind="thought.event.v1",
        source=_source(),
        correlation_id=_envelope_correlation_id(corr),
        causality_chain=causality,
        payload=payload,
    )
    await bus.publish(reply_to, envelope)
    await bus.publish(settings.channel_thought_artifact, envelope)
    logger.info(
        "stance_react complete corr=%s reply=%s artifact=%s disposition=%s",
        corr,
        reply_to,
        settings.channel_thought_artifact,
        thought.disposition,
    )
    return thought


async def run_bus_worker(stop_event: asyncio.Event | None = None) -> None:
    if not settings.orion_bus_enabled:
        logger.info("Bus disabled; worker not started")
        return

    channel = settings.channel_thought_request
    backoff_sec = 1.0

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        bus = OrionBusAsync(url=settings.orion_bus_url)
        reconnect = False
        idle_polls = 0
        try:
            await bus.connect()
            logger.info("subscribed channel=%s", channel)
            async with bus.subscribe(channel) as pubsub:
                backoff_sec = 1.0
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        idle_polls += 1
                        if idle_polls >= _PUBSUB_IDLE_POLLS_BEFORE_HEALTH:
                            idle_polls = 0
                            subs = await _thought_channel_subscribers(bus, channel)
                            if subs == 0:
                                logger.warning(
                                    "pubsub subscription missing channel=%s; reconnecting",
                                    channel,
                                )
                                reconnect = True
                                break
                        continue
                    except (ConnectionError, OSError) as exc:
                        logger.warning(
                            "pubsub read failed channel=%s err=%s; reconnecting",
                            channel,
                            exc,
                        )
                        reconnect = True
                        break

                    if not msg or msg.get("type") not in ("message", "pmessage"):
                        continue
                    idle_polls = 0
                    try:
                        await _handle_bus_message(bus, msg)
                    except Exception:
                        logger.exception("unhandled bus worker error")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("bus worker disconnect channel=%s", channel)
            reconnect = True
        finally:
            with suppress(Exception):
                await bus.close()

        if stop_event is not None and stop_event.is_set():
            return
        if not reconnect:
            return
        await asyncio.sleep(backoff_sec)
        backoff_sec = min(backoff_sec * 2, 30.0)


async def _handle_bus_message(bus: OrionBusAsync, raw_msg: dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    reply_channel = env.reply_to or (env.payload or {}).get("reply_channel")
    if not reply_channel:
        logger.warning("missing reply_to corr=%s", env.correlation_id)
        return

    kind = env.kind or ""
    if kind not in ("stance.react.request.v1", "legacy.message"):
        logger.warning("unsupported kind=%s", kind)
        return

    corr = str(env.correlation_id or uuid4())
    payload = env.payload or {}
    causality = list(env.causality_chain or [])

    try:
        request = StanceReactRequestV1.model_validate(payload)
        if not request.correlation_id:
            request = request.model_copy(update={"correlation_id": corr})
        await handle_stance_react_request(
            bus,
            request,
            reply_to=reply_channel,
            correlation_id=corr,
            causality_chain=causality,
        )
    except Exception as exc:
        logger.error("stance_react error corr=%s err=%s", corr, exc)
        session_id = None
        try:
            session_id = StanceReactRequestV1.model_validate(payload).session_id
        except Exception:
            if isinstance(payload, dict):
                session_id = payload.get("session_id")
        failure = build_stance_react_failure_thought(
            correlation_id=corr,
            session_id=session_id if isinstance(session_id, str) else None,
            reason=f"stance_react_failed: {exc}",
        )
        err_envelope = env.derive_child(
            kind="thought.event.v1",
            source=_source(),
            payload=failure.model_dump(mode="json"),
            reply_to=None,
        )
        await bus.publish(reply_channel, err_envelope)
