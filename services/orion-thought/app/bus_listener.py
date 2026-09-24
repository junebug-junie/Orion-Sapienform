from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync

from .rpc_health import fold_bus
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.thought import (
    AutonomySliceV1,
    GroundingCapsuleV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)
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
    work_shape_from_coloring,
)
from .settings import settings

logger = logging.getLogger("orion-thought.bus")

# Poll get_message with a 1s timeout; after this many idle polls verify Redis still
# lists us as a subscriber. A silent pubsub disconnect leaves health checks green
# but PUBSUB NUMSUB returns 0 — hub RPC then hangs until timeout and the message
# is lost (pubsub is not durable).
_PUBSUB_IDLE_POLLS_BEFORE_HEALTH = 30
_HANDLER_DRAIN_TIMEOUT_SEC = 120.0

_pending_handler_tasks: set[asyncio.Task[None]] = set()


async def _run_bus_message_handler(raw_msg: dict[str, Any]) -> None:
    """Handle one request on a dedicated bus connection so pubsub can keep draining."""
    bus = OrionBusAsync(url=settings.orion_bus_url)
    try:
        await bus.connect()
        await _handle_bus_message(bus, raw_msg)
    except Exception:
        logger.exception("bus message handler failed")
    finally:
        # Per-call bus: fold its RPC-health window before discarding it (app/rpc_health.py).
        fold_bus(bus)
        with suppress(Exception):
            await bus.close()


def _dispatch_bus_message(raw_msg: dict[str, Any]) -> None:
    task = asyncio.create_task(_run_bus_message_handler(raw_msg))
    _pending_handler_tasks.add(task)
    task.add_done_callback(_pending_handler_tasks.discard)


async def _drain_pending_handler_tasks(*, timeout_sec: float = _HANDLER_DRAIN_TIMEOUT_SEC) -> None:
    if not _pending_handler_tasks:
        return
    pending = set(_pending_handler_tasks)
    done, still = await asyncio.wait(pending, timeout=timeout_sec)
    if still:
        for task in still:
            task.cancel()
        await asyncio.gather(*still, return_exceptions=True)
    _ = done


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
    stance_inputs = (
        dict(request.stance_inputs)
        if isinstance(request.stance_inputs, dict)
        else {"user_message": request.user_message}
    )
    surface_context = stance_inputs.get("surface_context")
    metadata: dict[str, Any] = {
        "correlation_id": request.correlation_id,
        "session_id": request.session_id,
        "llm_profile": request.llm_profile,
        "mode": "brain",
    }
    if isinstance(surface_context, dict) and surface_context:
        metadata["surface_context"] = surface_context
    context: dict[str, Any] = {
        "user_message": request.user_message,
        "stance_inputs": stance_inputs,
        "association": slim_association_for_prompt(request.association),
        "repair_bundle": slim_repair_bundle_for_prompt(request.repair_bundle),
        "coalition_projection": _coalition_projection(request),
        "metadata": metadata,
        # Root cause of the recurring "stance_react exec result missing thought
        # payload" deferred turn (confirmed live 2026-09-10, corr=9c7e9272):
        # services/orion-cortex-exec/app/router.py's _structured_output_expected()
        # already treats "stance_react" as JSON-required and REJECTS a reply that
        # isn't parseable JSON (router.py:393-394) -- but nothing on the request
        # side ever told the gateway to actually constrain the model to JSON. The
        # model is free to just answer in prose, which is exactly what happened:
        # a good, on-topic 348-char reply got discarded whole because it wasn't a
        # JSON object, producing an empty final_text and this deferred turn.
        # `{"type": "json_object"}` is the same minimal llama.cpp/vLLM JSON-mode
        # constraint executor.py's MetacogDraftService branch already uses
        # successfully (executor.py:3324) -- reusing it here, not inventing a new
        # mechanism. This dict is forwarded verbatim into gateway_options by
        # executor.py's existing `ctx.get("response_format")` forwarding
        # (executor.py:4365-4367); no executor.py change is needed.
        "response_format": {"type": "json_object"},
    }
    if isinstance(surface_context, dict) and surface_context:
        context["surface_context"] = surface_context
    if mind_coloring is not None:
        context["mind_coloring"] = mind_coloring
    if request.resource_lease is not None:
        # Stance is part of the already admitted turn, including when admission
        # assigned a different lane from the original caller's preference.
        context["resource_lease"] = request.resource_lease.model_dump(mode="json")
        context["llm_route"] = request.resource_lease.lane
        context["llm_lane"] = request.resource_lease.lane
    elif request.llm_route:
        # Caller-requested gateway route override for stance_react's own LLM
        # call (see StanceReactRequestV1.llm_route's own docstring -- today
        # only orion.hub.turn_orchestrator's agent-lane resolution sets
        # this). Top-level "llm_route" is one of the two keys
        # services/orion-cortex-exec/app/executor.py's
        # _resolve_llm_route_override reads before falling back to
        # _default_llm_route_for_step's hardcoded "chat" default for this
        # verb.
        context["llm_route"] = request.llm_route
    return context


def build_stance_react_plan_request(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
    llm_route_override: str | None = None,
    step_timeout_cap_sec: float | None = None,
    request_id: str | None = None,
) -> PlanExecutionRequest:
    """Build the cortex-exec plan request for one stance_react attempt.

    ``llm_route_override`` replaces the context's gateway route for this attempt
    only (never when the request carries a resource_lease -- admission owns the
    lane then). ``step_timeout_cap_sec`` caps every step's ``timeout_ms``: the
    plan travels on the wire, and cortex-exec enforces the step's own
    ``timeout_ms`` as the real LLM-call cutoff (executor.py's
    ``step_timeout_sec = (step.timeout_ms or 60000) / 1000.0``) and forwards
    ``timeout_ms - 5s`` (floor 45s) to the gateway as the caller budget the
    capacity wait is bounded by -- so this is what actually bounds a short
    agent-lane attempt, not this service's RPC wait. ``request_id`` stamps the
    plan args for a fallback attempt that runs under a fresh correlation id.
    """
    plan = build_plan_for_verb("stance_react", mode="brain")
    if step_timeout_cap_sec is not None:
        cap_ms = max(1, int(step_timeout_cap_sec * 1000))
        plan = plan.model_copy(
            update={
                "steps": [
                    step.model_copy(update={"timeout_ms": min(int(step.timeout_ms), cap_ms)})
                    for step in plan.steps
                ]
            }
        )
    context = build_stance_react_context(request, mind_coloring=mind_coloring)
    if llm_route_override and request.resource_lease is None:
        context["llm_route"] = llm_route_override
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=request_id or request.correlation_id,
            trigger_source=settings.service_name,
            extra={
                "llm_profile": request.llm_profile,
                "mode": "brain",
            },
        ),
        context=context,
    )


# Wall-clock slack this service's RPC wait allows around a capped LLM step, for
# cortex-exec's own work either side of the gateway call (prompt render, JSON
# validation, grammar/trace publish -- 32s measured once on an 11 KB prompt,
# services/orion-thought/.env_example). The agent attempt's RPC wait is
# `agent budget + this`; the chat attempt's step cap is `remaining - this`.
STANCE_REACT_ATTEMPT_RPC_MARGIN_SEC = 30.0
# cortex-exec floors the gateway read timeout at 45s (executor.py's
# `max(45.0, min(timeout - 5.0, 900.0))`); a chat fallback with less budget
# than that would only generate for a caller that has already given up.
STANCE_REACT_MIN_STEP_TIMEOUT_SEC = 45.0


def lane_fallback_applies(request: StanceReactRequestV1, *, agent_lane_budget_sec: float) -> bool:
    """Whether this request gets the bounded agent attempt + chat fallback.

    Only turns that PREFER the agent lane without OWNING it: an explicit
    ``llm_route="agent"`` (orion.hub.turn_orchestrator sets this for every
    agent-model turn, i.e. autonomous reading and curiosity) and no
    ``resource_lease`` (a durable lease means admission already reserved the
    lane -- there is nothing to fall back from). A caller that runs its own
    caller-side lane fallback (endogenous outreach, PR #2163) opts out via
    ``caller_handles_lane_fallback`` so one outreach tick cannot stack four
    stance attempts. A non-positive budget disables the fallback outright.
    """
    return (
        request.llm_route == "agent"
        and request.resource_lease is None
        and not request.caller_handles_lane_fallback
        and agent_lane_budget_sec > 0
    )


async def _run_stance_react_attempt(
    request: StanceReactRequestV1,
    *,
    client: CortexExecClient,
    mind_coloring: dict[str, Any] | None,
    lane: str,
    llm_route_override: str | None,
    correlation_id: str,
    step_timeout_cap_sec: float | None,
    rpc_timeout_sec: float,
) -> tuple[tuple[dict[str, Any], dict[str, Any] | str] | None, str | None]:
    """One cortex-exec round trip. Returns ((exec_result, raw_payload), None) on
    success or (None, reason) on any failure -- RPC timeout, decode error, a
    named step failure, or an empty payload."""
    started = asyncio.get_running_loop().time()
    try:
        plan_request = build_stance_react_plan_request(
            request,
            mind_coloring=mind_coloring,
            llm_route_override=llm_route_override,
            step_timeout_cap_sec=step_timeout_cap_sec,
            request_id=correlation_id,
        )
        exec_result = await client.execute_plan(
            source=_source(),
            req=plan_request,
            correlation_id=correlation_id,
            timeout_sec=rpc_timeout_sec,
        )
        raw_payload = extract_stance_react_payload(exec_result)
    except Exception as exc:  # noqa: BLE001 -- every failure shape is a fallback trigger
        reason = str(exc).strip() or type(exc).__name__
        logger.warning(
            "stance_react_attempt corr=%s attempt_corr=%s lane=%s step_cap_sec=%s rpc_sec=%.0f "
            "elapsed=%.1fs outcome=failed reason=%s",
            request.correlation_id,
            correlation_id,
            lane,
            step_timeout_cap_sec,
            rpc_timeout_sec,
            asyncio.get_running_loop().time() - started,
            reason,
        )
        return None, reason
    logger.info(
        "stance_react_attempt corr=%s attempt_corr=%s lane=%s step_cap_sec=%s rpc_sec=%.0f "
        "elapsed=%.1fs outcome=ok",
        request.correlation_id,
        correlation_id,
        lane,
        step_timeout_cap_sec,
        rpc_timeout_sec,
        asyncio.get_running_loop().time() - started,
    )
    return (exec_result, raw_payload), None


async def execute_stance_react_with_lane_fallback(
    request: StanceReactRequestV1,
    *,
    client: CortexExecClient,
    mind_coloring: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    """Run the stance_react plan; for agent-preferring turns without a lease,
    try the agent lane briefly, then the chat lane once.

    Why (live 2026-09-19, corr 5258cae8): both GPU lanes are single-slot. A
    reading/curiosity stance asked for the agent lane and waited 235s behind a
    long autonomous agent turn, until the gateway shed it
    (``capacity_wait_budget_exhausted``) -- the verb's whole 240s budget spent
    queueing, nothing generated, and the chat lane idle the entire time.
    Outreach already had exactly this two-attempt shape caller-side (PR #2163);
    reading and curiosity got the agent preference without the bounded attempt
    or the fallback. This is the one seam every stance RPC passes through.

    Attempt 1 (agent): step cap ``STANCE_REACT_AGENT_LANE_BUDGET_SEC``, RPC
    wait ``cap + STANCE_REACT_ATTEMPT_RPC_MARGIN_SEC``, the turn's own
    correlation id. Attempt 2 (chat): only on failure, on a FRESH correlation
    id (the gateway keys capacity acquires by correlation id, and the abandoned
    attempt may still be queued there under the original), with whatever is
    left of ``STANCE_REACT_TIMEOUT_SEC``. Both fail -> one ValueError naming
    both reasons, e.g. ``agent=gateway_capacity_rejected:
    capacity_wait_budget_exhausted; chat=...``.
    """
    total_budget = float(settings.stance_react_timeout_sec)
    agent_budget = float(settings.stance_react_agent_lane_budget_sec)
    if 0 < agent_budget < STANCE_REACT_MIN_STEP_TIMEOUT_SEC:
        # Review finding (2026-09-19): cortex-exec floors its own gateway read
        # timeout at 45s (executor.py's `max(45.0, min(timeout - 5.0, 900.0))`)
        # regardless of what this service asks for. A configured budget below
        # that floor would make cortex-exec's RPC wait shorter than the
        # generation time it simultaneously tells the gateway it can use --
        # this call would time out here before the gateway's own shed/serve
        # decision could land. Clamp up to the floor instead of honoring a
        # misconfiguration that can only ever look like a spurious timeout.
        logger.warning(
            "stance_react_agent_lane_budget_below_gateway_floor configured=%.1fs floor=%.1fs "
            "-- clamping up",
            agent_budget,
            STANCE_REACT_MIN_STEP_TIMEOUT_SEC,
        )
        agent_budget = STANCE_REACT_MIN_STEP_TIMEOUT_SEC
    if not lane_fallback_applies(request, agent_lane_budget_sec=agent_budget):
        plan_request = build_stance_react_plan_request(request, mind_coloring=mind_coloring)
        exec_result = await client.execute_plan(
            source=_source(),
            req=plan_request,
            correlation_id=request.correlation_id,
            timeout_sec=total_budget,
        )
        return exec_result, extract_stance_react_payload(exec_result)

    loop = asyncio.get_running_loop()
    started = loop.time()
    agent_cap = min(agent_budget, total_budget)
    outcome, agent_reason = await _run_stance_react_attempt(
        request,
        client=client,
        mind_coloring=mind_coloring,
        lane="agent",
        llm_route_override="agent",
        correlation_id=request.correlation_id,
        step_timeout_cap_sec=agent_cap,
        rpc_timeout_sec=min(agent_cap + STANCE_REACT_ATTEMPT_RPC_MARGIN_SEC, total_budget),
    )
    if outcome is not None:
        return outcome

    remaining = total_budget - (loop.time() - started)
    if remaining <= STANCE_REACT_MIN_STEP_TIMEOUT_SEC:
        raise ValueError(f"agent={agent_reason}; chat=skipped:budget_remaining={remaining:.0f}s")
    chat_cap = max(STANCE_REACT_MIN_STEP_TIMEOUT_SEC, remaining - STANCE_REACT_ATTEMPT_RPC_MARGIN_SEC)
    fallback_correlation_id = str(uuid4())
    logger.info(
        "stance_react_lane_fallback corr=%s fallback_corr=%s from=agent to=chat agent_reason=%s "
        "remaining_sec=%.0f",
        request.correlation_id,
        fallback_correlation_id,
        agent_reason,
        remaining,
    )
    outcome, chat_reason = await _run_stance_react_attempt(
        request,
        client=client,
        mind_coloring=mind_coloring,
        lane="chat",
        llm_route_override="chat",
        correlation_id=fallback_correlation_id,
        step_timeout_cap_sec=chat_cap,
        rpc_timeout_sec=remaining,
    )
    if outcome is not None:
        return outcome
    raise ValueError(f"agent={agent_reason}; chat={chat_reason}")


MISSING_THOUGHT_PAYLOAD = "stance_react exec result missing thought payload"


def exec_failure_reason(result: dict[str, Any]) -> str | None:
    """The named reason cortex-exec gave for a failed plan, if it gave one.

    cortex-exec's PlanRunner sets ``error`` on a non-success plan to the last
    step's own ``error`` (services/orion-cortex-exec/app/router.py); since
    2026-09-19 a shed/refused gateway reply fails its step as e.g.
    ``gateway_capacity_rejected:capacity_wait_budget_exhausted`` (executor.py's
    ``gateway_error_step_failure``). Before that, the same outcome arrived as a
    *successful* plan with empty content, and the only thing this service could
    say was "missing thought payload" -- which is what 21 of the last 30
    world-pulse Stage 1 reads died with while the real cause sat in the
    gateway's log. Plan-level ``error`` wins; a failed step's ``error`` is the
    fallback for a result shape that carried steps but no top-level error.
    """
    if not isinstance(result, dict):
        return None
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    for step in reversed(result.get("steps") or []):
        if not isinstance(step, dict) or step.get("status") == "success":
            continue
        step_error = step.get("error")
        if isinstance(step_error, str) and step_error.strip():
            return step_error.strip()
    return None


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

    # No payload anywhere. If cortex-exec named why, say that -- the deferred
    # turn's label becomes `stance_react_failed: <that reason>` (see
    # _handle_bus_message) instead of the generic line below.
    reason = exec_failure_reason(result)
    if reason:
        raise ValueError(reason)
    raise ValueError(MISSING_THOUGHT_PAYLOAD)


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


def _extract_autonomy_slice(exec_result: dict[str, Any]) -> AutonomySliceV1 | None:
    metadata = exec_result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("autonomy_slice")
    if not isinstance(raw, dict):
        return None
    try:
        return AutonomySliceV1.model_validate(raw)
    except Exception:
        logger.warning("autonomy_slice_parse_failed corr=%s", exec_result.get("request_id"))
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
        coloring = select_mind_coloring(
            result,
            max_items=settings.mind_coloring_max_items,
            utterance_origin=mind_req.utterance_origin,
        )
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
    """Orion capability: stance/thought assembly for the unified turn.

    Produces the ThoughtEventV1 that constrains the FCC motor: executes the
    stance_react Cortex plan (a dynamic, bus-mediated edge invisible to static
    call graphs), optionally colors the request with Mind enrichment, then
    normalizes the payload and enriches it with the grounding capsule and
    autonomy slice. Hub honors the resulting defer/refuse disposition before
    any motor work; the imperative and stance_harness_slice shape the motor
    prompt.

    Runtime evidence: thought.event.v1 envelopes on the RPC reply and thought
    artifact channels, with disposition logged. Start here when a turn's
    imperative or stance slice looks wrong before the motor ever ran.
    """
    client = cortex_client or CortexExecClient(bus)
    mind_coloring = await _maybe_build_mind_coloring(request, bus=bus)
    exec_result, raw_payload = await execute_stance_react_with_lane_fallback(
        request, client=client, mind_coloring=mind_coloring
    )
    thought = parse_stance_react_payload(
        raw_payload,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
    )
    enriched = apply_stance_react_pipeline(thought, request)
    capsule = _extract_grounding_capsule(exec_result)
    if capsule is not None:
        enriched = enriched.model_copy(update={"grounding_capsule": capsule})
    slice_ = _extract_autonomy_slice(exec_result)
    if slice_ is not None:
        enriched = enriched.model_copy(update={"autonomy_slice": slice_})
    enriched = enriched.model_copy(
        update={"mind_work_shape": work_shape_from_coloring(mind_coloring)}
    )
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
            await asyncio.shield(_drain_pending_handler_tasks())
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
                        break
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
                        _dispatch_bus_message(msg)
                    except Exception:
                        logger.exception("unhandled bus worker error")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("bus worker disconnect channel=%s", channel)
            reconnect = True
        finally:
            if stop_event is not None and stop_event.is_set():
                await asyncio.shield(_drain_pending_handler_tasks())
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
