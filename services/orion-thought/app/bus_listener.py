from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from orion.cognition.plan_loader import build_plan_for_verb
from orion.cognition.recall_query import cap_retrieval_query
from orion.core.bus.async_service import OrionBusAsync

from .rpc_health import fold_bus
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.llm.resource_lease import GPU_LEASE_ROUTE
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_STEP_CHANNEL,
    REVERIE_VISUAL_STEP_REQUEST_KIND,
    REVERIE_VISUAL_STEP_RESULT_KIND,
    ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.stance_context_prepare import (
    STANCE_CONTEXT_PREPARE_REQUEST_KIND,
    STANCE_CONTEXT_PREPARE_RESULT_PREFIX,
    STANCE_PREPARE_REQUESTED_CTX_KEY,
    StanceContextPrepareRequestV1,
    StanceContextPrepareResultV1,
    stance_context_prepare_channel,
)
from orion.schemas.thought import (
    AutonomySliceV1,
    GroundingCapsuleV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)
from orion.thought.coalition import prompt_turn_refs
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
from .visual_steps import run_visual_step

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


async def _missing_subscriptions(bus: OrionBusAsync, channels: tuple[str, ...]) -> list[str]:
    """Channels Redis no longer lists us on. A failed probe (-1) is not evidence of loss."""
    missing = []
    for channel in channels:
        if await _thought_channel_subscribers(bus, channel) == 0:
            missing.append(channel)
    return missing


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
        "attended_node_ids": prompt_turn_refs(
            list(broadcast.attended_node_ids), request.association.correlation_id
        ),
        "open_loop_ids": [loop.id for loop in broadcast.frame.open_loops],
        "broadcast_stale": request.association.broadcast_stale,
    }


def build_stance_react_context(
    request: StanceReactRequestV1,
    *,
    mind_coloring: dict[str, Any] | None = None,
    stance_prepare_requested: bool = False,
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
        # Top level as well as metadata: cortex-exec's router
        # (mark_orion_turn -> conversation phase), metacog traces and grammar
        # events read ctx["session_id"], not metadata's copy. With it only
        # nested, every unified turn's stance step recorded "Orion spoke"
        # under the shared "global" phase key and emitted traces with no
        # session. cortex-orch's _build_context sets it top level too.
        "session_id": request.session_id,
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
    # What recall searches for on both stance recalls (cortex-exec's
    # run_recall_step reads ctx["retrieval_query"]). Top-level ctx only, never
    # stance_inputs: stance_react.j2 renders every stance_inputs key into the
    # stance LLM prompt, and this is recall's search text, not stance context.
    retrieval_query = cap_retrieval_query(request.retrieval_query)
    if retrieval_query:
        context["retrieval_query"] = retrieval_query
    if mind_coloring is not None:
        context["mind_coloring"] = mind_coloring
    if stance_prepare_requested:
        # cortex-exec waits for (and uses) the context it is already building for
        # this turn instead of building a second one (app/stance_prepare.py).
        context[STANCE_PREPARE_REQUESTED_CTX_KEY] = True
    if request.gpu_lease is not None:
        # Stance is part of the already admitted turn: it runs under the turn's GPU pool hold (the
        # gateway attaches the call to it) and names the hold's work-class route, whatever the
        # caller preferred.
        context["gpu_lease"] = request.gpu_lease.model_dump(mode="json")
        context["llm_route"] = GPU_LEASE_ROUTE
        context["llm_lane"] = GPU_LEASE_ROUTE
    if request.gpu_lease is None and request.llm_route:
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
    stance_prepare_requested: bool = False,
) -> PlanExecutionRequest:
    """Build the cortex-exec plan request for the stance_react verb (one attempt, on the turn's own
    route and correlation id; which GPU serves it is orion-gpu-pool's decision)."""
    plan = build_plan_for_verb("stance_react", mode="brain")
    context = build_stance_react_context(
        request, mind_coloring=mind_coloring, stance_prepare_requested=stance_prepare_requested
    )
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
        context=context,
    )


async def execute_stance_react(
    request: StanceReactRequestV1,
    *,
    client: CortexExecClient,
    mind_coloring: dict[str, Any] | None = None,
    stance_prepare_requested: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    """Run the stance_react plan once, on the turn's own route and correlation id, with the whole
    STANCE_REACT_TIMEOUT_SEC budget.

    This used to try the agent lane briefly and then retry on the chat lane under a fresh
    correlation id (live 2026-09-19: a reading stance spent its whole budget queued behind a long
    agent turn while chat sat idle). Since 2026-09-24 every LLM call leases from orion-gpu-pool,
    which spills agent work onto any card the YAML allows -- the gpu2 agent seat, or chat's gpu0
    while it is lent -- and returns a typed "unavailable" instead of a silent shed. A second
    caller-side placement decision on top would only fight the pool, so it is gone.
    """
    plan_request = build_stance_react_plan_request(
        request, mind_coloring=mind_coloring, stance_prepare_requested=stance_prepare_requested
    )
    exec_result = await client.execute_plan(
        source=_source(),
        req=plan_request,
        correlation_id=request.correlation_id,
        timeout_sec=float(settings.stance_react_timeout_sec),
    )
    return exec_result, extract_stance_react_payload(exec_result)


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


# Bound on how long the background prepare RPC waits for cortex-exec's reply.
# The reply only feeds the overlap log; stance_react never waits on it here.
_STANCE_PREPARE_REPLY_TIMEOUT_SEC = 120.0


async def send_stance_context_prepare(
    request: StanceReactRequestV1,
    *,
    request_channel: str,
    bus: OrionBusAsync | None = None,
) -> StanceContextPrepareResultV1 | None:
    """Ask cortex-exec to build stance_react's context now (unified-turn latency L4).

    Sent on the prepare channel of the same exec lane stance_react will use
    (``request_channel``), so the cached context is on the container that serves
    stance_react. Own bus connection: it runs concurrently with the mind call
    and the stance RPC. Fail-open: any failure returns None and stance_react
    builds its own context after a short wait.
    """
    channel = stance_context_prepare_channel(request_channel)
    if channel is None:
        return None
    plan_request = build_stance_react_plan_request(request)
    payload = StanceContextPrepareRequestV1(
        correlation_id=request.correlation_id, plan_request=plan_request
    )
    reply_channel = f"{STANCE_CONTEXT_PREPARE_RESULT_PREFIX}:{uuid4()}"
    env = BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_REQUEST_KIND,
        source=_source(),
        correlation_id=_envelope_correlation_id(request.correlation_id),
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    own_bus = bus is None
    rpc_bus = bus or OrionBusAsync(url=settings.orion_bus_url)
    try:
        if own_bus:
            await rpc_bus.connect()
        msg = await rpc_bus.rpc_request(
            channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=_STANCE_PREPARE_REPLY_TIMEOUT_SEC,
        )
        decoded = rpc_bus.codec.decode(msg.get("data"))
        if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
            return None
        return StanceContextPrepareResultV1.model_validate(decoded.envelope.payload)
    except Exception as exc:  # noqa: BLE001 -- the prepare must never fail the turn
        logger.warning(
            "stance_prepare_rpc_failed corr=%s channel=%s err=%s: %s",
            request.correlation_id,
            channel,
            type(exc).__name__,
            exc,
        )
        return None
    finally:
        if own_bus:
            fold_bus(rpc_bus)
            with suppress(Exception):
                await rpc_bus.close()


def _start_stance_prepare(
    request: StanceReactRequestV1, client: CortexExecClient
) -> asyncio.Task[StanceContextPrepareResultV1 | None] | None:
    if not settings.stance_prepare_parallel_enabled:
        return None
    exec_channel = getattr(client, "request_channel", None)
    if not isinstance(exec_channel, str) or stance_context_prepare_channel(exec_channel) is None:
        logger.info(
            "stance_prepare_skipped corr=%s reason=no_prepare_channel exec_channel=%s",
            request.correlation_id,
            exec_channel,
        )
        return None
    return asyncio.create_task(
        send_stance_context_prepare(request, request_channel=exec_channel),
        name=f"stance-prepare-{request.correlation_id}",
    )


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
    # The stance_react lane is decided once, here: the prepare goes to the
    # prepare channel of client.request_channel, the stance RPC to that channel.
    prepare_task = _start_stance_prepare(request, client)
    mind_started = time.perf_counter()
    mind_ms: float | None = None
    exec_result: dict[str, Any] | None = None
    try:
        mind_coloring = await _maybe_build_mind_coloring(request, bus=bus)
        mind_ms = round((time.perf_counter() - mind_started) * 1000.0, 1)
        exec_result, raw_payload = await execute_stance_react(
            request,
            client=client,
            mind_coloring=mind_coloring,
            stance_prepare_requested=prepare_task is not None,
        )
    finally:
        if prepare_task is not None:
            # Also on a mind/stance failure or cancellation: never leave the
            # prepare RPC task (and its bus connection) orphaned.
            _log_stance_prepare_overlap(
                request, prepare_task, mind_ms=mind_ms, exec_result=exec_result
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


def _log_stance_prepare_overlap(
    request: StanceReactRequestV1,
    prepare_task: asyncio.Task[StanceContextPrepareResultV1 | None],
    *,
    mind_ms: float | None,
    exec_result: dict[str, Any] | None = None,
) -> None:
    """One line per prepared turn: mind time vs build time, and how long
    stance_react waited for the build. cortex-exec logs its own side
    (outcome, wait_ms) under the same prefix."""
    prepare: StanceContextPrepareResultV1 | None = None
    if prepare_task.done() and not prepare_task.cancelled():
        prepare = prepare_task.result()
    else:
        prepare_task.cancel()
    metadata = exec_result.get("metadata") if isinstance(exec_result, dict) else None
    overlap = metadata.get("stance_prepare_overlap") if isinstance(metadata, dict) else None
    overlap = overlap if isinstance(overlap, dict) else {}
    build_ms = prepare.build_ms if prepare is not None else None
    if build_ms is None:
        build_ms = overlap.get("build_ms")
    logger.info(
        "stance_prepare_overlap corr=%s side=orion-thought mind_ms=%s build_ms=%s wait_ms=%s "
        "outcome=%s prepare_status=%s",
        request.correlation_id,
        mind_ms,
        build_ms,
        overlap.get("wait_ms"),
        overlap.get("outcome") or ("stance_failed" if exec_result is None else "unreported"),
        prepare.status if prepare is not None else "no_reply",
    )


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


def _message_channel(raw_msg: dict[str, Any]) -> str:
    channel = raw_msg.get("channel")
    return channel.decode() if isinstance(channel, bytes) else str(channel or "")


_INVALID_STEP_RETRY_AFTER_SEC = 60.0


async def handle_visual_step_request(bus: OrionBusAsync, env: BaseEnvelope, *, reply_to: str) -> None:
    """One `reverie.visual` stage from orion-durable-runs; replies on `reply_to` under the
    request envelope's own correlation id (durable-runs fences replies by it)."""
    payload = env.payload or {}
    try:
        request = ReverieVisualStepRequestV1.model_validate(payload)
    except ValidationError as exc:
        logger.error("reverie visual step request invalid corr=%s err=%s", env.correlation_id, exc)
        try:
            # A retry, never terminal: during a rolling deploy the sender and this worker
            # can disagree on the schema, and that skew must not kill in-flight runs.
            # durable-runs fences replies by the envelope correlation, so fill what the
            # payload lacks.
            result = ReverieVisualStepResultV1(
                run_id=str(payload.get("run_id") or "unknown"),
                correlation_id=str(payload.get("correlation_id") or env.correlation_id),
                step=payload.get("step"), status="retry", reason="invalid_step_request",
                retry_after_sec=_INVALID_STEP_RETRY_AFTER_SEC,
            )
        except (ValidationError, AttributeError):
            logger.error("reverie visual step request unanswerable (no valid step) corr=%s reply_to=%s",
                         env.correlation_id, reply_to)
            return
    else:
        result = await run_visual_step(bus, request)
        logger.info("reverie visual step run=%s step=%s status=%s reason=%s elapsed=%s",
                    request.run_id, request.step, result.status, result.reason, result.elapsed_sec)
    await bus.publish(
        reply_to,
        BaseEnvelope(
            kind=REVERIE_VISUAL_STEP_RESULT_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=list(env.causality_chain or []),
            # exclude_none: a waking reply never carries the dream-only `caption: null`;
            # every result field defaults to None, so the receiver reads the same model.
            payload=result.model_dump(mode="json", exclude_none=True),
        ),
    )


async def run_bus_worker(stop_event: asyncio.Event | None = None) -> None:
    if not settings.orion_bus_enabled:
        logger.info("Bus disabled; worker not started")
        return

    channel = settings.channel_thought_request
    channels = (channel, REVERIE_VISUAL_STEP_CHANNEL)
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
            logger.info("subscribed channels=%s", ",".join(channels))
            async with bus.subscribe(*channels) as pubsub:
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
                            missing = await _missing_subscriptions(bus, channels)
                            if missing:
                                logger.warning(
                                    "pubsub subscription missing channels=%s; reconnecting",
                                    ",".join(missing),
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

    if _message_channel(raw_msg) == REVERIE_VISUAL_STEP_CHANNEL or env.kind == REVERIE_VISUAL_STEP_REQUEST_KIND:
        await handle_visual_step_request(bus, env, reply_to=reply_channel)
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
