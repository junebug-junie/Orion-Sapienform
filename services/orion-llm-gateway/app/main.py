from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from pydantic import ValidationError

# [FIX] Added ServiceRef to imports
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, Envelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1
from orion.schemas.vector.schemas import VectorUpsertV1

from orion.gpu_pool.client import Lease, LeaseUnavailable

from .ctx_overflow import CONTEXT_OVERFLOW_ERROR
from .llm_backend import (
    ChatDispatchPlan,
    RouteTarget,
    plan_llm_chat,
    resolve_caller_budget_sec,
    run_llm_chat,
)
from .anthropic_passthrough import register_anthropic_passthrough_routes
from .openai_passthrough import register_openai_passthrough_routes
from . import pool_placement, upstream_cancel
from .embed_publish import publish_assistant_embedding
from .models import ChatBody
from .resource_lease import LeaseGuard, ResourceLeaseRejected
from .settings import settings

logger = logging.getLogger("orion-llm-gateway")
bus_handle: Optional[Any] = None
app = FastAPI()
app.include_router(register_anthropic_passthrough_routes())
app.include_router(register_openai_passthrough_routes())


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


@app.get("/health")
async def health() -> Dict[str, Any]:
    routes = sorted(pool_placement.pool_routes().keys())
    return {
        "status": "ok",
        "service": settings.service_name,
        "node": settings.node_name,
        "routes": routes,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    if bus_handle is None or not getattr(bus_handle, "enabled", False):
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_llm_intake,
            subscriber_count=0,
            dependency_status="unavailable",
            error="bus not connected",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(bus_handle, "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_llm_intake,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=settings.channel_llm_intake,
        service_name=settings.service_name,
        heartbeat_ttl_sec=float(settings.heartbeat_interval_sec) * 3.0,
        check_heartbeat=False,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    pool_bus_up = pool_placement.pool_bus_ready()
    if not pool_bus_up:
        # Every LLM call leases over the forked pool RPC client; without it each one answers
        # gpu_pool_unavailable (pool_bus_unavailable), so the gateway is not ready to serve.
        body = body.model_copy(update={
            "ok": False, "dependency_status": "unavailable",
            "error": "; ".join(e for e in (body.error, "gpu_pool_bus_unavailable") if e),
        })
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code,
                        headers={"X-Gpu-Pool-Bus": "up" if pool_bus_up else "down"})


@app.get("/routes")
async def routes_catalog() -> Dict[str, Any]:
    """Compatibility view generated from orion-gpu-pool state (pool_placement.build_routes_compat).
    Kept until durable-runs, fcc_motor, situational context, context-exec and the Hub read pool
    state directly; removed in stage 6 of the GPU pool spec."""
    return await pool_placement.get_routes_payload()


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=getattr(settings, "node_name", None),
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(getattr(settings, "heartbeat_interval_sec", 10.0) or 10.0),
    )


# [FIX] Helper to replace the missing .service_ref() method
def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=getattr(settings, "node_name", None),
        version=settings.service_version,
    )


async def _maybe_publish_latent_upsert(
    *,
    env: BaseEnvelope,
    spark_vector: Optional[list[float]],
    backend: Optional[str],
    model_used: Optional[str],
    session_id: Optional[str],
    user_id: Optional[str],
) -> None:
    if not spark_vector:
        return
    if backend not in ("vllm", "llama-cola"):
        return
    if not bus_handle or not getattr(bus_handle, "enabled", False):
        logger.warning("Latent upsert skipped: bus unavailable.")
        return

    doc_id = str(env.correlation_id or env.id)
    meta: Dict[str, Any] = {
        "source_service": env.source.name,
        "original_channel": settings.channel_llm_intake,
        "role": "assistant",
        "timestamp": env.created_at.isoformat() if env.created_at else None,
        "correlation_id": str(env.correlation_id),
        "backend": backend,
        "model_used": model_used,
        "session_id": session_id,
        "user_id": user_id,
        "envelope_id": str(env.id),
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    upsert = VectorUpsertV1(
        doc_id=doc_id,
        collection=settings.orion_vector_latent_collection,
        embedding=spark_vector,
        embedding_kind="latent",
        embedding_model=model_used,
        embedding_dim=len(spark_vector),
        text=None,
        meta=meta,
    )
    envelope = BaseEnvelope(
        kind="vector.upsert.v1",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=upsert.model_dump(mode="json"),
    )
    try:
        await bus_handle.publish(settings.channel_vector_latent_upsert, envelope)
    except Exception as exc:
        logger.warning("Latent upsert publish failed doc_id=%s error=%s", doc_id, exc)


class _UpstreamFailed(Exception):
    """Raised inside the lease block so the pool records ``upstream_error``; carries the result
    the caller still gets back unchanged."""

    def __init__(self, result: Dict[str, Any]):
        super().__init__(str((result.get("raw") or {}).get("error") or result.get("text") or "upstream_error")[:300])
        self.result = result


class _ContextOverflow(_UpstreamFailed):
    def __init__(self, result: Dict[str, Any], ctx_per_slot: Optional[int]):
        super().__init__(result)
        self.ctx_per_slot = ctx_per_slot


def _result_error(result: Dict[str, Any]) -> Optional[str]:
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    vision = raw.get("vision")
    if isinstance(vision, dict) and vision.get("status") in ("refused", "fetch_failed"):
        return None  # refused before any generation: not the GPU's failure
    if raw.get("error"):
        return str(raw.get("error"))
    if str(result.get("text") or "").startswith("[Error:"):
        return "upstream_error"
    return None


def _pool_unavailable_result(plan: ChatDispatchPlan, reason: str) -> Dict[str, Any]:
    """Same shape as every early gateway error: empty text, `raw.error` set. Consumers that check
    for empty content treat it as a failed call."""
    return {
        "text": "",
        "content": "",
        "spark_meta": {},
        "raw": {
            "error": pool_placement.POOL_UNAVAILABLE,
            "details": {"reason": reason, "route": plan.route, "work_class": plan.work_class},
        },
        "route": plan.route,
        "served_by": None,
    }


def _revoked_result(plan: ChatDispatchPlan, lease: Lease, reason: str) -> Dict[str, Any]:
    """The pool took the slot back mid-call and the upstream was stopped. Empty content, raw.error
    set: consumers that check for empty content treat it as a failed call."""
    return {
        "text": "",
        "content": "",
        "spark_meta": {},
        "raw": {
            "error": pool_placement.POOL_RECALLED,
            "details": {"reason": f"lease_{reason}", "lease_id": lease.lease_id, "route": plan.route},
        },
        "route": plan.route,
        "served_by": lease.grant.served_by,
    }


def _deadline_result(plan: ChatDispatchPlan, lease: Lease, budget_s: float) -> Dict[str, Any]:
    return {
        "text": "",
        "content": "",
        "spark_meta": {},
        "raw": {
            "error": "timeout",
            "details": {"reason": "caller_budget_exhausted", "budget_sec": round(budget_s, 3), "route": plan.route},
        },
        "route": plan.route,
        "served_by": lease.grant.served_by,
    }


async def _run_on_grant(plan: ChatDispatchPlan, lease: Lease, read_timeout_s: float) -> Dict[str, Any]:
    """Run the sync call in the granted role's thread, and stop it from here when the pool takes
    the lease back (lost, or recalled past its grace), the caller's budget runs out, or this task is
    cancelled. Stopping = shutting down the worker's upstream sockets (upstream_cancel.py): the
    blocked read fails, the worker returns, and the slot the pool now counts as free is free."""
    grant = lease.grant
    run_body = plan.body.model_copy(update={"options": {
        **(plan.body.options or {}), "gateway_read_timeout_sec": read_timeout_s,
    }})
    run_plan = dataclasses.replace(
        plan, body=run_body,
        route_target=RouteTarget(url=grant.url, backend=pool_placement.LLAMACPP_BACKEND, served_by=grant.served_by),
    )
    loop = asyncio.get_running_loop()
    handle = upstream_cancel.UpstreamCancel()
    future = loop.run_in_executor(pool_placement.executor_for(grant.url), upstream_cancel.run_cancellable,
                                  handle, run_llm_chat, run_body, run_plan)
    watch = asyncio.ensure_future(pool_placement.wait_lease_revoked(lease))
    # The upstream client floors its read timeout at 30s; the caller may have less. Hold the whole
    # call to what is left of the caller's budget.
    budget = asyncio.ensure_future(asyncio.sleep(max(0.0, read_timeout_s)))
    try:
        try:
            await asyncio.wait({future, watch, budget}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The caller has gone: stop the upstream, keep the lease until the thread is out so the
            # pool's view of the role's busy slots stays true, then let the cancellation through.
            handle.cancel("caller_cancelled")
            await asyncio.wait({future})
            raise
        if future.done():
            return future.result()
        if watch.done():
            reason = watch.result()
            handle.cancel(f"lease_{reason}")
            logger.warning("gpu_pool_lease_revoked_mid_call lease_id=%s route=%s reason=%s served_by=%s",
                           lease.lease_id, plan.route, reason, grant.served_by)
            result = await asyncio.shield(future)
            if _result_error(result) is None:
                return result  # it finished before the sockets went down
            pool_placement.mark_revoked(lease, reason)
            return _revoked_result(plan, lease, reason)
        handle.cancel("caller_budget_exhausted")
        logger.warning("gateway_caller_budget_exhausted route=%s served_by=%s budget_sec=%.1f",
                       plan.route, grant.served_by, read_timeout_s)
        result = await asyncio.shield(future)
        if _result_error(result) is None:
            return result
        lease.release_outcome, lease.release_detail = "timeout", "caller_budget_exhausted"
        return _deadline_result(plan, lease, read_timeout_s)
    finally:
        watch.cancel()
        budget.cancel()


async def _dispatch_chat(body: ChatBody, *, correlation_id: str, holder: str = "llm-gateway") -> Dict[str, Any]:
    plan = plan_llm_chat(body)
    if plan.error is not None:
        return dict(plan.error)
    # A durable-run lease is validated first (don't take a GPU for a stale token), then kept
    # checked for the whole call. It is admission only: placement is the pool lease below.
    guard = LeaseGuard((body.options or {}).get("resource_lease"), lane=plan.route)
    try:
        await guard.check()
        return await guard.run(_dispatch_on_pool(plan, correlation_id=correlation_id, holder=holder))
    except ResourceLeaseRejected as exc:
        logger.warning("resource_lease_rejected correlation_id=%s reason=%s", correlation_id, exc)
        return {"text": "", "content": "", "route": plan.route,
                "raw": {"error": "resource_lease_rejected", "details": {"reason": str(exc)}}}


async def _dispatch_on_pool(plan: ChatDispatchPlan, *, correlation_id: str, holder: str) -> Dict[str, Any]:
    """Lease -> run on the granted URL -> release. At most three acquires, never a loop:

    * the pool refuses a prompt bigger than every role of its class (``min_ctx_exceeds_class:<max>``)
      -> re-acquire ONCE at the class's largest ctx, so llama.cpp answers with its real overflow;
    * the upstream overflows -> release and re-lease ONCE at that role's ctx_per_slot + 1. If the
      pool then says nothing that big exists, the overflow itself is the answer -- never
      gpu_pool_unavailable, and never a second clamp. An overflow on the already-clamped (largest)
      role is returned as is.

    One deadline for the whole stay: the caller's own budget (`resolve_caller_budget_sec`). The
    pool wait is capped by it (and by LLM_GATEWAY_POOL_[BACKGROUND_]WAIT_SEC), and whatever is
    left after the grant bounds the upstream call, so a call that queued for most of its budget is
    not then given a fresh budget to generate for a caller that has gone.
    """
    budget_s = resolve_caller_budget_sec(plan.body)
    deadline = time.monotonic() + budget_s
    options = plan.body.options or {}
    estimate = pool_placement.estimate_min_ctx_tokens(plan.body.messages or [], options.get("max_tokens"))
    min_ctx = estimate
    overflow: Optional[Dict[str, Any]] = None
    clamped = False
    for _ in range(3):
        remaining = deadline - time.monotonic()
        wait_s = min(pool_placement.wait_budget_sec(plan.priority or "system"), remaining)
        try:
            async with pool_placement.lease_for_route(
                plan.route, holder=holder, turn_correlation_id=correlation_id,
                min_ctx_tokens=min_ctx, deadline_sec=wait_s,
            ) as lease:
                read_timeout_s = deadline - time.monotonic()
                if read_timeout_s <= 0:
                    # The grant came after the caller's budget ran out: generate nothing.
                    return overflow or _pool_unavailable_result(plan, "deadline")
                result = await _run_on_grant(plan, lease, read_timeout_s)
                error = _result_error(result)
                if error == CONTEXT_OVERFLOW_ERROR and overflow is None and not clamped:
                    raise _ContextOverflow(result, lease.grant.ctx_per_slot)
                if error is not None:
                    raise _UpstreamFailed(result)
                return result
        except _ContextOverflow as exc:
            overflow = exc.result
            min_ctx = int(exc.ctx_per_slot or estimate) + 1
            logger.warning(
                "gpu_pool_context_overflow correlation_id=%s route=%s ctx_per_slot=%s -> re-lease min_ctx=%s",
                correlation_id, plan.route, exc.ctx_per_slot, min_ctx,
            )
            continue
        except _UpstreamFailed as exc:
            return exc.result
        except LeaseUnavailable as exc:
            if overflow is not None:
                # Nothing bigger could take it: the honest answer is the overflow itself.
                return overflow
            max_ctx = pool_placement.class_max_ctx(exc.reason)
            if max_ctx is not None and not clamped and max_ctx < min_ctx:
                # Bigger than every role of the class by the estimate: place it on the biggest one
                # anyway and let llama.cpp's own tokenizer decide (chars/4 is only a guess).
                logger.warning(
                    "gpu_pool_min_ctx_exceeds_class correlation_id=%s route=%s estimate=%s -> clamp min_ctx=%s",
                    correlation_id, plan.route, min_ctx, max_ctx,
                )
                clamped, min_ctx = True, max_ctx
                continue
            logger.warning(
                "gpu_pool_unavailable correlation_id=%s route=%s class=%s reason=%s",
                correlation_id, plan.route, plan.work_class, exc.reason,
            )
            return _pool_unavailable_result(plan, exc.reason)
    return overflow or _pool_unavailable_result(plan, "overflow_retry_exhausted")


async def handle_chat(env: BaseEnvelope) -> BaseEnvelope:
    if env.kind not in ("llm.chat.request", "legacy.message"):
        return BaseEnvelope(
            kind="system.error",
            source=_source(),  # [FIX]
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any] = {}
    if env.kind == "legacy.message":
        raw = env.payload if isinstance(env.payload, dict) else {}
        if raw.get("event") == "chat":
            payload_obj = raw.get("payload") or raw.get("body") or {}
        else:
            payload_obj = raw
    else:
        payload_obj = env.payload if isinstance(env.payload, dict) else {}

    try:
        typed_req = Envelope[ChatRequestPayload].model_validate(
            {**env.model_dump(), "kind": "llm.chat.request", "payload": payload_obj}
        )
    except ValidationError as ve:
        return BaseEnvelope(
            kind="llm.chat.result",
            source=_source(),  # [FIX]
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    req_options = typed_req.payload.options or {}
    body = ChatBody(
        model=typed_req.payload.model,
        messages=[m.model_dump() for m in typed_req.payload.messages],
        raw_user_text=typed_req.payload.raw_user_text,
        options=req_options,
        attachments=list(typed_req.payload.attachments or []),
        profile_name=typed_req.payload.profile,
        route=typed_req.payload.route,
        trace_id=str(typed_req.correlation_id),
        user_id=typed_req.payload.user_id,
        session_id=typed_req.payload.session_id,
        source=typed_req.source.name,
        verb=str(req_options.get("verb") or "").strip() or None,
    )
    messages = body.messages or []
    memory_marker = "RELEVANT MEMORY"
    marked_message = next(
        (m for m in messages if memory_marker in str(getattr(m, "content", "") or "")), None
    )
    combined_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    fallback_message = next(
        (m for m in messages if str(getattr(m, "role", "") or "").lower() == "user"), None
    )
    snippet_source = marked_message or fallback_message or (messages[0] if messages else None)
    snippet = str(getattr(snippet_source, "content", "") or "")[:160]
    mind_phase = req_options.get("mind_phase")
    logger.info(
        "gateway_llm_request_received event=gateway_llm_request_received correlation_id=%s route=%s "
        "reply_to=%s mind_phase=%s request_source=%s msgs_count=%s",
        typed_req.correlation_id,
        typed_req.payload.route,
        env.reply_to,
        mind_phase,
        typed_req.source.name if typed_req.source else None,
        len(messages),
    )

    holder = (typed_req.source.name if typed_req.source else None) or "llm-gateway"
    result = await _dispatch_chat(body, correlation_id=str(typed_req.correlation_id), holder=holder)
    text = result.get("text") if isinstance(result, dict) else str(result)

    # Optional Spark/NeuralHost enrichments. These may be absent depending on
    # which gateway instance handled the request.
    spark_meta = (result.get("spark_meta") if isinstance(result, dict) else None) or {}
    spark_vector = (result.get("spark_vector") if isinstance(result, dict) else None)
    reasoning_content = (result.get("reasoning_content") if isinstance(result, dict) else None)
    inline_think_content = (result.get("inline_think_content") if isinstance(result, dict) else None)
    thinking_source = "provider_reasoning" if bool(str(reasoning_content or "").strip()) else (
        "inline_think" if bool(str(inline_think_content or "").strip()) else "none"
    )
    backend = (result.get("backend") if isinstance(result, dict) else None)
    model_used = (result.get("model") if isinstance(result, dict) else None)
    route_used = (result.get("route") if isinstance(result, dict) else None) or typed_req.payload.route
    served_by = (result.get("served_by") if isinstance(result, dict) else None)
    model_used_early = (result.get("model") if isinstance(result, dict) else None)
    if route_used or served_by:
        logger.info(
            "gateway_llm_route_selected correlation_id=%s route=%s served_by=%s model=%s",
            typed_req.correlation_id,
            route_used,
            served_by,
            model_used_early,
        )
    gateway_label = f"{settings.node_name or 'gateway'}-{settings.service_name}"
    structured_diag = (
        result.get("structured_output_diagnostics") if isinstance(result, dict) else None
    )
    meta = {
        "served_by": served_by,
        "gateway": gateway_label,
        "route": route_used,
        "provider_reasoning_available": bool(str(reasoning_content or "").strip()),
        "inline_think_extracted": bool(str(inline_think_content or "").strip()),
        "thinking_source": thinking_source,
    }
    if isinstance(structured_diag, dict) and structured_diag:
        meta["structured_output_diagnostics"] = structured_diag
    llm_uncertainty = result.get("llm_uncertainty") if isinstance(result, dict) else None
    if isinstance(llm_uncertainty, dict):
        meta["llm_uncertainty"] = llm_uncertainty
    meta = {k: v for k, v in meta.items() if v is not None}
    if _thought_debug_enabled():
        logger.info(
            "THOUGHT_DEBUG_LLM stage=handle_chat_result corr=%s model=%s reasoning_exists=%s reasoning_len=%s content_len=%s reasoning_snippet=%r content_snippet=%r",
            typed_req.correlation_id,
            model_used,
            bool(str(reasoning_content or "").strip()),
            _debug_len(reasoning_content),
            _debug_len(text),
            _debug_snippet(reasoning_content),
            _debug_snippet(text),
        )

    reasoning_trace = (result.get("reasoning_trace") if isinstance(result, dict) else None)
    if reasoning_content and not (isinstance(reasoning_trace, dict) and str(reasoning_trace.get("content") or "").strip()):
        reasoning_trace = {
            "role": "reasoning",
            "stage": "post_answer",
            "content": reasoning_content,
        }
    out = Envelope[ChatResultPayload](
        kind="llm.chat.result",
        source=_source(),  # [FIX]
        correlation_id=typed_req.correlation_id,
        causality_chain=typed_req.causality_chain,
        payload=ChatResultPayload(
            model_used=model_used,
            content=text or "",
            reasoning_content=reasoning_content,
            inline_think_content=inline_think_content,
            thinking_source=thinking_source,
            reasoning_trace=reasoning_trace,
            usage=(result.get("raw") or {}).get("usage", {}) if isinstance(result, dict) else {},
            raw=(result.get("raw") if isinstance(result, dict) else None) or {},
            spark_meta=spark_meta,
            spark_vector=spark_vector,
            meta=meta or None,
        ),
    )
    trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
    print(
        "===THINK_HOP=== hop=llm_gateway_out "
        f"corr={typed_req.correlation_id} "
        f"has_reasoning_content={bool(reasoning_content)} "
        f"reasoning_len={len(reasoning_content) if reasoning_content else 0} "
        f"trace_len={len(trace_content) if trace_content else 0} "
        f"inline_think_len={len(inline_think_content) if isinstance(inline_think_content, str) else 0} "
        f"preview={_preview_text(reasoning_content or trace_content)}",
        flush=True,
    )
    response_payload = out.payload
    try:
        resp_keys = sorted(response_payload.model_dump().keys())
    except Exception:
        resp_keys = [type(response_payload).__name__]
    print(
        "===THINK_HOP=== hop=llm_gateway_response_shape "
        f"corr={typed_req.correlation_id} keys={resp_keys}",
        flush=True,
    )
    if bus_handle and text:
        doc_id = str(typed_req.correlation_id or env.id)
        try:
            asyncio.create_task(
                publish_assistant_embedding(
                    bus_handle,
                    text=text,
                    doc_id=doc_id,
                    trace_id=typed_req.correlation_id,
                )
            )
        except Exception as exc:
            logger.warning("Embedding publish schedule failed doc_id=%s error=%s", doc_id, exc)
    await _maybe_publish_latent_upsert(
        env=env,
        spark_vector=spark_vector,
        backend=backend,
        model_used=model_used or out.payload.model_used,
        session_id=typed_req.payload.session_id,
        user_id=typed_req.payload.user_id,
    )
    return out.model_copy(update={"reply_to": None})


async def _serve_health() -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.llm_gateway_health_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _connect_pool_bus(parent: Any) -> None:
    """Fork the RPC client the pool lease client uses; retry until the bus is reachable. Until
    then every call answers gpu_pool_unavailable (reason pool_bus_unavailable), never a guess."""
    from orion.core.bus.rpc_fork import fork_rpc_client

    delay = 1.0
    while pool_placement.get_bus() is None:
        try:
            pool_placement.set_bus(await fork_rpc_client(parent))
            logger.info("[LLM-GW] gpu pool RPC client ready")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM-GW] gpu pool RPC client not ready (%s); retrying in %.0fs", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[LLM-GW] %(levelname)s - %(message)s")
    cfg = _cfg()
    chat_svc = Rabbit(
        cfg,
        request_channel=settings.channel_llm_intake,
        handler=handle_chat,
        concurrent_handlers=settings.llm_gateway_concurrent_handlers,
    )
    global bus_handle
    bus_handle = chat_svc.bus
    # Placement is the pool's: every call leases over a forked RPC client (lease replies must
    # not be consumed by the chassis' own intake subscriber).
    pool_bus_task = asyncio.create_task(_connect_pool_bus(chat_svc.bus), name="gpu-pool-bus")
    routes = pool_placement.pool_routes()
    logger.info(
        "[LLM-GW] startup gpu_pool_config=%s routes=[%s] timeouts=connect:%s read:%s bus=%s channel=%s",
        settings.gpu_pool_config_path,
        ",".join(f"{name}={spec.work_class}/{spec.priority}" for name, spec in sorted(routes.items())),
        settings.connect_timeout_sec,
        settings.read_timeout_sec,
        cfg.bus_url,
        settings.channel_llm_intake,
    )
    logger.info(
        "Rabbit listening channels=%s bus=%s",
        settings.channel_llm_intake,
        cfg.bus_url,
    )
    await asyncio.gather(chat_svc.start(), _serve_health(), pool_bus_task)


if __name__ == "__main__":
    asyncio.run(main())
