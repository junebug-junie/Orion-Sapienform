"""Recall bundle prefetch for Mind preflight (Orch, before Exec)."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from orion.cognition.recall_query import (
    DEFAULT_RECALL_REPLY_PREFIX,
    build_recall_query_v1,
    last_user_message_from_ctx,
    recall_ctx_merge_from_reply,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import RecallReplyV1

logger = logging.getLogger("orion.cognition.recall_prefetch")


def _fresh_prefetch_diagnostics(
    *,
    correlation_id: str,
    recall_channel: str,
    timeout_sec: float,
    enabled: bool,
    reason: str,
    profile: str | None = None,
) -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "recall_channel": recall_channel,
        "endpoint": f"bus:{recall_channel}",
        "timeout_sec": timeout_sec,
        "enabled": enabled,
        "reason": reason,
        "profile": profile,
        "ok": False,
        "elapsed_ms": 0,
        "timed_out": False,
        "exception_type": None,
        "status_code": None,
        "result_count": 0,
        "bundle_keys": [],
        "normalized_ctx_keys_written": [],
        "recall_bundle_present_after_write": False,
        "retryable": True,
        "degradation_reason": None,
    }


def log_mind_projection_prebuild_ctx_summary(
    *,
    correlation_id: str,
    ctx: dict[str, Any],
    recall_prefetch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured pre-build ctx summary (after recall prefetch, before projection)."""
    recall_bundle = ctx.get("recall_bundle") if isinstance(ctx.get("recall_bundle"), dict) else {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle.get("fragments"), list) else []
    social_keys = (
        "social_inspection_snapshot",
        "social_stance_snapshot",
        "social_turn_policy",
    )
    summary = {
        "correlation_id": correlation_id,
        "orion_identity_count": len(ctx.get("orion_identity_summary") or [])
        if isinstance(ctx.get("orion_identity_summary"), list)
        else 0,
        "juniper_relationship_count": len(ctx.get("juniper_relationship_summary") or [])
        if isinstance(ctx.get("juniper_relationship_summary"), list)
        else 0,
        "response_policy_count": len(ctx.get("response_policy_summary") or [])
        if isinstance(ctx.get("response_policy_summary"), list)
        else 0,
        "recall_bundle_present": bool(fragments),
        "recall_result_count": len(fragments),
        "recall_bundle_keys": list(recall_bundle.keys()) if isinstance(recall_bundle, dict) else [],
        "situation_summary_present": isinstance(ctx.get("chat_situation_summary"), dict),
        "social_snapshot_present": any(ctx.get(k) is not None for k in social_keys),
        "message_count": len(ctx.get("messages") or []) if isinstance(ctx.get("messages"), list) else 0,
        "user_message_present": bool(last_user_message_from_ctx(ctx)),
        "session_id_present": bool(ctx.get("session_id")),
        "recall_prefetch": recall_prefetch,
    }
    logger.info("mind_projection_prebuild_ctx_summary %s", summary)
    return summary


async def prefetch_recall_bundle_for_projection(
    bus: Any,
    *,
    source: ServiceRef,
    ctx: dict[str, Any],
    correlation_id: str,
    recall_enabled: bool,
    recall_profile: str | None,
    recall_channel: str,
    timeout_sec: float,
    recall_cfg: dict[str, Any] | None = None,
    recall_reply_prefix: str = DEFAULT_RECALL_REPLY_PREFIX,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run recall bus RPC; return (ctx_merge, diagnostics).

    On failure/timeout returns (None, diagnostics) without writing fake recall_bundle.
    """
    recall_cfg = recall_cfg if isinstance(recall_cfg, dict) else {}
    profile = (recall_profile or recall_cfg.get("profile") or "reflect.v1").strip() or "reflect.v1"
    diagnostics = _fresh_prefetch_diagnostics(
        correlation_id=correlation_id,
        recall_channel=recall_channel,
        timeout_sec=timeout_sec,
        enabled=recall_enabled,
        reason="disabled" if not recall_enabled else "start",
        profile=profile,
    )

    if not recall_enabled:
        diagnostics["reason"] = "recall_disabled"
        diagnostics["retryable"] = False
        return None, diagnostics

    if isinstance(ctx.get("recall_bundle"), dict) and ctx["recall_bundle"].get("fragments"):
        diagnostics["reason"] = "recall_bundle_already_present"
        diagnostics["ok"] = True
        diagnostics["recall_bundle_present_after_write"] = True
        diagnostics["result_count"] = len(ctx["recall_bundle"]["fragments"])
        return None, diagnostics

    fragment_text = last_user_message_from_ctx(ctx)
    if not fragment_text:
        diagnostics["reason"] = "empty_query_text"
        diagnostics["retryable"] = False
        return None, diagnostics

    reply_prefix = (recall_reply_prefix or DEFAULT_RECALL_REPLY_PREFIX).strip().rstrip(":")
    reply_channel = f"{reply_prefix}:{uuid4()}"
    diagnostics["recall_reply_prefix"] = reply_prefix
    req = build_recall_query_v1(
        ctx,
        correlation_id=correlation_id,
        recall_profile=recall_profile,
        recall_cfg=recall_cfg,
        reply_to=reply_channel,
    )
    if req is None:
        diagnostics["reason"] = "query_build_failed"
        diagnostics["retryable"] = False
        return None, diagnostics

    history_count = len(ctx.get("messages") or []) if isinstance(ctx.get("messages"), list) else 0
    logger.info(
        "mind_recall_prefetch_start correlation_id=%s session_id=%s recall_channel=%s endpoint=%s "
        "timeout_sec=%s profile=%s query_preview=%r history_count=%s enabled=%s reason=%s",
        correlation_id,
        ctx.get("session_id"),
        recall_channel,
        diagnostics["endpoint"],
        timeout_sec,
        req.profile,
        fragment_text[:120],
        history_count,
        recall_enabled,
        diagnostics["reason"],
    )

    env = BaseEnvelope(
        kind="recall.query.v1",
        source=source,
        correlation_id=correlation_id,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    t0 = time.perf_counter()
    try:
        msg = await bus.rpc_request(
            recall_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics["elapsed_ms"] = elapsed_ms
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            diagnostics["reason"] = "decode_failed"
            diagnostics["degradation_reason"] = str(decoded.error)
            logger.warning(
                "mind_recall_prefetch_result correlation_id=%s ok=false elapsed_ms=%s "
                "exception_type=decode_error degradation_reason=%s",
                correlation_id,
                elapsed_ms,
                decoded.error,
            )
            return None, diagnostics
        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        if payload.get("error"):
            diagnostics["reason"] = "recall_service_error"
            diagnostics["degradation_reason"] = str(payload.get("error"))
            logger.warning(
                "mind_recall_prefetch_result correlation_id=%s ok=false elapsed_ms=%s "
                "recall_error=%s",
                correlation_id,
                elapsed_ms,
                payload.get("error"),
            )
            return None, diagnostics
        reply = RecallReplyV1.model_validate(payload)
        merge = recall_ctx_merge_from_reply(reply)
        bundle = merge.get("recall_bundle") if isinstance(merge.get("recall_bundle"), dict) else {}
        fragment_count = len(bundle.get("fragments") or []) if isinstance(bundle.get("fragments"), list) else 0
        diagnostics.update(
            {
                "ok": True,
                "reason": "success",
                "result_count": fragment_count,
                "bundle_keys": list(bundle.keys()) if isinstance(bundle, dict) else [],
                "normalized_ctx_keys_written": list(merge.keys()),
                "recall_bundle_present_after_write": fragment_count > 0,
                "retryable": False,
                "degradation_reason": None,
            }
        )
        logger.info(
            "mind_recall_prefetch_result correlation_id=%s ok=true elapsed_ms=%s result_count=%s "
            "bundle_keys=%s normalized_ctx_keys_written=%s recall_bundle_present_after_write=%s",
            correlation_id,
            elapsed_ms,
            fragment_count,
            diagnostics["bundle_keys"],
            diagnostics["normalized_ctx_keys_written"],
            diagnostics["recall_bundle_present_after_write"],
        )
        return merge, diagnostics
    except TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "timed_out": True,
                "exception_type": type(exc).__name__,
                "reason": "timeout",
                "degradation_reason": str(exc),
            }
        )
        logger.warning(
            "mind_recall_prefetch_timeout correlation_id=%s elapsed_ms=%s timeout_sec=%s "
            "service_url=bus:%s endpoint=%s profile=%s retryable=%s degradation_reason=%s",
            correlation_id,
            elapsed_ms,
            timeout_sec,
            recall_channel,
            diagnostics["endpoint"],
            req.profile,
            diagnostics["retryable"],
            diagnostics["degradation_reason"],
        )
        return None, diagnostics
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "exception_type": type(exc).__name__,
                "reason": "exception",
                "degradation_reason": str(exc),
            }
        )
        logger.warning(
            "mind_recall_prefetch_result correlation_id=%s ok=false elapsed_ms=%s timed_out=%s "
            "exception_type=%s degradation_reason=%s",
            correlation_id,
            elapsed_ms,
            diagnostics["timed_out"],
            diagnostics["exception_type"],
            diagnostics["degradation_reason"],
        )
        return None, diagnostics
