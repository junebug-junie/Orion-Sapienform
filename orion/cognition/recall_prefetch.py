"""Recall bundle prefetch for Mind preflight (Orch, before Exec)."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1

logger = logging.getLogger("orion.cognition.recall_prefetch")


def _last_user_message(ctx: dict[str, Any]) -> str:
    for key in ("user_message", "raw_user_text"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = ctx.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue
            content = message.get("content") or message.get("text") or ""
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _recall_bundle_from_reply(reply: RecallReplyV1) -> dict[str, Any]:
    recall_fragments: list[dict[str, Any]] = []
    recall_citations: list[dict[str, Any]] = []
    for item in reply.bundle.items[:12]:
        snippet = str(getattr(item, "snippet", "") or "")[:480]
        recall_fragment = {
            "id": str(getattr(item, "id", "") or ""),
            "snippet": snippet,
            "score": float(getattr(item, "score", 0.0) or 0.0),
            "tags": [str(tag) for tag in (getattr(item, "tags", []) or []) if tag],
            "source": str(getattr(item, "source", "") or ""),
            "source_ref": getattr(item, "source_ref", None),
            "uri": getattr(item, "uri", None),
        }
        recall_fragments.append(recall_fragment)
        recall_citations.append(
            {
                "id": recall_fragment["id"],
                "source": recall_fragment["source"],
                "source_ref": recall_fragment["source_ref"],
                "uri": recall_fragment["uri"],
            }
        )
    memory_digest = reply.bundle.rendered if hasattr(reply.bundle, "rendered") else ""
    return {
        "recall_bundle": {
            "fragments": recall_fragments,
            "citations": recall_citations,
            "rendered": memory_digest,
        },
        "memory_digest": memory_digest,
        "recall_fragments": recall_fragments,
        "memory_used": bool(recall_fragments),
    }


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
) -> dict[str, Any] | None:
    """Run recall RPC and return ctx fragments for the recall substrate producer."""
    if not recall_enabled:
        return None
    if isinstance(ctx.get("recall_bundle"), dict) and ctx["recall_bundle"].get("fragments"):
        return None

    fragment_text = _last_user_message(ctx)
    if not fragment_text:
        return None

    active_turn_ids: list[str] = []
    for candidate in (correlation_id, ctx.get("trace_id"), ctx.get("session_id")):
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value and value not in active_turn_ids:
            active_turn_ids.append(value)

    req = RecallQueryV1(
        fragment=fragment_text,
        verb=str(ctx.get("verb") or "unknown"),
        intent=ctx.get("intent"),
        session_id=ctx.get("session_id"),
        node_id=ctx.get("node_id"),
        profile=(recall_profile or "reflect.v1").strip() or "reflect.v1",
        exclude={
            "active_turn_ids": active_turn_ids,
            "active_turn_text": fragment_text,
            "active_turn_ts": time.time(),
        },
    )
    reply_channel = f"orion:orch:result:RecallService:{uuid4()}"
    env = BaseEnvelope(
        kind="recall.query.v1",
        source=source,
        correlation_id=correlation_id,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    try:
        msg = await bus.rpc_request(
            recall_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.warning("mind_recall_prefetch_decode_failed corr=%s err=%s", correlation_id, decoded.error)
            return None
        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        if payload.get("error"):
            logger.warning("mind_recall_prefetch_error corr=%s payload=%s", correlation_id, payload.get("error"))
            return None
        reply = RecallReplyV1.model_validate(payload)
        bundle = _recall_bundle_from_reply(reply)
        logger.info(
            "mind_recall_prefetch_ok corr=%s fragment_count=%s profile=%s",
            correlation_id,
            len(bundle.get("recall_fragments") or []),
            req.profile,
        )
        return bundle
    except Exception as exc:
        logger.warning("mind_recall_prefetch_failed corr=%s err=%s", correlation_id, exc)
        return None
