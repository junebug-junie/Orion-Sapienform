from __future__ import annotations

import logging
import os
from typing import Any, Iterable, List, Optional, Tuple
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.chat_history import (
    ChatHistoryMessageEnvelope,
    ChatHistoryMessageV1,
    ChatHistoryTurnEnvelope,
    ChatHistoryTurnV1,
    ChatRole,
)
from orion.schemas.social_chat import SocialRoomTurnV1
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1

from .social_room import build_social_room_turn

from .settings import settings

logger = logging.getLogger("orion-hub.chat_history")
SOCIAL_ROOM_TURN_CHANNEL = "orion:chat:social:turn"


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: object) -> int:
    return len(str(value or ""))


def _debug_snippet(value: object, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


def select_reasoning_trace_for_history(
    *,
    correlation_id: UUID | str | None,
    reasoning_trace: Optional[MetacognitiveTraceV1 | dict[str, Any]],
    metacog_traces: Optional[List[dict[str, Any]]],
    reasoning_content: Optional[str],
    session_id: Optional[str],
    message_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Optional[dict[str, Any]], str]:
    traces_exist = isinstance(metacog_traces, list) and len(metacog_traces) > 0
    explicit_trace_exists = bool(reasoning_trace)
    content_exists = bool(str(reasoning_content or "").strip())
    selected_source = "none"
    selected_trace: Optional[dict[str, Any]] = None

    explicit_candidate: Optional[dict[str, Any]] = None
    if hasattr(reasoning_trace, "model_dump"):
        try:
            explicit_candidate = reasoning_trace.model_dump(mode="json")
        except Exception:
            explicit_candidate = None
    elif isinstance(reasoning_trace, dict):
        explicit_candidate = dict(reasoning_trace)

    if isinstance(explicit_candidate, dict):
        content = str(explicit_candidate.get("content") or "").strip()
        if content:
            selected_trace = explicit_candidate
            selected_source = "reasoning_trace"

    if selected_trace is None and content_exists:
        selected_trace = {
            "correlation_id": str(correlation_id) if correlation_id is not None else "",
            "session_id": session_id,
            "message_id": message_id,
            "trace_role": "reasoning",
            "trace_stage": "post_answer",
            "content": str(reasoning_content or "").strip(),
            "model": str(model or "unknown"),
            "metadata": {"source": "hub_reasoning_content_fallback"},
        }
        selected_source = "reasoning_content"

    if selected_trace is None and isinstance(metacog_traces, list):
        for idx, trace in enumerate(metacog_traces):
            if not isinstance(trace, dict):
                continue
            trace_role = str(trace.get("trace_role") or trace.get("role") or "").strip().lower()
            if trace_role != "reasoning":
                continue
            content = str(trace.get("content") or "").strip()
            if content:
                selected_trace = trace
                selected_source = f"metacog_traces[{idx}]"
                break

    if _thought_debug_enabled():
        selected_content = (
            selected_trace.get("content") if isinstance(selected_trace, dict) else None
        )
        logger.info(
            "THOUGHT_DEBUG_HUB stage=reasoning_trace_select corr=%s shape=%s selected_source=%s selected_content_len=%s selected_content_snippet=%r",
            correlation_id,
            {
                "explicit_reasoning_trace_exists": explicit_trace_exists,
                "metacog_traces_exists": traces_exist,
                "reasoning_content_exists": content_exists,
            },
            selected_source,
            _debug_len(selected_content),
            _debug_snippet(selected_content),
        )

    return selected_trace, selected_source


def build_chat_history_envelope(
    *,
    content: str,
    role: ChatRole,
    session_id: Optional[str],
    correlation_id: UUID | str | None,
    speaker: Optional[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tags: Optional[List[str]] = None,
    message_id: Optional[str] = None,
    memory_status: Optional[str] = None,
    memory_tier: Optional[str] = None,
    memory_reason: Optional[str] = None,
    client_meta: Optional[dict] = None,
    reasoning_trace: Optional[MetacognitiveTraceV1 | dict] = None,
) -> ChatHistoryMessageEnvelope:
    """
    Construct a versioned chat history envelope with Orion's canonical bus schema.
    """
    payload_kwargs = {
        "session_id": session_id,
        "role": role,
        "speaker": speaker,
        "content": content,
        "model": model,
        "provider": provider,
        "tags": tags or [],
        "memory_status": memory_status,
        "memory_tier": memory_tier,
        "memory_reason": memory_reason,
        "client_meta": client_meta,
        "reasoning_trace": reasoning_trace,
    }
    if message_id:
        payload_kwargs["message_id"] = message_id

    payload = ChatHistoryMessageV1(**payload_kwargs)

    return ChatHistoryMessageEnvelope(
        correlation_id=correlation_id or uuid4(),
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=payload,
    )


async def publish_chat_history(
    bus, envelopes: Iterable[ChatHistoryMessageEnvelope]
) -> None:
    """
    Publish one or more chat history envelopes to the configured channel.
    """
    if not bus or not getattr(bus, "enabled", False):
        return
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return

    channel = settings.chat_history_channel
    for env in envelopes:
        try:
            if _thought_debug_enabled():
                payload = env.payload
                rt = getattr(payload, "reasoning_trace", None)
                rc = getattr(payload, "reasoning_content", None)
                thought_candidate = (
                    (rt.content if hasattr(rt, "content") else (rt.get("content") if isinstance(rt, dict) else None))
                    or rc
                )
                logger.info(
                    "THOUGHT_DEBUG_HUB stage=publish_chat_history corr=%s channel=%s target=chat.history.message.v1 reasoning_trace_exists=%s reasoning_content_exists=%s thought_candidate_len=%s thought_candidate_snippet=%r content_len=%s",
                    env.correlation_id,
                    channel,
                    bool(rt),
                    bool(str(rc or "").strip()),
                    _debug_len(thought_candidate),
                    _debug_snippet(thought_candidate),
                    _debug_len(getattr(payload, "content", None)),
                )
            await bus.publish(channel, env)
        except Exception as e:
            logger.warning("Failed to publish chat history: %s", e, exc_info=True)

def build_chat_turn_envelope(
    *,
    prompt: str,
    response: str,
    session_id: Optional[str],
    correlation_id: UUID | str | None,
    user_id: Optional[str],
    source_label: str = "hub_ws",
    spark_meta: Optional[dict] = None,
    turn_id: Optional[str] = None,
    memory_status: Optional[str] = None,
    memory_tier: Optional[str] = None,
    memory_reason: Optional[str] = None,
    client_meta: Optional[dict] = None,
    reasoning_trace: Optional[MetacognitiveTraceV1 | dict] = None,
) -> ChatHistoryTurnEnvelope:
    """Construct a turn-level chat history envelope (prompt + response)."""
    merged_spark_meta = dict(spark_meta or {})
    if client_meta:
        merged_spark_meta.setdefault("client_meta", client_meta)
    payload = ChatHistoryTurnV1(
        id=turn_id,
        correlation_id=str(correlation_id) if correlation_id is not None else None,
        source=source_label,
        prompt=prompt,
        response=response,
        user_id=user_id,
        session_id=session_id,
        spark_meta=merged_spark_meta or None,
        memory_status=memory_status,
        memory_tier=memory_tier,
        memory_reason=memory_reason,
        client_meta=client_meta,
        reasoning_trace=reasoning_trace,
    )
    return ChatHistoryTurnEnvelope(
        correlation_id=correlation_id or uuid4(),
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=payload,
    )


async def publish_chat_turn(bus, env: ChatHistoryTurnEnvelope) -> None:
    """Publish a turn-level chat history envelope to the configured turn channel."""
    if not bus or not getattr(bus, "enabled", False):
        return
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return

    channel = settings.chat_history_turn_channel
    try:
        turn_payload = env.payload
        explicit_reasoning_trace = getattr(turn_payload, "reasoning_trace", None)
        spark_meta = getattr(turn_payload, "spark_meta", None)
        reasoning_content = getattr(turn_payload, "reasoning_content", None)
        if not isinstance(reasoning_content, str) and isinstance(spark_meta, dict):
            maybe_reasoning_content = spark_meta.get("reasoning_content")
            if isinstance(maybe_reasoning_content, str):
                reasoning_content = maybe_reasoning_content
        metacog_traces = getattr(turn_payload, "metacog_traces", None)
        if not isinstance(metacog_traces, list) and isinstance(spark_meta, dict):
            maybe_metacog_traces = spark_meta.get("metacog_traces")
            if isinstance(maybe_metacog_traces, list):
                metacog_traces = maybe_metacog_traces
        selected_source = "none"
        selected_content: Optional[str] = None
        selected_trace: Optional[dict[str, Any]] = None
        if isinstance(explicit_reasoning_trace, dict) and explicit_reasoning_trace.get("content"):
            selected_source = "reasoning_trace.content"
            selected_content = str(explicit_reasoning_trace.get("content")).strip()
            selected_trace = dict(explicit_reasoning_trace)
        elif isinstance(reasoning_content, str) and reasoning_content.strip():
            selected_source = "reasoning_content"
            selected_content = reasoning_content.strip()
            selected_trace = {
                "correlation_id": str(env.correlation_id),
                "session_id": getattr(turn_payload, "session_id", None),
                "message_id": getattr(turn_payload, "id", None) or str(env.correlation_id),
                "trace_role": "reasoning",
                "trace_stage": "post_answer",
                "content": selected_content,
                "metadata": {"source": "chat_history_reasoning_content_fallback"},
            }
        elif isinstance(metacog_traces, list):
            for idx, trace in enumerate(metacog_traces):
                if not isinstance(trace, dict):
                    continue
                trace_role = str(trace.get("trace_role") or trace.get("role") or "").strip().lower()
                if trace_role != "reasoning":
                    continue
                content = str(trace.get("content") or "").strip()
                if content:
                    selected_source = f"metacog_traces[{idx}].content"
                    selected_content = content
                    selected_trace = dict(trace)
                    break
        if selected_trace is not None and hasattr(turn_payload, "reasoning_trace"):
            turn_payload.reasoning_trace = selected_trace
        print(
            "===THINK_HOP=== hop=chat_history_publish "
            f"corr={env.correlation_id} "
            f"source={selected_source} "
            f"len={len(selected_content) if selected_content else 0} "
            f"preview={_preview_text(selected_content)}",
            flush=True,
        )
        payload_dump = turn_payload.model_dump() if hasattr(turn_payload, "model_dump") else turn_payload
        print(
            "===THINK_HOP=== hop=chat_history_payload "
            f"corr={env.correlation_id} "
            f"keys={sorted(payload_dump.keys()) if isinstance(payload_dump, dict) else type(payload_dump).__name__}",
            flush=True,
        )
        if _thought_debug_enabled():
            payload = env.payload
            rt = getattr(payload, "reasoning_trace", None)
            rc = getattr(payload, "reasoning_content", None)
            thought_candidate = (
                (rt.content if hasattr(rt, "content") else (rt.get("content") if isinstance(rt, dict) else None))
                or rc
            )
            logger.info(
                "THOUGHT_DEBUG_HUB stage=publish_chat_turn corr=%s channel=%s target=chat.history.turn reasoning_trace_exists=%s reasoning_content_exists=%s thought_candidate_len=%s thought_candidate_snippet=%r prompt_len=%s response_len=%s",
                env.correlation_id,
                channel,
                bool(rt),
                bool(str(rc or "").strip()),
                _debug_len(thought_candidate),
                _debug_snippet(thought_candidate),
                _debug_len(getattr(payload, "prompt", None)),
                _debug_len(getattr(payload, "response", None)),
            )
            logger.info(
                "THOUGHT_DEBUG_HUB stage=publish_chat_turn_shape corr=%s shape=%s",
                env.correlation_id,
                {
                    "payload_keys": sorted(list(payload.model_dump(mode="json").keys())) if hasattr(payload, "model_dump") else [],
                    "reasoning_trace_keys": sorted(list(rt.keys())) if isinstance(rt, dict) else (sorted(list(rt.model_dump(mode="json").keys())) if hasattr(rt, "model_dump") else []),
                    "reasoning_trace_content_len": _debug_len((rt.get("content") if isinstance(rt, dict) else getattr(rt, "content", None))),
                    "reasoning_content_len": _debug_len(rc),
                },
            )
        await bus.publish(channel, env)
    except Exception as e:
        logger.warning("Failed to publish chat turn history: %s", e, exc_info=True)


async def publish_social_room_turn(
    bus,
    *,
    prompt: str,
    response: str,
    session_id: Optional[str],
    correlation_id: UUID | str | None,
    user_id: Optional[str],
    source_label: str,
    recall_profile: Optional[str],
    trace_verb: Optional[str],
    client_meta: Optional[dict] = None,
    memory_digest: Optional[str] = None,
) -> Optional[SocialRoomTurnV1]:
    """Publish an append-only social_room turn event when the UI selected that chat profile."""
    social_meta = dict(client_meta or {})
    if str(social_meta.get("chat_profile") or "").strip().lower() != "social_room":
        return None
    if not bus or not getattr(bus, "enabled", False):
        return None
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return None
    turn = build_social_room_turn(
        prompt=prompt,
        response=response,
        session_id=session_id,
        correlation_id=str(correlation_id) if correlation_id is not None else None,
        user_id=user_id,
        source=source_label,
        recall_profile=recall_profile,
        trace_verb=trace_verb,
        client_meta=social_meta,
        memory_digest=memory_digest,
    )
    env = BaseEnvelope(
        kind="social.turn.v1",
        correlation_id=correlation_id or uuid4(),
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=turn,
    )
    try:
        await bus.publish(SOCIAL_ROOM_TURN_CHANNEL, env)
        return turn
    except Exception as e:
        logger.warning("Failed to publish social_room turn history: %s", e, exc_info=True)
        return None
