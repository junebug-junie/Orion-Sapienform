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
from orion.schemas.chat_response_feedback import (
    ChatResponseFeedbackEnvelope,
    ChatResponseFeedbackV1,
)
from orion.schemas.social_chat import SocialRoomTurnV1
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1

from .social_room import build_social_room_turn

from .settings import settings

logger = logging.getLogger("orion-hub.chat_history")
SOCIAL_ROOM_TURN_CHANNEL = "orion:chat:social:turn"
_VALID_TRACE_ROLES = {"reasoning", "planning", "self_check", "critique", "reflection", "stance"}
_VALID_TRACE_STAGES = {"pre_answer", "mid_answer", "post_answer"}


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


def _best_model_name(*candidates: object) -> str:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return "unknown"


def _normalize_reasoning_trace(
    *,
    trace: Optional[MetacognitiveTraceV1 | dict[str, Any]],
    correlation_id: UUID | str | None,
    session_id: Optional[str],
    message_id: Optional[str],
    model: Optional[str],
    metadata_source: str,
) -> Optional[dict[str, Any]]:
    if trace is None:
        return None
    candidate: Optional[dict[str, Any]] = None
    if hasattr(trace, "model_dump"):
        try:
            candidate = trace.model_dump(mode="json")
        except Exception:
            candidate = None
    elif isinstance(trace, dict):
        candidate = dict(trace)
    if not isinstance(candidate, dict):
        return None

    content = str(candidate.get("content") or "").strip()
    if not content:
        return None

    trace_role = str(candidate.get("trace_role") or candidate.get("role") or "reasoning").strip().lower()
    if trace_role not in _VALID_TRACE_ROLES:
        trace_role = "reasoning"
    trace_stage = str(candidate.get("trace_stage") or candidate.get("stage") or "post_answer").strip().lower()
    if trace_stage not in _VALID_TRACE_STAGES:
        trace_stage = "post_answer"

    candidate_meta = candidate.get("metadata")
    metadata = dict(candidate_meta) if isinstance(candidate_meta, dict) else {}
    metadata.setdefault("source", metadata_source)

    model_name = _best_model_name(
        candidate.get("model"),
        model,
        metadata.get("model"),
        metadata.get("model_name"),
    )
    corr = _best_model_name(candidate.get("correlation_id"), correlation_id, uuid4())

    normalized: dict[str, Any] = {
        "trace_id": str(candidate.get("trace_id") or uuid4()),
        "correlation_id": str(corr),
        "session_id": str(candidate.get("session_id") or session_id) if (candidate.get("session_id") or session_id) is not None else None,
        "message_id": str(candidate.get("message_id") or message_id) if (candidate.get("message_id") or message_id) is not None else None,
        "trace_role": trace_role,
        "trace_stage": trace_stage,
        "content": content,
        "model": model_name,
        "token_count": candidate.get("token_count"),
        "confidence": candidate.get("confidence"),
        "metadata": metadata,
        "created_at": candidate.get("created_at"),
    }
    return {k: v for k, v in normalized.items() if v is not None}


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

    explicit_candidate = _normalize_reasoning_trace(
        trace=reasoning_trace,
        correlation_id=correlation_id,
        session_id=session_id,
        message_id=message_id,
        model=model,
        metadata_source="hub_reasoning_trace",
    )
    if isinstance(explicit_candidate, dict):
        selected_trace = explicit_candidate
        selected_source = "reasoning_trace"

    if selected_trace is None and content_exists:
        selected_trace = _normalize_reasoning_trace(
            trace={
                "correlation_id": str(correlation_id) if correlation_id is not None else "",
                "session_id": session_id,
                "message_id": message_id,
                "trace_role": "reasoning",
                "trace_stage": "post_answer",
                "content": str(reasoning_content or "").strip(),
                "model": str(model or "unknown"),
                "metadata": {"source": "hub_reasoning_content_fallback"},
            },
            correlation_id=correlation_id,
            session_id=session_id,
            message_id=message_id,
            model=model,
            metadata_source="hub_reasoning_content_fallback",
        )
        selected_source = "reasoning_content"

    if selected_trace is None and isinstance(metacog_traces, list):
        for idx, trace in enumerate(metacog_traces):
            normalized_trace = _normalize_reasoning_trace(
                trace=trace if isinstance(trace, dict) else None,
                correlation_id=correlation_id,
                session_id=session_id,
                message_id=message_id,
                model=model,
                metadata_source=f"hub_metacog_traces_{idx}",
            )
            if isinstance(normalized_trace, dict) and normalized_trace.get("trace_role") == "reasoning":
                selected_trace = normalized_trace
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
    normalized_reasoning_trace = _normalize_reasoning_trace(
        trace=reasoning_trace,
        correlation_id=correlation_id,
        session_id=session_id,
        message_id=message_id,
        model=model,
        metadata_source="chat_history_message",
    )
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
        "reasoning_trace": normalized_reasoning_trace,
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
    normalized_reasoning_trace = _normalize_reasoning_trace(
        trace=reasoning_trace,
        correlation_id=correlation_id,
        session_id=session_id,
        message_id=turn_id,
        model=(spark_meta or {}).get("model") if isinstance(spark_meta, dict) else None,
        metadata_source="chat_turn_envelope",
    )
    normalized_preview = (
        str((normalized_reasoning_trace or {}).get("content") or "").strip()
        if isinstance(normalized_reasoning_trace, dict)
        else ""
    )
    print(
        "===THINK_HOP=== hop=chat_history_reasoning_trace_shape "
        f"corr={correlation_id} "
        f"keys={sorted(list((normalized_reasoning_trace or {}).keys())) if isinstance(normalized_reasoning_trace, dict) else []} "
        f"preview={_preview_text(normalized_preview)}",
        flush=True,
    )
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
        reasoning_trace=normalized_reasoning_trace,
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
        inferred_model = (
            str((spark_meta or {}).get("model") or (spark_meta or {}).get("model_name") or "").strip()
            if isinstance(spark_meta, dict)
            else None
        )
        normalized_explicit = _normalize_reasoning_trace(
            trace=explicit_reasoning_trace if isinstance(explicit_reasoning_trace, dict) else None,
            correlation_id=env.correlation_id,
            session_id=getattr(turn_payload, "session_id", None),
            message_id=getattr(turn_payload, "id", None) or str(env.correlation_id),
            model=inferred_model,
            metadata_source="chat_turn_existing_reasoning_trace",
        )
        if isinstance(normalized_explicit, dict) and normalized_explicit.get("content"):
            selected_source = "reasoning_trace.content"
            selected_content = str(normalized_explicit.get("content")).strip()
            selected_trace = normalized_explicit
        elif isinstance(reasoning_content, str) and reasoning_content.strip():
            selected_source = "reasoning_content"
            selected_content = reasoning_content.strip()
            selected_trace = _normalize_reasoning_trace(
                trace={
                    "correlation_id": str(env.correlation_id),
                    "session_id": getattr(turn_payload, "session_id", None),
                    "message_id": getattr(turn_payload, "id", None) or str(env.correlation_id),
                    "trace_role": "reasoning",
                    "trace_stage": "post_answer",
                    "content": selected_content,
                    "model": inferred_model or "unknown",
                    "metadata": {"source": "chat_history_reasoning_content_fallback"},
                },
                correlation_id=env.correlation_id,
                session_id=getattr(turn_payload, "session_id", None),
                message_id=getattr(turn_payload, "id", None) or str(env.correlation_id),
                model=inferred_model,
                metadata_source="chat_history_reasoning_content_fallback",
            )
        elif isinstance(metacog_traces, list):
            for idx, trace in enumerate(metacog_traces):
                normalized_metacog = _normalize_reasoning_trace(
                    trace=trace if isinstance(trace, dict) else None,
                    correlation_id=env.correlation_id,
                    session_id=getattr(turn_payload, "session_id", None),
                    message_id=getattr(turn_payload, "id", None) or str(env.correlation_id),
                    model=inferred_model,
                    metadata_source=f"chat_turn_metacog_traces_{idx}",
                )
                if isinstance(normalized_metacog, dict) and normalized_metacog.get("trace_role") == "reasoning":
                    selected_source = f"metacog_traces[{idx}].content"
                    selected_content = str(normalized_metacog.get("content") or "").strip()
                    selected_trace = normalized_metacog
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


def build_chat_response_feedback_envelope(
    *,
    feedback_payload: ChatResponseFeedbackV1,
    correlation_id: UUID | str | None,
) -> ChatResponseFeedbackEnvelope:
    """Construct a feedback envelope linked to a specific assistant response."""
    normalized_corr: UUID
    corr_candidate = correlation_id or feedback_payload.target_correlation_id
    try:
        normalized_corr = UUID(str(corr_candidate)) if corr_candidate is not None else uuid4()
    except Exception:
        normalized_corr = uuid4()

    return ChatResponseFeedbackEnvelope(
        correlation_id=normalized_corr,
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=feedback_payload,
    )


async def publish_chat_response_feedback(
    bus,
    env: ChatResponseFeedbackEnvelope,
    *,
    channel: str = "orion:chat:response:feedback",
) -> None:
    if not bus or not getattr(bus, "enabled", False):
        return
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return
    try:
        await bus.publish(channel, env)
    except Exception as e:
        logger.warning("Failed to publish chat response feedback: %s", e, exc_info=True)
