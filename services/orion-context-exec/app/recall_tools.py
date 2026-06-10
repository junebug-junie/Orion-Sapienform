from __future__ import annotations

import logging
import time
from uuid import UUID, uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1

from .schemas import RecallResult
from .settings import settings

logger = logging.getLogger("orion-context-exec.recall_tools")


def _corr_uuid(value: str | None) -> UUID:
    if not value:
        return uuid4()
    try:
        return UUID(str(value))
    except ValueError:
        return uuid4()


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


async def recall_query(
    bus: OrionBusAsync | None,
    *,
    query: str,
    profile: str = "assist.light.v1",
    limit: int | None = None,
    correlation_id: str | None = None,
    session_id: str | None = None,
) -> RecallResult:
    cap = limit if limit is not None else settings.context_exec_recall_limit
    if not settings.context_exec_real_recall_enabled or bus is None or not settings.orion_bus_enabled:
        return RecallResult(hits=[], profile=profile, query=query)

    reply_channel = f"{settings.channel_recall_reply_prefix}:{uuid4().hex}"
    corr = _corr_uuid(correlation_id)
    req = RecallQueryV1(
        fragment=query,
        profile=profile,
        session_id=session_id,
        reply_to=reply_channel,
    )
    env = BaseEnvelope(
        kind="recall.query.v1",
        source=_source(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    started = time.perf_counter()
    try:
        msg = await bus.rpc_request(
            settings.channel_recall_intake,
            env,
            reply_channel=reply_channel,
            timeout_sec=settings.context_exec_recall_timeout_sec,
        )
    except Exception as exc:
        logger.warning("recall_query rpc failed corr=%s err=%s", corr, exc)
        return RecallResult(hits=[], profile=profile, query=query, error=str(exc))

    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        return RecallResult(
            hits=[],
            profile=profile,
            query=query,
            error=decoded.error or "decode_failed",
        )

    payload_data = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    if payload_data.get("error"):
        return RecallResult(
            hits=[],
            profile=profile,
            query=query,
            error=str(payload_data.get("error")),
        )

    try:
        reply = RecallReplyV1.model_validate(payload_data)
    except Exception as exc:
        return RecallResult(hits=[], profile=profile, query=query, error=str(exc))

    latency_ms = int((time.perf_counter() - started) * 1000)
    hits: list[dict] = []
    for item in reply.bundle.items[:cap]:
        hits.append(
            {
                "id": item.id,
                "snippet": item.snippet,
                "title": item.title,
                "source": item.source,
                "source_ref": item.source_ref or item.uri,
                "score": item.score,
                "tags": item.tags,
            }
        )
    return RecallResult(
        hits=hits,
        profile=profile,
        query=query,
        latency_ms=latency_ms,
        rendered_head=(reply.bundle.rendered or "")[:500],
    )
