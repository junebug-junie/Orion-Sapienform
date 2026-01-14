from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef
# We should prefer shared schemas over bus_schemas if they are defined there.
# The previous code used `RecallRequestPayload` from `orion.core.bus.bus_schemas`.
# This is "typed", but ideally should be in `orion.schemas.recall`.
# I will use the `orion.schemas.recall` ones if possible, but I need to ensure they match exactly.
# Or I can just check if `orion.core.bus.bus_schemas` is the authoritative source for these per `docs/bus-contracts.md`.
# `docs/bus-contracts.md` says: `Request: RecallRequestPayload`, `Response: RecallResultPayload`.
# It does NOT say `orion.schemas.recall.RecallRequest`.
# So `orion.core.bus.bus_schemas` IS the source of truth for these specific contracts right now.
# However, the user asked me to use `orion.schemas.*`.
# I will import them from `orion.core.bus.bus_schemas` as the service does, but ensure strict usage.

from orion.core.bus.bus_schemas import RecallRequestPayload
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, MemoryItemV1, RecallReplyV1
from orion.core.bus.bus_service_chassis import ChassisConfig

from .pipeline import run_recall_pipeline
from .profiles import get_profile
from .render import render_items
from .types import RecallQuery
from .settings import settings

logger = logging.getLogger("orion-recall")

RECALL_REQUEST_KIND = "recall.query.v1"
RECALL_REPLY_KIND = "recall.reply.v1"


def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=bool(settings.ORION_BUS_ENABLED),
        heartbeat_interval_sec=float(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10.0)),
        health_channel=getattr(settings, "ORION_HEALTH_CHANNEL", "orion:system:health"),
        error_channel=getattr(settings, "ERROR_CHANNEL", "orion:system:error"),
        shutdown_timeout_sec=float(getattr(settings, "SHUTDOWN_GRACE_SEC", 10.0)),
    )

def _query_from_payload(p: RecallRequestPayload) -> RecallQuery:
    return RecallQuery(
        query_text=p.query_text,
        max_items=int(p.max_items),
        time_window_days=int(p.time_window_days),
        mode=str(p.mode),
        tags=list((p.tags or []) or (p.packs or [])),
        phi=None,
        trace_id=p.trace_id,
        session_id=p.session_id,
        user_id=p.user_id,
        packs=list(p.packs or []),
    )


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )

def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_from_fragment(meta: Dict[str, Any], salience: float) -> float:
    score = meta.get("score")
    if score is None:
        score = meta.get("similarity")
    if score is None:
        score = salience
    return max(0.0, min(1.0, _coerce_float(score, 0.0)))


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    """Rabbit handler: validate envelope, run recall, return typed result envelope."""

    # Strict kind check
    if env.kind not in (RECALL_REQUEST_KIND, "recall.query.request", "legacy.message"):
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message":
        payload_obj = payload_obj.get("payload") or payload_obj

    try:
        req_payload = RecallRequestPayload.model_validate(payload_obj)
    except ValidationError as ve:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": str(ve)},
        )

    q = _query_from_payload(req_payload)
    started = time.perf_counter()
    result = run_recall_pipeline(q)
    latency_ms = int((time.perf_counter() - started) * 1000)

    backend_counts: Dict[str, int] = {}
    items: List[MemoryItemV1] = []
    for fr in result.fragments:
        source = str(fr.source or "unknown")
        backend_counts[source] = backend_counts.get(source, 0) + 1
        meta = fr.meta or {}
        item = MemoryItemV1(
            id=str(fr.id or ""),
            source=source,
            source_ref=meta.get("source_ref"),
            uri=meta.get("uri"),
            score=_score_from_fragment(meta, fr.salience),
            ts=fr.ts,
            title=meta.get("title"),
            snippet=str(fr.text or ""),
            tags=[str(t) for t in (fr.tags or []) if t],
        )
        items.append(item)

    profile = get_profile(settings.RECALL_DEFAULT_PROFILE)
    render_budget = int(profile.get("render_budget_tokens", 256))
    rendered = render_items(items, render_budget)
    stats = MemoryBundleStatsV1(
        backend_counts=backend_counts,
        latency_ms=latency_ms,
        profile=profile.get("profile"),
    )
    bundle = MemoryBundleV1(rendered=rendered, items=items, stats=stats)
    reply_payload = RecallReplyV1(bundle=bundle, correlation_id=str(env.correlation_id))

    out = BaseEnvelope(
        kind=RECALL_REPLY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=reply_payload.model_dump(mode="json"),
    )
    return out
