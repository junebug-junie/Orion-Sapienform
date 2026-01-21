from __future__ import annotations

import logging
import time
from typing import Any, Dict

from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, ServiceRef
from orion.core.bus.bus_schemas import RecallRequestPayload
from orion.core.contracts.recall import RecallReplyV1, MemoryBundleV1, MemoryItemV1, MemoryBundleStatsV1
from orion.core.bus.bus_service_chassis import ChassisConfig

from .profiles import get_profile
from .render import render_items
from .types import RecallQuery
from .settings import settings

def run_recall_pipeline(*args, **kwargs):
    from .pipeline import run_recall_pipeline as _impl
    return _impl(*args, **kwargs)

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

def _query_from_contract(p: RecallQueryV1) -> RecallQuery:
    return RecallQuery(
        query_text=p.fragment,
        max_items=int(settings.RECALL_DEFAULT_MAX_ITEMS),
        time_window_days=int(settings.RECALL_DEFAULT_TIME_WINDOW_DAYS),
        mode=str(settings.RECALL_DEFAULT_MODE),
        tags=[],
        phi=None,
        trace_id=None,
        session_id=p.session_id,
        user_id=None,
        packs=[],
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
    start_ts = time.perf_counter()

    # Strict kind check
    if env.kind != RECALL_REQUEST_KIND:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=RecallReplyV1(
                correlation_id=str(env.correlation_id),
                bundle=MemoryBundleV1(
                    rendered=f"Unsupported message kind: {env.kind}",
                    items=[],
                    stats=MemoryBundleStatsV1()
                )
            ).model_dump(mode="json"),
        )

    payload_obj: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}

    try:
        req_contract = RecallQueryV1.model_validate(payload_obj)
    except ValidationError as ve:
         # FIXED: Return a compliant RecallReplyV1 with error info in 'rendered'
         return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=RecallReplyV1(
                correlation_id=str(env.correlation_id),
                bundle=MemoryBundleV1(
                    rendered=f"Validation failed: {ve}",
                    items=[],
                    stats=MemoryBundleStatsV1()
                )
            ).model_dump(mode="json"),
        )
    profile_name = req_contract.profile or settings.RECALL_DEFAULT_PROFILE
    q = _query_from_contract(req_contract)
    started = time.perf_counter()
    result = run_recall_pipeline(q)
    latency_ms = int((time.perf_counter() - start_ts) * 1000)

    # Map internal fragments to strict MemoryItemV1
    items = []
    for fr in result.fragments:
        items.append(MemoryItemV1(
            id=fr.id,
            source=fr.source,
            snippet=fr.text,  # Mapping full text to snippet as per contract
            score=fr.salience or 0.0,
            ts=fr.ts.timestamp() if fr.ts else None,
            tags=fr.tags or [],
            title=fr.meta.get("title") if fr.meta else None
        ))

    out = BaseEnvelope(
        kind=RECALL_REPLY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=RecallReplyV1(
            correlation_id=str(env.correlation_id),
            bundle=MemoryBundleV1(
                rendered=f"Recalled {len(items)} items",
                items=items,
                stats=MemoryBundleStatsV1(
                    latency_ms=latency_ms,
                    backend_counts=result.debug.get("counts", {}) if result.debug else {}
                )
            )
        ).model_dump(mode="json"),
    )
    return out
