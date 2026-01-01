from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, Envelope, RecallRequestPayload, RecallResultPayload, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig

from .pipeline import run_recall_pipeline
from .types import RecallQuery
from .settings import settings

logger = logging.getLogger("orion-recall")


def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=bool(settings.ORION_BUS_ENABLED),
        heartbeat_interval_sec=float(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10.0)),
        health_channel=getattr(settings, "HEALTH_CHANNEL", "system.health"),
        error_channel=getattr(settings, "ERROR_CHANNEL", "system.error"),
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


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    """Rabbit handler: validate envelope, run recall, return typed result envelope."""

    # Strict kind check before validation
    if env.kind not in ("recall.query.request", "legacy.message"):
        return BaseEnvelope(
            kind="recall.query.result",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message":
        # legacy callers often send {text/query/max_items/...} without an envelope
        payload_obj = payload_obj.get("payload") or payload_obj

    # If it is legacy.message, we might want to pretend it is recall.query.request for schema validation purposes?
    # Or just validate the payload against RecallRequestPayload.
    # The previous code overwrote 'kind' in the input dict to force validation to pass Envelope[RecallRequestPayload].
    # That is actually okay IF we trust the check above.

    # However, to be cleaner, we can manually validate the payload into RecallRequestPayload
    # instead of casting the whole Envelope to Envelope[RecallRequestPayload].

    try:
        req_payload = RecallRequestPayload.model_validate(payload_obj)
    except ValidationError as ve:
         return BaseEnvelope(
            kind="recall.query.result",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    q = _query_from_payload(req_payload)
    result = run_recall_pipeline(q)

    fragments = [
        {
            "id": fr.id,
            "kind": fr.kind,
            "source": fr.source,
            "text": fr.text,
            "ts": fr.ts,
            "tags": fr.tags,
            "salience": fr.salience,
            "valence": fr.valence,
            "arousal": fr.arousal,
            "meta": fr.meta,
        }
        for fr in result.fragments
    ]

    out = Envelope[RecallResultPayload](
        kind="recall.query.result",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=RecallResultPayload(fragments=fragments, debug=result.debug),
    )
    return out.model_copy(update={"reply_to": None})
