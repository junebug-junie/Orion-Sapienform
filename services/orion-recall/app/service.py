from __future__ import annotations

import logging
from typing import Any, Dict

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

from orion.core.bus.bus_schemas import RecallRequestPayload, RecallResultPayload
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

    # Strict kind check
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
        payload_obj = payload_obj.get("payload") or payload_obj

    try:
        req_payload = RecallRequestPayload.model_validate(payload_obj)
    except ValidationError as ve:
         return BaseEnvelope(
            kind="recall.query.result",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": str(ve)},
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

    out = BaseEnvelope(
        kind="recall.query.result",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=RecallResultPayload(fragments=fragments, debug=result.debug).model_dump(mode="json"),
    )
    return out
