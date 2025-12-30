from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import ValidationError

from orion.core.bus.bus_schemas import (
    BaseEnvelope,
    Envelope,
    RecallRequestPayload,
    RecallResultPayload,
    ServiceRef,
)
from orion.core.bus.bus_service_chassis import ChassisConfig

from .pipeline import run_recall_pipeline
from .types import RecallQuery
from .settings import settings

logger = logging.getLogger("orion-recall")


def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=getattr(settings, "ORION_NODE_NAME", None),
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=bool(settings.ORION_BUS_ENABLED),
        heartbeat_interval_sec=float(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10.0)),
        health_channel=getattr(settings, "HEALTH_CHANNEL", "system.health"),
        error_channel=getattr(settings, "ERROR_CHANNEL", "system.error"),
        shutdown_timeout_sec=float(getattr(settings, "SHUTDOWN_GRACE_SEC", 10.0)),
    )

def _service_ref() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=getattr(settings, "ORION_NODE_NAME", None),
    )


def _query_from_payload(p: RecallRequestPayload) -> RecallQuery:
    return RecallQuery(
        text=p.query_text,
        max_items=int(p.max_items),
        time_window_days=int(p.time_window_days),
        mode=str(p.mode),
        tags=list(p.tags or []),
        phi=None,
        trace_id=p.trace_id,
    )


async def handle(env: BaseEnvelope) -> BaseEnvelope:
    """Rabbit handler: validate envelope, run recall, return typed result envelope."""

    if env.kind not in ("recall.query.request", "legacy.message"):
        return Envelope[RecallResultPayload](
            kind="recall.query.result",
            source=_service_ref(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=RecallResultPayload(ok=False, error=f"unsupported_kind:{env.kind}"),
        ).model_copy(update={"reply_to": None})

    payload_obj: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message":
        # legacy callers often send {text/query/max_items/...} without an envelope
        payload_obj = payload_obj.get("payload") or payload_obj

    try:
        typed = Envelope[RecallRequestPayload].model_validate(
            {**env.model_dump(), "kind": "recall.query.request", "payload": payload_obj}
        )
    except ValidationError as ve:
        return Envelope[RecallResultPayload](
            kind="recall.query.result",
            source=_service_ref(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=RecallResultPayload(ok=False, error="validation_failed", debug={"errors": ve.errors()}),
        ).model_copy(update={"reply_to": None})

    q = _query_from_payload(typed.payload)
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
        source=_service_ref(),
        correlation_id=typed.correlation_id,
        causality_chain=typed.causality_chain,
        payload=RecallResultPayload(ok=True, fragments=fragments, debug=result.debug),
    )
    return out.model_copy(update={"reply_to": None})
