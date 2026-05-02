from __future__ import annotations

import asyncio

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.world_pulse import (
    ClaimRecordV1,
    EntityRecordV1,
    EventRecordV1,
    SituationChangeV1,
    TopicSituationBriefV1,
    WorldLearningDeltaV1,
    WorldPulseRunResultV1,
)

from app.services.emit_sql import build_sql_envelopes
from app.settings import settings


def _source_ref() -> ServiceRef:
    return ServiceRef(name=settings.service_name, node=settings.node_name, version=settings.service_version)


async def _publish_envelopes(run_result: WorldPulseRunResultV1) -> int:
    claims = [ClaimRecordV1.model_validate(c) for c in (run_result.publish_status or {}).get("claims", [])]
    events = [EventRecordV1.model_validate(e) for e in (run_result.publish_status or {}).get("events", [])]
    entities = [EntityRecordV1.model_validate(e) for e in (run_result.publish_status or {}).get("entities", [])]
    briefs = [TopicSituationBriefV1.model_validate(b) for b in (run_result.publish_status or {}).get("briefs", [])]
    changes = [SituationChangeV1.model_validate(c) for c in (run_result.publish_status or {}).get("changes", [])]
    learning = [WorldLearningDeltaV1.model_validate(l) for l in (run_result.publish_status or {}).get("learning", [])]

    envelopes = build_sql_envelopes(
        source_ref=_source_ref(),
        run_result=run_result,
        claims=claims,
        events=events,
        entities=entities,
        briefs=briefs,
        changes=changes,
        learning=learning,
    )
    bus = OrionBusAsync(
        settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
        enforce_catalog=settings.orion_bus_enforce_catalog,
    )
    await bus.connect()
    try:
        for channel, envelope in envelopes:
            await bus.publish(channel, envelope)
    finally:
        await bus.close()
    return len(envelopes)


def emit_sql_runtime(run_result: WorldPulseRunResultV1) -> dict:
    if not settings.world_pulse_sql_enabled:
        return {"ok": False, "status": "disabled", "count": 0}
    if run_result.run.dry_run:
        return {"ok": True, "status": "dry_run", "count": 0}
    try:
        count = asyncio.run(_publish_envelopes(run_result))
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": str(exc), "count": 0}
    return {"ok": True, "status": "published", "count": count}
