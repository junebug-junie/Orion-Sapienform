from __future__ import annotations

import asyncio

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.world_pulse import HubWorldPulseMessageV1

from app.settings import settings


def _source_ref() -> ServiceRef:
    return ServiceRef(name=settings.service_name, node=settings.node_name, version=settings.service_version)


def _hub_envelope(message: HubWorldPulseMessageV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind="hub.messages.create.v1",
        source=_source_ref(),
        payload=message.model_dump(mode="json"),
    )


async def _publish_hub_envelope(channel: str, envelope: BaseEnvelope) -> None:
    bus = OrionBusAsync(
        settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
        enforce_catalog=settings.orion_bus_enforce_catalog,
    )
    await bus.connect()
    try:
        await bus.publish(channel, envelope)
    finally:
        await bus.close()


def publish_hub_message(*, message: HubWorldPulseMessageV1, dry_run: bool | None = None) -> dict:
    channel = settings.world_pulse_hub_message_channel
    envelope = _hub_envelope(message)
    effective_dry_run = settings.world_pulse_dry_run if dry_run is None else dry_run
    if effective_dry_run:
        return {
            "ok": True,
            "status": "dry_run",
            "would_publish": True,
            "channel": channel,
            "kind": envelope.kind,
            "payload_preview": {
                "message_id": message.message_id,
                "run_id": message.run_id,
                "title": message.title,
                "date": message.date,
                "executive_summary": message.executive_summary,
            },
        }
    try:
        asyncio.run(_publish_hub_envelope(channel, envelope))
    except Exception as exc:
        return {"ok": False, "status": "failed", "channel": channel, "kind": envelope.kind, "error": str(exc)}
    return {"ok": True, "status": "published", "channel": channel, "kind": envelope.kind}
