from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.topic_foundry import (
    KgEdgeIngestV1,
    TopicFoundryDriftAlertV1,
    TopicFoundryEnrichCompleteV1,
    TopicFoundryRunCompleteV1,
)

from app.settings import settings


logger = logging.getLogger("topic-foundry.bus")

RUN_COMPLETE_CHANNEL = "orion:topic:foundry:run:complete.v1"
ENRICH_COMPLETE_CHANNEL = "orion:topic:foundry:enrich:complete.v1"
DRIFT_ALERT_CHANNEL = "orion:topic:foundry:drift:alert.v1"
KG_EDGE_INGEST_CHANNEL = "orion:kg:edge:ingest.v1"


def _safe_run(coro) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bus publish failed: %s", exc)
    else:
        loop.create_task(_wrap_publish(coro))


async def _wrap_publish(coro) -> None:
    try:
        await coro
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bus publish failed: %s", exc)


class TopicFoundryBusPublisher:
    def __init__(self) -> None:
        self._enabled = settings.orion_bus_enabled
        self._bus_url = settings.orion_bus_url
        self._service_ref = ServiceRef(
            name=settings.service_name,
            node=settings.node_name,
            version=settings.service_version,
        )

    async def _publish(self, channel: str, kind: str, payload: Any) -> None:
        if not self._enabled:
            return
        bus = OrionBusAsync(url=self._bus_url)
        await bus.connect()
        try:
            env = BaseEnvelope(
                kind=kind,
                source=self._service_ref,
                payload=payload,
            )
            await bus.publish(channel, env)
        finally:
            await bus.close()

    def publish_run_complete(self, payload: TopicFoundryRunCompleteV1) -> None:
        _safe_run(
            self._publish(
                RUN_COMPLETE_CHANNEL,
                "topic.foundry.run.complete.v1",
                payload.model_dump(mode="json"),
            )
        )

    def publish_enrich_complete(self, payload: TopicFoundryEnrichCompleteV1) -> None:
        _safe_run(
            self._publish(
                ENRICH_COMPLETE_CHANNEL,
                "topic.foundry.enrich.complete.v1",
                payload.model_dump(mode="json"),
            )
        )

    def publish_drift_alert(self, payload: TopicFoundryDriftAlertV1) -> None:
        _safe_run(
            self._publish(
                DRIFT_ALERT_CHANNEL,
                "topic.foundry.drift.alert.v1",
                payload.model_dump(mode="json"),
            )
        )

    def publish_kg_edges(self, payload: KgEdgeIngestV1) -> None:
        _safe_run(
            self._publish(
                KG_EDGE_INGEST_CHANNEL,
                "kg.edge.ingest.v1",
                payload.model_dump(mode="json"),
            )
        )


_publisher: Optional[TopicFoundryBusPublisher] = None


def get_bus_publisher() -> TopicFoundryBusPublisher:
    global _publisher
    if _publisher is None:
        _publisher = TopicFoundryBusPublisher()
    return _publisher
