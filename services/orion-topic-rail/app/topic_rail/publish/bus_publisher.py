from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.platform import SystemErrorV1
from orion.schemas.topic import TopicSummaryEventV1, TopicShiftEventV1


logger = logging.getLogger("topic-rail.bus")


class TopicRailBusPublisher:
    def __init__(self, *, bus_url: str, service_name: str, node_name: str, service_version: str) -> None:
        self.bus = OrionBusAsync(url=bus_url)
        self.service_ref = ServiceRef(name=service_name, node=node_name, version=service_version)

    async def publish_summary_rows(self, channel: str, rows: Iterable[Dict[str, Any]]) -> int:
        await self.bus.connect()
        sent = 0
        try:
            for row in rows:
                payload = TopicSummaryEventV1(**row)
                env = BaseEnvelope(
                    kind="topic.summary.v1",
                    source=self.service_ref,
                    payload=payload.model_dump(mode="json"),
                )
                await self.bus.publish(channel, env)
                sent += 1
        finally:
            await self.bus.close()
        logger.info("Published %s topic summaries", sent)
        return sent

    async def publish_shift_rows(self, channel: str, rows: Iterable[Dict[str, Any]]) -> int:
        await self.bus.connect()
        sent = 0
        try:
            for row in rows:
                payload = TopicShiftEventV1(**row)
                env = BaseEnvelope(
                    kind="topic.shift.v1",
                    source=self.service_ref,
                    payload=payload.model_dump(mode="json"),
                )
                await self.bus.publish(channel, env)
                sent += 1
        finally:
            await self.bus.close()
        logger.info("Published %s topic shifts", sent)
        return sent

    async def publish_warning(self, message: str, details: Dict[str, Any]) -> None:
        await self.bus.connect()
        try:
            payload = SystemErrorV1(error=message, details=details)
            env = BaseEnvelope(
                kind="system.error",
                source=self.service_ref,
                payload=payload.model_dump(mode="json"),
            )
            await self.bus.publish("orion:system:error", env)
        finally:
            await self.bus.close()
