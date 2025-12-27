"""orion.core.bus.service_async

Async Redis client wrapper for Orion Bus.

This intentionally stays tiny:
  - connect / close
  - publish JSON
  - create PubSub objects

The higher-level service patterns (Rabbit/Hunter/Clock) live in
`bus_service_chassis.py`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import redis.asyncio as redis


logger = logging.getLogger("orionbus.async")


class OrionBusAsync:
    """Async Redis bus wrapper (redis-py asyncio).

    Defaults are intentionally opinionated (convention over config).
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        enabled: bool | None = None,
        decode_responses: bool = True,
    ):
        self.url = url or os.getenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
        self.enabled = (
            str(enabled).lower() == "true"
            if enabled is not None
            else os.getenv("ORION_BUS_ENABLED", "true").lower() == "true"
        )
        self.decode_responses = decode_responses
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        if not self.enabled:
            return
        if self.client is not None:
            return
        try:
            self.client = redis.Redis.from_url(self.url, decode_responses=self.decode_responses)
            await self.client.ping()
            logger.info("Connected to Orion bus at %s", self.url)
        except Exception as e:
            logger.error("Failed to connect to Orion bus at %s: %s", self.url, e)
            self.client = None
            self.enabled = False

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            await self.client.close()
        finally:
            self.client = None

    async def publish(self, channel: str, message: Dict[str, Any]) -> None:
        """Publish a dict as JSON to `channel`."""
        if not self.enabled or self.client is None:
            return
        try:
            await self.client.publish(channel, json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error("Publish error on %s: %s", channel, e)

    def pubsub(self) -> "redis.client.PubSub":
        if not self.enabled or self.client is None:
            raise RuntimeError("OrionBusAsync is disabled or not connected")
        return self.client.pubsub()
