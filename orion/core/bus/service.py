import json
import logging
import os
import redis

logger = logging.getLogger("orionbus")

class OrionBus:
    """Minimal Redis client wrapper for Orion Mesh telemetry + events."""

    def __init__(self, url: str | None = None, enabled: bool | None = None):
        self.url = url or os.getenv("ORION_BUS_URL", "redis://orion-redis:6379/0")
        self.enabled = (
            str(enabled).lower() == "true"
            if enabled is not None
            else os.getenv("ORION_BUS_ENABLED", "true").lower() == "true"
        )
        self.client = None

        if self.enabled:
            try:
                self.client = redis.Redis.from_url(self.url, decode_responses=True)
                self.client.ping()
                logger.info(f"Connected to Orion bus at {self.url}")
            except Exception as e:
                logger.error(f"Failed to connect to Orion bus at {self.url}: {e}")
                self.enabled = False

    def publish(self, channel: str, message: dict):
        if not self.enabled or not self.client:
            return
        try:
            self.client.publish(channel, json.dumps(message))
            logger.debug(f"[BUS] Published to {channel}: {message}")
        except Exception as e:
            logger.error(f"Publish error on {channel}: {e}")

    def subscribe(self, channel: str):
        """Blocking subscribe generator."""
        if not self.enabled or not self.client:
            yield from ()
            return
        pubsub = self.client.pubsub()
        pubsub.subscribe(channel)
        logger.info(f"Subscribed to channel: {channel}")

        print(f"[BUS] Entering blocking listen() loop for channel: {channel}", flush=True)

        for message in pubsub.listen():

            print(f"!!! [BUS] RAW MESSAGE RECEIVED on {channel}: {message}", flush=True)

            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
            except Exception as e:
                logger.error(f"[BUS] JSON parse failed on {channel}: {e} — raw={message['data']!r}")
                continue
            yield data
