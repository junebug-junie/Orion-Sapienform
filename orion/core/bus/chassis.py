import asyncio
import signal
import logging
import json
import os
import sys
from typing import Any, Awaitable, Callable, Dict, Optional, Type
import redis.asyncio as redis
from pydantic import BaseModel

from .schemas import BaseEnvelope

logger = logging.getLogger("orion.chassis")

class ServiceChassis:
    """
    The Invisible Chassis: Abstracts Redis, Loops, and Signals.
    """
    def __init__(self, service_name: str, bus_url: str = None):
        self.service_name = service_name
        self.bus_url = bus_url or os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
        self.enabled = os.getenv("ORION_BUS_ENABLED", "true").lower() == "true"
        self._redis: Optional[redis.Redis] = None
        self._pubsub = None
        self._shutdown_event = asyncio.Event()
        self._handlers: Dict[str, Callable[[BaseEnvelope], Awaitable[Any]]] = {}
        self._loop_task: Optional[Callable[[], Awaitable[None]]] = None

    async def connect(self):
        """Initializes the Redis connection."""
        if not self.enabled:
            logger.warning("Bus disabled by config.")
            return

        logger.info(f"Connecting to Bus at {self.bus_url}...")
        try:
            self._redis = redis.from_url(self.bus_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Bus connected.")
        except Exception as e:
            logger.error(f"Failed to connect to bus: {e}")
            sys.exit(1)

    async def close(self):
        """Closes the Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Bus connection closed.")

    def register_rpc(self, channel: str, handler: Callable[[BaseEnvelope], Awaitable[Any]]):
        """Registers a handler for RPC-style requests (Rabbit Pattern)."""
        self._handlers[channel] = handler

    def register_consumer(self, channel: str, handler: Callable[[BaseEnvelope], Awaitable[None]]):
        """Registers a handler for Fire-and-Forget messages (Hunter Pattern)."""
        self._handlers[channel] = handler

    def register_loop(self, loop_func: Callable[[], Awaitable[None]]):
        """Registers a custom background loop (Clock Pattern)."""
        self._loop_task = loop_func

    async def _heartbeat_loop(self):
        """Publishes system.health heartbeats."""
        while not self._shutdown_event.is_set():
            if self._redis:
                payload = {
                    "service": self.service_name,
                    "status": "ok",
                    "pid": os.getpid()
                }
                # We use a raw publish for health to avoid infinite recursion/complexity
                await self._redis.publish("system.health", json.dumps(payload))
            await asyncio.sleep(5)

    async def _message_handler_loop(self):
        """Main loop for consuming messages from Redis."""
        if not self._handlers or not self._redis:
            return

        self._pubsub = self._redis.pubsub()
        channels = list(self._handlers.keys())
        await self._pubsub.subscribe(*channels)
        logger.info(f"Subscribed to {channels}")

        async for msg in self._pubsub.listen():
            if self._shutdown_event.is_set():
                break

            if msg["type"] != "message":
                continue

            channel = msg["channel"]
            data_str = msg["data"]

            try:
                # 1. Parse Raw JSON
                raw_dict = json.loads(data_str)

                # 2. Legacy Support (Upcasting)
                # Map trace_id -> correlation_id if missing
                if "correlation_id" not in raw_dict and "trace_id" in raw_dict:
                    raw_dict["correlation_id"] = raw_dict["trace_id"]
                # 3. Inflate to Envelope (Titanium Contract)
                envelope = BaseEnvelope[Any](**raw_dict)

                # 4. Causality Tracking
                envelope.add_causality(self.service_name, "consume")

                # 5. Invoke Handler
                handler = self._handlers.get(channel)
                if handler:
                    result = await handler(envelope)

                    # 6. Handle RPC Reply
                    if envelope.reply_channel and result is not None:
                        await self.publish(envelope.reply_channel, result, correlation_id=envelope.correlation_id)

            except Exception as e:
                logger.exception(f"Error processing message on {channel}: {e}")

    async def publish(self, channel: str, payload: Any, correlation_id: str = None):
        """Publishes a message to the bus."""
        if not self._redis:
            return

        # If payload is already an Envelope, dump it. Otherwise wrap it.
        if isinstance(payload, BaseEnvelope):
            envelope = payload
        else:
            envelope = BaseEnvelope(
                event="publish", # Generic event if not specified
                source=self.service_name,
                correlation_id=correlation_id or "new",
                payload=payload
            )

        await self._redis.publish(channel, envelope.model_dump_json())

    async def run(self):
        """Entry point: starts loops, handles signals, waits for shutdown."""
        # Setup Signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown_event.set())

        await self.connect()

        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._message_handler_loop())
        ]

        if self._loop_task:
            tasks.append(asyncio.create_task(self._loop_task()))

        logger.info(f"{self.service_name} Chassis Started. Waiting for signals...")

        # Wait until shutdown signal
        await self._shutdown_event.wait()

        logger.info("Shutdown signal received. Cleaning up...")
        for t in tasks:
            t.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        await self.close()
        logger.info("Chassis Shutdown Complete.")
