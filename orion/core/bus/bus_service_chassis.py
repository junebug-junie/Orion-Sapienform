"""orion.core.bus.bus_service_chassis

The "Invisible" Chassis.

Three service patterns that hide Redis loops entirely:

  - Rabbit: RPC / request→reply (synchronous semantics over pubsub)
  - Hunter: async fire-and-forget consumer (pattern subscriptions)
  - Clock: periodic crawler/ticker (safe from overlapping ticks)

All patterns provide, by default:
  - Graceful shutdown on SIGINT/SIGTERM
  - Heartbeats (system.health)
  - Error wrapping + emission (system.error)

Constraints:
  - redis-py asyncio (redis.asyncio)
  - Pydantic v2

Notes:
  - PubSub gives no delivery guarantees. This chassis focuses on ergonomic
    correctness (typed envelopes, shutdown, health, error reporting).
    If/when you move to Redis Streams, these patterns remain the API layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, Iterable, Optional, Type, TypeVar

from pydantic import BaseModel

from .bus_schemas import (
    BaseEnvelope,
    SystemHealthPayload,
    SystemErrorPayload,
    build_error_payload,
    utcnow,
)
from .service_async import OrionBusAsync


logger = logging.getLogger("orion.chassis")

ReqEnvT = TypeVar("ReqEnvT", bound=BaseModel)
RespT = TypeVar("RespT")


@dataclass(frozen=True)
class ChassisConfig:
    service_name: str
    service_version: str = "0.1.0"

    # Bus
    bus_url: str = os.getenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
    bus_enabled: bool = os.getenv("ORION_BUS_ENABLED", "true").lower() == "true"

    # Health / error
    health_channel: str = os.getenv("ORION_HEALTH_CHANNEL", "orion:system:health")
    error_channel: str = os.getenv("ORION_ERROR_CHANNEL", "orion:system:error")
    heartbeat_sec: float = float(os.getenv("ORION_HEARTBEAT_SEC", "30"))

    # PubSub polling
    poll_timeout_sec: float = float(os.getenv("ORION_PUBSUB_POLL_SEC", "1.0"))


class BaseChassis:
    """Common lifecycle features for all chassis patterns."""

    def __init__(self, config: ChassisConfig, *, bus: OrionBusAsync | None = None):
        self.config = config
        self.bus = bus or OrionBusAsync(url=config.bus_url, enabled=config.bus_enabled)

        self._stop = asyncio.Event()
        self._started_at = time.monotonic()
        self._tasks: list[asyncio.Task] = []

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.bus.connect()
        self._install_signal_handlers()
        try:
            await self._publish_health(status="starting")
        except Exception as e:
            logger.warning("[health] failed during start: %s", e)
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="heartbeat"))

    async def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            await self._publish_health(status="stopping")
        except Exception as e:
            logger.warning("[health] failed during stop: %s", e)

        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        await self.bus.close()

    async def run(self) -> None:
        """Run the chassis until stopped."""
        await self.start()
        try:
            await self._run_impl()
        finally:
            await self.stop()

    async def _run_impl(self) -> None:
        raise NotImplementedError

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop.set)
        except NotImplementedError:
            # Windows or embedded loops
            pass
        except RuntimeError:
            # No running loop (unit tests)
            pass

    # ────────────────────────────────────────────────────────────────────
    # Health + errors
    # ────────────────────────────────────────────────────────────────────

    def uptime_s(self) -> float:
        return time.monotonic() - self._started_at

    async def _publish_health(self, *, status: str, detail: Dict[str, Any] | None = None) -> None:
        payload = SystemHealthPayload(status=status, uptime_s=self.uptime_s(), detail=detail or {})
        env = BaseEnvelope[SystemHealthPayload](
            event="system.health",
            service=self.config.service_name,
            correlation_id=f"health-{self.config.service_name}",
            ts=utcnow(),
            payload=payload,
        )
        await self.bus.publish(self.config.health_channel, env.to_bus_dict())

    async def _publish_error(self, exc: BaseException, *, context: Dict[str, Any] | None = None) -> None:
        payload = build_error_payload(exc, context=context)
        env = BaseEnvelope[SystemErrorPayload](
            event="system.error",
            service=self.config.service_name,
            correlation_id=f"error-{self.config.service_name}-{int(time.time())}",
            ts=utcnow(),
            payload=payload,
        )
        await self.bus.publish(self.config.error_channel, env.to_bus_dict())

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._publish_health(status="ok")
            except Exception as e:
                # Health shouldn't crash the service.
                logger.warning("[health] failed: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.heartbeat_sec)
            except asyncio.TimeoutError:
                continue


class Rabbit(BaseChassis, Generic[ReqEnvT, RespT]):
    """RPC-style request→reply over PubSub.

    Contract:
      - listen on `intake_channel`
      - incoming JSON must validate as `request_model` (or be a compatible legacy envelope)
      - response is published to `reply_to` taken from the request
    """

    def __init__(
        self,
        config: ChassisConfig,
        *,
        intake_channel: str,
        request_model: Type[ReqEnvT],
        handler: Callable[[ReqEnvT], Awaitable[RespT]],
        response_publisher: Callable[[ReqEnvT, RespT], Dict[str, Any]] | None = None,
        bus: OrionBusAsync | None = None,
    ):
        super().__init__(config, bus=bus)
        self.intake_channel = intake_channel
        self.request_model = request_model
        self.handler = handler
        self.response_publisher = response_publisher

    async def _run_impl(self) -> None:
        if not self.bus.enabled:
            logger.error("Rabbit %s: bus disabled", self.config.service_name)
            return

        pubsub = self.bus.pubsub()
        await pubsub.subscribe(self.intake_channel)
        logger.info("[%s] Rabbit listening on %s", self.config.service_name, self.intake_channel)

        try:
            while not self._stop.is_set():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=self.config.poll_timeout_sec)
                if not msg:
                    continue

                try:
                    data = msg.get("data")
                    if isinstance(data, str):
                        import json

                        raw = json.loads(data)
                    else:
                        raw = data

                    req = self.request_model.model_validate(raw)
                    result = await self.handler(req)

                    reply_to = getattr(req, "reply_to", None) or getattr(req, "reply_channel", None) or getattr(req, "response_channel", None)
                    if not reply_to:
                        logger.warning("[%s] request missing reply_to/reply_channel", self.config.service_name)
                        continue

                    if self.response_publisher:
                        out = self.response_publisher(req, result)
                    else:
                        # Default: if handler returns a dict, publish it.
                        if isinstance(result, BaseModel):
                            out = result.model_dump(mode="json")
                        elif isinstance(result, dict):
                            out = result
                        else:
                            out = {"result": result}

                    await self.bus.publish(str(reply_to), out)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self._publish_error(e, context={"intake": self.intake_channel})
        finally:
            await pubsub.close()


class Hunter(BaseChassis, Generic[ReqEnvT]):
    """Fire-and-forget consumer.

    - pattern subscriptions (psubscribe)
    - optional filter function
    """

    def __init__(
        self,
        config: ChassisConfig,
        *,
        patterns: Iterable[str],
        message_model: Type[ReqEnvT],
        handler: Callable[[ReqEnvT], Awaitable[None]],
        predicate: Callable[[str, Dict[str, Any]], bool] | None = None,
        bus: OrionBusAsync | None = None,
    ):
        super().__init__(config, bus=bus)
        self.patterns = list(patterns)
        self.message_model = message_model
        self.handler = handler
        self.predicate = predicate

    async def _run_impl(self) -> None:
        if not self.bus.enabled:
            logger.error("Hunter %s: bus disabled", self.config.service_name)
            return

        pubsub = self.bus.pubsub()
        await pubsub.psubscribe(*self.patterns)
        logger.info("[%s] Hunter psubscribed to %s", self.config.service_name, self.patterns)

        try:
            while not self._stop.is_set():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=self.config.poll_timeout_sec)
                if not msg:
                    continue

                try:
                    channel = (msg.get("channel") or msg.get("pattern") or "").decode() if isinstance(msg.get("channel"), (bytes, bytearray)) else str(msg.get("channel") or msg.get("pattern") or "")

                    data = msg.get("data")
                    if isinstance(data, str):
                        import json

                        raw = json.loads(data)
                    else:
                        raw = data

                    if self.predicate and not self.predicate(channel, raw):
                        continue

                    env = self.message_model.model_validate(raw)
                    await self.handler(env)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self._publish_error(e, context={"patterns": self.patterns})
        finally:
            await pubsub.close()


class Clock(BaseChassis):
    """Ticker-style worker.

    Calls `tick()` every `interval_sec`.
    Ensures ticks don't overlap (race-safe).
    """

    def __init__(
        self,
        config: ChassisConfig,
        *,
        interval_sec: float,
        tick: Callable[[], Awaitable[None]],
        jitter_sec: float = 0.0,
        bus: OrionBusAsync | None = None,
    ):
        super().__init__(config, bus=bus)
        self.interval_sec = interval_sec
        self.jitter_sec = jitter_sec
        self.tick = tick
        self._lock = asyncio.Lock()

    async def _run_impl(self) -> None:
        logger.info("[%s] Clock started interval=%.2fs", self.config.service_name, self.interval_sec)
        while not self._stop.is_set():
            try:
                async with self._lock:
                    await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._publish_error(e, context={"interval_sec": self.interval_sec})

            # Sleep with stop-awareness
            sleep_for = float(self.interval_sec)
            if self.jitter_sec:
                import random

                sleep_for += random.uniform(0.0, self.jitter_sec)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                continue
