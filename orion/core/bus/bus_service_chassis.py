# orion/core/bus/bus_service_chassis.py
from __future__ import annotations

import asyncio
import signal
import traceback
from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from .async_service import OrionBusAsync
from .bus_schemas import BaseEnvelope, ErrorInfo, ServiceRef


Handler = Callable[[BaseEnvelope], Awaitable[BaseEnvelope | None]]


@dataclass(frozen=True)
class ChassisConfig:
    service_name: str
    service_version: str
    node_name: str
    bus_url: str = "redis://100.92.216.81:6379/0"
    bus_enabled: bool = True

    # system behaviors
    heartbeat_interval_sec: float = 10.0
    connect_timeout_sec: float = 10.0
    shutdown_timeout_sec: float = 10.0

    # system channels (stable defaults)
    health_channel: str = "system.health"
    error_channel: str = "system.error"


class _SystemHealthPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str = "ok"
    service: str
    node: str
    version: str
    details: dict[str, Any] = Field(default_factory=dict)


class BaseChassis:
    """
    Shared chassis behavior:

    - bus connect/disconnect + SIGTERM shutdown
    - periodic heartbeat publishing
    - exception wrapping to system.error
    """

    def __init__(self, cfg: ChassisConfig):
        self.cfg = cfg
        self.bus = OrionBusAsync(cfg.bus_url, enabled=cfg.bus_enabled)

        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._started = False

    def _source(self) -> ServiceRef:
        return ServiceRef(name=self.cfg.service_name, version=self.cfg.service_version, node=self.cfg.node_name)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._install_signal_handlers()

        # IMPORTANT: compute timeout before creating coroutine, so failures don't leak "never awaited".
        timeout = float(self.cfg.connect_timeout_sec or 10.0)
        logger.info(f"Connecting bus url={self.cfg.bus_url}")
        await asyncio.wait_for(self.bus.connect(), timeout=timeout)

        # start system heartbeat
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="orion-heartbeat"))

        # run subclass logic
        self._tasks.append(asyncio.create_task(self._run(), name=f"{self.cfg.service_name}-run"))

        await self._stop.wait()
        await self._shutdown()

    async def stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _handler() -> None:
            logger.warning("SIGTERM/SIGINT received: shutting down")
            self._stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handler)
            except NotImplementedError:
                # Windows / limited environments
                signal.signal(sig, lambda *_: _handler())

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                node = self.cfg.node_name or "unknown"
                env = BaseEnvelope(
                    kind="system.health",
                    source=self._source(),
                    payload=_SystemHealthPayload(
                        service=self.cfg.service_name,
                        node=node,
                        version=self.cfg.service_version,
                        details={},
                    ).model_dump(mode="json"),
                )
                await self.bus.publish(self.cfg.health_channel, env)
            except Exception as e:
                # Don't recurse into _publish_error if the bus is down; just log.
                logger.warning(f"Heartbeat publish failed: {e}")
            await asyncio.sleep(float(self.cfg.heartbeat_interval_sec or 10.0))

    async def _publish_error(self, err: BaseException, *, when: str, env: BaseEnvelope | None = None) -> None:
        try:
            info = ErrorInfo(
                type=type(err).__name__,
                message=str(err),
                stack="".join(traceback.format_exception(type(err), err, err.__traceback__)),
                details={"when": when},
            )
            payload = info.model_dump(mode="json")
            out = BaseEnvelope(
                kind="system.error",
                source=self._source(),
                correlation_id=(env.correlation_id if env else uuid4()),
                causality_chain=(env.causality_chain if env else []),
                payload=payload,
            )
            await self.bus.publish(self.cfg.error_channel, out)
        except Exception:
            logger.exception("Failed publishing system.error")

    async def _shutdown(self) -> None:
        # cancel tasks
        for t in self._tasks:
            if not t.done():
                t.cancel()

        try:
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=float(self.cfg.shutdown_timeout_sec or 10.0))
        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout waiting for tasks")

        try:
            await self.bus.close()
        except Exception:
            logger.exception("Bus close failed")

    async def _run(self) -> None:
        raise NotImplementedError


class Rabbit(BaseChassis):
    """
    RPC / synchronous pattern.
    Listens on a single request channel and replies to reply_to.
    """

    def __init__(self, cfg: ChassisConfig, *, request_channel: str, handler: Handler):
        super().__init__(cfg)
        self.request_channel = request_channel
        self.handler = handler

    async def _run(self) -> None:
        logger.info(f"Rabbit listening channel={self.request_channel} bus={self.cfg.bus_url}")

        async with self.bus.subscribe(self.request_channel) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                if not isinstance(msg, dict):
                    continue
                data = msg.get("data")
                if data is None:
                    continue

                decoded = self.bus.codec.decode(data)
                if not decoded.ok or decoded.envelope is None:
                    await self._publish_error(
                        RuntimeError(decoded.error or "decode_failed"),
                        when="rabbit.decode",
                        env=None,
                    )
                    continue

                env = decoded.envelope
                try:
                    out = await self.handler(env)
                    if out is not None and env.reply_to:
                        await self.bus.publish(env.reply_to, out)
                except Exception as e:
                    await self._publish_error(e, when="rabbit.handle", env=env)


class Hunter(BaseChassis):
    """
    Fire-and-forget consumer. Subscribes to patterns, filters, and acts.

    Uses Redis PSUBSCRIBE behind the scenes (patterns=True).
    """

    def __init__(self, cfg: ChassisConfig, *, pattern: str, handler: Callable[[BaseEnvelope], Awaitable[None]]):
        super().__init__(cfg)
        self.pattern = pattern
        self.handler = handler

    async def _run(self) -> None:
        logger.info(f"Hunter subscribing pattern={self.pattern} bus={self.cfg.bus_url}")

        async with self.bus.subscribe(self.pattern, patterns=True) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                if not isinstance(msg, dict):
                    continue
                data = msg.get("data")
                if data is None:
                    continue

                decoded = self.bus.codec.decode(data)
                if not decoded.ok or decoded.envelope is None:
                    await self._publish_error(
                        RuntimeError(decoded.error or "decode_failed"),
                        when="hunter.decode",
                        env=None,
                    )
                    continue

                env = decoded.envelope
                try:
                    await self.handler(env)
                except Exception as e:
                    await self._publish_error(e, when="hunter.handle", env=env)


class Clock(BaseChassis):
    """
    Periodic ticker/loop with safe cancellation.
    """

    def __init__(self, cfg: ChassisConfig, *, interval_sec: float, tick: Callable[[], Awaitable[None]]):
        super().__init__(cfg)
        self.interval_sec = float(interval_sec)
        self.tick = tick

    async def _run(self) -> None:
        logger.info(f"Clock starting interval={self.interval_sec}s bus={self.cfg.bus_url}")
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception as e:
                await self._publish_error(e, when="clock.tick", env=None)
            await asyncio.sleep(self.interval_sec)
