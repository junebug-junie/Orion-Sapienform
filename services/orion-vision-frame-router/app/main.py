from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync

from .dispatcher import FrameDispatcher
from .metrics import RouterMetrics, make_health_envelope
from .policy import FrameDispatchPolicy
from .settings import Settings
from .state import RouterState


class FrameRouterService:
    def __init__(
        self,
        settings: Settings | None = None,
        bus: OrionBusAsync | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.bus = bus or OrionBusAsync(
            url=self.settings.ORION_BUS_URL,
            enforce_catalog=self.settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self.state = RouterState()
        self.metrics = RouterMetrics()
        self.policy: FrameDispatchPolicy | None = None
        self.dispatcher: FrameDispatcher | None = None
        self._frames_task: Optional[asyncio.Task] = None
        self._reply_task: Optional[asyncio.Task] = None
        self._timeout_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)

        self.policy = FrameDispatchPolicy.load(self.settings)
        self.dispatcher = FrameDispatcher(
            settings=self.settings,
            policy=self.policy,
            state=self.state,
            metrics=self.metrics,
            bus=self.bus,
        )

        await self.bus.connect()
        self._shutdown.clear()
        self._frames_task = asyncio.create_task(self._frames_loop())
        self._reply_task = asyncio.create_task(self._reply_loop())
        self._timeout_task = asyncio.create_task(self._timeout_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(f"[FRAME-ROUTER] Started → {self.settings.CHANNEL_FRAMES_IN}")

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._frames_task, self._reply_task, self._timeout_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.bus.close()

    async def _frames_loop(self) -> None:
        assert self.dispatcher is not None
        async with self.bus.subscribe(self.settings.CHANNEL_FRAMES_IN) as pubsub:
            while not self._shutdown.is_set():
                try:
                    async for msg in self.bus.iter_messages(pubsub):
                        if self._shutdown.is_set():
                            break
                        data = msg.get("data")
                        if not data:
                            continue
                        decoded = self.bus.codec.decode(data)
                        if decoded.ok and decoded.envelope:
                            asyncio.create_task(self.dispatcher.handle_frame_envelope(decoded.envelope))
                except Exception as exc:
                    if not self._shutdown.is_set():
                        logger.error(f"[FRAME-ROUTER] frames consumer error: {exc}")
                        await asyncio.sleep(1)

    async def _reply_loop(self) -> None:
        assert self.dispatcher is not None
        pattern = f"{self.settings.CHANNEL_REPLY_PREFIX}:*"
        async with self.bus.subscribe(pattern, patterns=True) as pubsub:
            while not self._shutdown.is_set():
                try:
                    async for msg in self.bus.iter_messages(pubsub):
                        if self._shutdown.is_set():
                            break
                        data = msg.get("data")
                        if not data:
                            continue
                        decoded = self.bus.codec.decode(data)
                        if decoded.ok and decoded.envelope:
                            asyncio.create_task(self.dispatcher.handle_reply_envelope(decoded.envelope))
                except Exception as exc:
                    if not self._shutdown.is_set():
                        logger.error(f"[FRAME-ROUTER] reply consumer error: {exc}")
                        await asyncio.sleep(1)

    async def _timeout_loop(self) -> None:
        assert self.dispatcher is not None
        while not self._shutdown.is_set():
            try:
                self.dispatcher.sweep_timeouts(now=time.time())
            except Exception as exc:
                logger.warning(f"[FRAME-ROUTER] timeout sweep failed: {exc}")
            await asyncio.sleep(1.0)

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                env = make_health_envelope(
                    service_name=self.settings.SERVICE_NAME,
                    service_version=self.settings.SERVICE_VERSION,
                    router_enabled=self.settings.ROUTER_ENABLED,
                    dry_run=self.settings.DRY_RUN,
                    policy_path=self.settings.ROUTER_POLICY_PATH,
                    metrics=self.metrics,
                    state=self.state,
                )
                await self.bus.publish(self.settings.CHANNEL_SYSTEM_HEALTH, env)
            except Exception as exc:
                logger.warning(f"[FRAME-ROUTER] health publish failed: {exc}")
            await asyncio.sleep(self.settings.HEALTH_INTERVAL_SECONDS)

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self.metrics),
            "inflight_total": self.state.inflight_total(),
            "pending_count": len(self.state.pending),
            "router_enabled": self.settings.ROUTER_ENABLED,
            "dry_run": self.settings.DRY_RUN,
            "policy_path": self.settings.ROUTER_POLICY_PATH,
        }


service = FrameRouterService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()


app = FastAPI(title="Orion Vision Frame Router", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return service.metrics_snapshot()
