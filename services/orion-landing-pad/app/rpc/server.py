from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, Optional

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pad import KIND_PAD_RPC_REQUEST_V1, KIND_PAD_RPC_RESPONSE_V1, PadRpcRequestV1, PadRpcResponseV1

from ..observability.stats import PadStatsTracker
from ..settings import Settings
from ..store.redis_store import PadStore


class PadRpcServer:
    def __init__(self, *, bus: OrionBusAsync, store: PadStore, settings: Settings, stats: PadStatsTracker):
        self.bus = bus
        self.store = store
        self.settings = settings
        self.stats = stats
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def _source(self) -> ServiceRef:
        return ServiceRef(name=self.settings.app_name, version=self.settings.service_version, node=self.settings.node_name)

    async def start(self) -> None:
        self._stop.clear()
        logger.info("Starting pad RPC listener channel=%s", self.settings.pad_rpc_request_channel)
        self._task = asyncio.create_task(self._run(), name="pad-rpc")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run(self) -> None:
        async with self.bus.subscribe(self.settings.pad_rpc_request_channel) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                data = msg.get("data")
                if data is None:
                    continue
                decoded = self.bus.codec.decode(data)
                if not decoded.ok or decoded.envelope is None:
                    self.stats.increment_rpc_errors()
                    if decoded.error:
                        logger.warning("RPC decode failed error=%s", decoded.error)
                    continue
                env = decoded.envelope
                if env.kind != KIND_PAD_RPC_REQUEST_V1:
                    logger.debug(
                        "Ignoring RPC message kind=%s expected=%s",
                        env.kind,
                        KIND_PAD_RPC_REQUEST_V1,
                    )
                    continue
                await self._handle_request(env)

    async def _handle_request(self, env: BaseEnvelope) -> None:
        payload = env.payload if isinstance(env.payload, dict) else {}
        try:
            req = PadRpcRequestV1.model_validate(payload)
        except Exception as exc:
            logger.warning(f"Invalid RPC request: {exc}")
            self.stats.increment_rpc_errors()
            return
        started = time.perf_counter()
        logger.info(
            "RPC request received method=%s request_id=%s corr=%s reply=%s",
            req.method,
            req.request_id,
            env.correlation_id,
            req.reply_channel,
        )

        handler = self._handler_for(req.method)
        if handler is None:
            resp = PadRpcResponseV1(request_id=req.request_id, ok=False, error="unsupported_method")
        else:
            try:
                result = await handler(req.args or {})
                resp = PadRpcResponseV1(request_id=req.request_id, ok=True, result=result)
            except Exception as exc:
                logger.exception("RPC handler failed")
                resp = PadRpcResponseV1(request_id=req.request_id, ok=False, error=str(exc))
                self.stats.increment_rpc_errors()

        self.stats.increment_rpc_requests()
        if req.reply_channel:
            out = BaseEnvelope(
                kind=KIND_PAD_RPC_RESPONSE_V1,
                source=self._source(),
                correlation_id=env.correlation_id,
                causality_chain=env.causality_chain,
                payload=resp.model_dump(mode="json"),
            )
            try:
                await self.bus.publish(req.reply_channel, out)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.warning(
                    "RPC response publish failed method=%s request_id=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                    req.method,
                    req.request_id,
                    env.correlation_id,
                    req.reply_channel,
                    elapsed,
                    exc,
                )
                self.stats.increment_rpc_errors()
                return
            elapsed = time.perf_counter() - started
            logger.info(
                "RPC response sent method=%s request_id=%s corr=%s reply=%s elapsed=%.2fs ok=%s",
                req.method,
                req.request_id,
                env.correlation_id,
                req.reply_channel,
                elapsed,
                resp.ok,
            )
        else:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC response dropped (missing reply_channel) method=%s request_id=%s corr=%s elapsed=%.2fs",
                req.method,
                req.request_id,
                env.correlation_id,
                elapsed,
            )

    def _handler_for(self, method: str) -> Optional[Callable[[dict], asyncio.Future]]:
        mapping: Dict[str, Callable[[dict], asyncio.Future]] = {
            "get_latest_frame": self._get_latest_frame,
            "get_frames": self._get_frames,
            "get_salient_events": self._get_salient_events,
            "get_latest_tensor": self._get_latest_tensor,
        }
        return mapping.get(method)

    async def _get_latest_frame(self, args: dict) -> dict:
        frame = await self.store.get_latest_frame()
        if frame is None:
            return {"status": "missing"}
        return {"frame": frame.model_dump(mode="json")}

    async def _get_frames(self, args: dict) -> dict:
        limit = int(args.get("limit") or 10)
        frames = await self.store.get_frames(limit=limit)
        return {"frames": [f.model_dump(mode="json") for f in frames]}

    async def _get_salient_events(self, args: dict) -> dict:
        limit = int(args.get("limit") or 20)
        events = await self.store.get_salient_events(limit=limit)
        return {"events": [e.model_dump(mode="json") for e in events]}

    async def _get_latest_tensor(self, args: dict) -> dict:
        tensor = await self.store.get_latest_tensor()
        if tensor is None:
            return {"status": "missing"}
        return {"tensor": tensor.model_dump(mode="json")}
