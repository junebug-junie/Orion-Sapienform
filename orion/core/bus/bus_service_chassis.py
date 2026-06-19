# orion/core/bus/bus_service_chassis.py
from __future__ import annotations

import asyncio
import signal
import traceback
from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Awaitable, Callable, Optional, List, Union
from datetime import datetime, timezone

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback when loguru is unavailable
    import logging
    logger = logging.getLogger("orion.bus")

from .async_service import OrionBusAsync
from .bus_schemas import BaseEnvelope, ErrorInfo, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1


Handler = Callable[[BaseEnvelope], Awaitable[BaseEnvelope | None]]


@dataclass(frozen=True)
class ChassisConfig:
    service_name: str
    service_version: str
    node_name: str
    instance_id: Optional[str] = None
    bus_url: str = "redis://100.92.216.81:6379/0"
    bus_enabled: bool = True

    # system behaviors
    heartbeat_interval_sec: float = 10.0
    connect_timeout_sec: float = 10.0
    shutdown_timeout_sec: float = 10.0

    # system channels (stable defaults)
    health_channel: str = "orion:system:health"
    error_channel: str = "orion:system:error"


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
        self.boot_id = str(uuid4())

        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._started = False

    def _source(self) -> ServiceRef:
        return ServiceRef(
            name=self.cfg.service_name,
            version=self.cfg.service_version,
            node=self.cfg.node_name,
            instance=self.cfg.instance_id,
        )

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._install_signal_handlers()

        timeout = float(self.cfg.connect_timeout_sec or 10.0)
        logger.info(f"Connecting bus url={self.cfg.bus_url}")
        await asyncio.wait_for(self.bus.connect(), timeout=timeout)

        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="orion-heartbeat"))
        self._tasks.append(asyncio.create_task(self._supervise_run(), name=f"{self.cfg.service_name}-run"))

        await self._stop.wait()
        await self._shutdown()

    async def start_background(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """Non-blocking start for use in other runtimes (e.g. FastAPI lifespan)."""
        if self._started:
            return
        self._started = True
        
        timeout = float(self.cfg.connect_timeout_sec or 10.0)
        logger.info(f"Connecting bus url={self.cfg.bus_url} (background)")
        await asyncio.wait_for(self.bus.connect(), timeout=timeout)

        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="orion-heartbeat"))
        self._tasks.append(asyncio.create_task(self._supervise_run(), name=f"{self.cfg.service_name}-run"))

    async def stop(self) -> None:
        self._stop.set()
        await self._shutdown()

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
                signal.signal(sig, lambda *_: _handler())

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                node = self.cfg.node_name or "unknown"
                now = datetime.now(timezone.utc)
                v1_payload = SystemHealthV1(
                    service=self.cfg.service_name,
                    node=node,
                    version=self.cfg.service_version,
                    instance=self.cfg.instance_id,
                    boot_id=self.boot_id,
                    status="ok",
                    last_seen_ts=now,
                    heartbeat_interval_sec=float(self.cfg.heartbeat_interval_sec or 10.0),
                    details={},
                )
                v1_env = BaseEnvelope(
                    kind="system.health.v1",
                    source=self._source(),
                    payload=v1_payload.model_dump(mode="json"),
                )
                await self.bus.publish(self.cfg.health_channel, v1_env)
            except Exception as e:
                logger.warning("Heartbeat publish failed: %s; reconnecting", e)
                try:
                    await self.bus.reconnect()
                    await self.bus.publish(self.cfg.health_channel, v1_env)
                except Exception as retry_exc:
                    logger.warning("Heartbeat publish retry failed: %s", retry_exc)
            await asyncio.sleep(float(self.cfg.heartbeat_interval_sec or 10.0))

    async def _publish_error(self, err: BaseException, *, when: str, env: BaseEnvelope | None = None) -> None:
        # [FIX] LOG THE ERROR TO STDOUT SO WE CAN SEE IT
        logger.error(f"System Error in {self.cfg.service_name} ({when}): {err}\n{traceback.format_exc()}")
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

    async def _supervise_run(self) -> None:
        """Restart subscriber loops if they exit without an explicit stop signal."""
        backoff_sec = 1.0
        while not self._stop.is_set():
            try:
                await self._run()
                if self._stop.is_set():
                    break
                logger.warning(
                    "subscriber run loop exited without stop service={} bus={}; restarting in {:.1f}s",
                    self.cfg.service_name,
                    self.cfg.bus_url,
                    backoff_sec,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "subscriber run loop crashed service={} err={}; restarting in {:.1f}s",
                    self.cfg.service_name,
                    exc,
                    backoff_sec,
                )
            if self._stop.is_set():
                break
            try:
                await self.bus.reconnect()
            except Exception:
                logger.exception("bus reconnect failed after subscriber crash service=%s", self.cfg.service_name)
            await asyncio.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2.0, 30.0)


class Rabbit(BaseChassis):
    """
    RPC / synchronous pattern.
    Listens on a single request channel and replies to reply_to.
    """

    def __init__(
        self,
        cfg: ChassisConfig,
        *,
        request_channel: str,
        handler: Handler,
        concurrent_handlers: bool = False,
    ):
        super().__init__(cfg)
        self.request_channel = request_channel
        self.handler = handler
        self.concurrent_handlers = bool(concurrent_handlers)
        self._inflight_tasks: set[asyncio.Task] = set()

    async def _handle_decoded_envelope(self, *, channel: str, env: BaseEnvelope) -> None:
        trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
        logger.info(
            f"Rabbit request received channel={channel} kind={env.kind} "
            f"schema_id={env.schema_id} trace_id={trace_id} source={env.source}"
        )
        out = await self.handler(env)
        if out is not None and env.reply_to:
            payload_obj = getattr(out, "payload", None)
            payload_keys = []
            reasoning_content = None
            trace_content = None
            if isinstance(payload_obj, dict):
                payload_keys = sorted(payload_obj.keys())
                reasoning_content = payload_obj.get("reasoning_content")
                rt = payload_obj.get("reasoning_trace")
                trace_content = rt.get("content") if isinstance(rt, dict) else None
            elif hasattr(payload_obj, "model_dump"):
                try:
                    dumped = payload_obj.model_dump(mode="json")
                except Exception:
                    dumped = payload_obj.model_dump()
                if isinstance(dumped, dict):
                    payload_keys = sorted(dumped.keys())
                    reasoning_content = dumped.get("reasoning_content")
                    rt = dumped.get("reasoning_trace")
                    trace_content = rt.get("content") if isinstance(rt, dict) else None
                else:
                    payload_keys = [type(payload_obj).__name__]
            else:
                payload_keys = [type(payload_obj).__name__ if payload_obj is not None else "NoneType"]
            preview_text = str(reasoning_content or trace_content or "")[:220]
            print(
                "===THINK_HOP=== hop=llm_gateway_bus_reply "
                f"corr={getattr(out, 'correlation_id', None) or env.correlation_id} "
                f"kind={getattr(out, 'kind', None)} "
                f"payload_keys={payload_keys} "
                f"reasoning_len={len(reasoning_content) if isinstance(reasoning_content, str) else 0} "
                f"trace_len={len(trace_content) if isinstance(trace_content, str) else 0} "
                f"preview={repr(preview_text)}",
                flush=True,
            )
            corr_id = getattr(out, "correlation_id", None) or env.correlation_id
            is_gateway = self.cfg.service_name == "llm-gateway"
            if is_gateway:
                result_status = "error" if getattr(out, "kind", None) == "system.error" else "ok"
                logger.info(
                    "gateway_llm_reply_publish_start event=gateway_llm_reply_publish_start "
                    "correlation_id=%s reply_to=%s result_status=%s payload_kind=%s",
                    corr_id,
                    env.reply_to,
                    result_status,
                    getattr(out, "kind", None),
                )
            try:
                await self.bus.publish(env.reply_to, out)
            except Exception as exc:
                if is_gateway:
                    catalog_hint = ""
                    if "catalog" in str(exc).lower() or "Channel not found" in str(exc):
                        catalog_hint = "catalog_enforcement_suspected"
                    logger.error(
                        "gateway_llm_reply_publish_failed event=gateway_llm_reply_publish_failed "
                        "correlation_id=%s reply_to=%s error_type=%s err=%s catalog_hint=%s",
                        corr_id,
                        env.reply_to,
                        type(exc).__name__,
                        exc,
                        catalog_hint or "none",
                    )
                raise
            if is_gateway:
                logger.info(
                    "gateway_llm_reply_publish_ok event=gateway_llm_reply_publish_ok "
                    "correlation_id=%s reply_to=%s",
                    corr_id,
                    env.reply_to,
                )

    async def _run(self) -> None:
        backoff_sec = 1.0
        while not self._stop.is_set():
            logger.info(f"Rabbit listening channel={self.request_channel} bus={self.cfg.bus_url}")
            try:
                if not self.bus.enabled:
                    break
                if self.bus.redis is None:
                    await self.bus.connect()
                async with self.bus.subscribe(self.request_channel) as pubsub:
                    backoff_sec = 1.0
                    try:
                        async for msg in self.bus.iter_messages(pubsub):
                            if self._stop.is_set():
                                break
                            if not isinstance(msg, dict):
                                continue
                            data = msg.get("data")
                            if data is None:
                                continue

                            channel = msg.get("channel")
                            if hasattr(channel, "decode"):
                                channel = channel.decode("utf-8")

                            decoded = self.bus.codec.decode(data)
                            if not decoded.ok or decoded.envelope is None:
                                logger.warning(
                                    f"Rabbit decode failed channel={channel} error={decoded.error}"
                                )
                                await self._publish_error(
                                    RuntimeError(decoded.error or "decode_failed"),
                                    when="rabbit.decode",
                                    env=None,
                                )
                                continue

                            env = decoded.envelope
                            if self.concurrent_handlers:
                                async def _run_handler(envelope: BaseEnvelope, channel_name: str) -> None:
                                    try:
                                        await self._handle_decoded_envelope(channel=channel_name, env=envelope)
                                    except Exception as e:
                                        await self._publish_error(e, when="rabbit.handle", env=envelope)

                                task = asyncio.create_task(_run_handler(env, str(channel)))
                                self._inflight_tasks.add(task)
                                task.add_done_callback(lambda t: self._inflight_tasks.discard(t))
                                continue

                            try:
                                await self._handle_decoded_envelope(channel=str(channel), env=env)
                            except Exception as e:
                                await self._publish_error(e, when="rabbit.handle", env=env)
                    finally:
                        if self._inflight_tasks:
                            for task in list(self._inflight_tasks):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Rabbit subscriber loop failed channel={} err={}; reconnecting in {:.1f}s",
                    self.request_channel,
                    exc,
                    backoff_sec,
                )
                try:
                    await self.bus.reconnect()
                except Exception:
                    logger.exception(
                        "bus reconnect failed after Rabbit subscriber error channel=%s",
                        self.request_channel,
                    )
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2.0, 30.0)


class Hunter(BaseChassis):
    """
    Fire-and-forget consumer. Subscribes to patterns, filters, and acts.
    """
    # [FIX] Support list of patterns
    def __init__(
        self,
        cfg: ChassisConfig,
        *,
        handler: Callable[[BaseEnvelope], Awaitable[None]],
        patterns: Union[List[str], str, None] = None,
        pattern: Optional[str] = None,
        concurrent_handlers: bool = False,
    ):
        super().__init__(cfg)
        
        self.patterns: List[str] = []
        if patterns is not None:
            if isinstance(patterns, str):
                self.patterns.append(patterns)
            else:
                self.patterns.extend(patterns)
        
        if pattern is not None:
            self.patterns.append(pattern)

        seen = set()
        deduped: List[str] = []
        for value in self.patterns:
            value = (value or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        self.patterns = deduped

        if not self.patterns:
            raise ValueError("Hunter requires at least one pattern (via 'patterns' list or 'pattern' str)")
            
        self.handler = handler
        self.concurrent_handlers = bool(concurrent_handlers)
        self._inflight_tasks: set[asyncio.Task] = set()

    async def _run(self) -> None:
        uses_glob = any(any(ch in pattern for ch in "*?[") for pattern in self.patterns)
        backoff_sec = 1.0
        while not self._stop.is_set():
            logger.info(
                "Hunter subscribing patterns={} use_patterns={} bus={}",
                self.patterns,
                uses_glob,
                self.cfg.bus_url,
            )
            try:
                if not self.bus.enabled:
                    break
                if self.bus.redis is None:
                    await self.bus.connect()
                async with self.bus.subscribe(*self.patterns, patterns=uses_glob) as pubsub:
                    backoff_sec = 1.0
                    try:
                        async for msg in self.bus.iter_messages(pubsub):
                            if self._stop.is_set():
                                break
                            if not isinstance(msg, dict):
                                continue
                            data = msg.get("data")
                            if data is None:
                                continue

                            channel = msg.get("channel")
                            if hasattr(channel, "decode"):
                                channel = channel.decode("utf-8")
                            pattern = msg.get("pattern")
                            if hasattr(pattern, "decode"):
                                pattern = pattern.decode("utf-8")

                            decoded = self.bus.codec.decode(data)
                            if not decoded.ok or decoded.envelope is None:
                                logger.warning(
                                    f"Hunter decode failed channel={channel} pattern={pattern} error={decoded.error}"
                                )
                                await self._publish_error(
                                    RuntimeError(decoded.error or "decode_failed"),
                                    when="hunter.decode",
                                    env=None,
                                )
                                continue

                            env = decoded.envelope
                            trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
                            logger.info(
                                f"Hunter intake channel={channel} pattern={pattern} kind={env.kind} "
                                f"schema_id={env.schema_id} trace_id={trace_id} source={env.source}"
                            )
                            if self.concurrent_handlers:
                                async def _run_handler(envelope: BaseEnvelope) -> None:
                                    try:
                                        await self.handler(envelope)
                                    except Exception as e:
                                        await self._publish_error(e, when="hunter.handle", env=envelope)

                                task = asyncio.create_task(_run_handler(env))
                                self._inflight_tasks.add(task)
                                task.add_done_callback(lambda t: self._inflight_tasks.discard(t))
                                continue

                            try:
                                await self.handler(env)
                            except Exception as e:
                                await self._publish_error(e, when="hunter.handle", env=env)
                    finally:
                        if self._inflight_tasks:
                            for task in list(self._inflight_tasks):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Hunter subscriber loop failed patterns={} err={}; reconnecting in {:.1f}s",
                    self.patterns,
                    exc,
                    backoff_sec,
                )
                try:
                    await self.bus.reconnect()
                except Exception:
                    logger.exception(
                        "bus reconnect failed after Hunter subscriber error patterns=%s",
                        self.patterns,
                    )
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2.0, 30.0)


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
