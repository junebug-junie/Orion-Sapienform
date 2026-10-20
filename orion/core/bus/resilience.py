from __future__ import annotations

import asyncio
import inspect
import logging as _logging
from typing import Any, Awaitable, Callable

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback when loguru is unavailable
    import logging
    logger = logging.getLogger("orion.bus")

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope

_sub_logger = _logging.getLogger("orion.bus.resilience")

MessageHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]


async def publish_with_reconnect(
    bus: OrionBusAsync,
    channel: str,
    msg: BaseEnvelope | dict[str, Any],
    *,
    log_label: str = "bus_publish",
) -> None:
    """Publish once; on transport failure reconnect the command client and retry."""
    try:
        await bus.publish(channel, msg)
    except Exception as exc:
        logger.warning("{} failed channel={} err={}; reconnecting", log_label, channel, exc)
        await bus.reconnect()
        await bus.publish(channel, msg)


async def run_subscription(
    bus: OrionBusAsync,
    channel: str,
    handle: MessageHandler,
    *,
    stop_event: asyncio.Event | None,
    name: str,
    max_backoff_sec: float = 60.0,
    stuck_reads: int = 5,
) -> None:
    """Pass every message on ``channel`` to ``handle`` until ``stop_event`` is set.

    A subscription that fails or drops is opened again, waiting 1 s then doubling
    up to ``max_backoff_sec``. Without this, one Redis error ends a listener for
    good while its process stays up and looks healthy (2026-10-20: a startup that
    blocked the event loop for ~3 min timed out every subscription in
    orion-substrate-runtime, and chat lost finalize appraisal until a code fix).
    A handler error is logged and the next message is read. ``handle`` may be sync
    or async.

    The backoff resets only once a session delivers a message, so a Redis that
    accepts SUBSCRIBE and drops at once still backs off. ``get_message`` returns
    None on a quiet channel within its own 1 s timeout; ``stuck_reads`` reads in a
    row that overrun the outer 1.2 s guard mean a dead connection, so the session
    is reopened instead of polled forever.
    """
    backoff = min(1.0, max_backoff_sec)
    try:
        while stop_event is None or not stop_event.is_set():
            _sub_logger.info("%s subscribing channel=%s", name, channel)
            try:
                async with bus.subscribe(channel) as pubsub:
                    overruns = 0
                    while stop_event is None or not stop_event.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                                timeout=1.2,
                            )
                        except asyncio.TimeoutError:
                            overruns += 1
                            if overruns >= stuck_reads:
                                raise ConnectionError(
                                    f"{overruns} reads in a row overran the read timeout"
                                ) from None
                            continue
                        overruns = 0
                        if not msg or msg.get("type") not in ("message", "pmessage"):
                            continue
                        backoff = min(1.0, max_backoff_sec)
                        try:
                            result = handle(msg)
                            if inspect.isawaitable(result):
                                await result
                        except Exception:
                            _sub_logger.exception("%s handle_failed channel=%s", name, channel)
            except asyncio.CancelledError:
                raise
            except Exception:
                _sub_logger.exception(
                    "%s subscription_failed channel=%s retry_in=%.0fs", name, channel, backoff
                )
                if stop_event is None:
                    await asyncio.sleep(backoff)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                    except asyncio.TimeoutError:
                        pass
                backoff = min(backoff * 2.0, max_backoff_sec)
    finally:
        _sub_logger.info("%s stopped channel=%s", name, channel)
