from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.harness.fcc_motor import cancel_fcc_turn
from orion.schemas.harness_finalize import HarnessRunCancelV1

from .settings import settings

logger = logging.getLogger("orion-harness-governor.cancel")


def apply_harness_run_cancel(payload: dict[str, Any] | HarnessRunCancelV1) -> bool:
    """Validate cancel payload and kill the matching FCC subprocess if live."""
    if isinstance(payload, HarnessRunCancelV1):
        cancel = payload
    else:
        cancel = HarnessRunCancelV1.model_validate(payload)
    killed = cancel_fcc_turn(cancel.correlation_id)
    logger.info(
        "harness_run_cancel corr=%s reason=%s killed=%s",
        cancel.correlation_id,
        cancel.reason,
        killed,
    )
    return killed


async def handle_cancel_bus_message(bus: OrionBusAsync, raw_msg: dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("harness cancel decode failed: %s", decoded.error)
        return
    env = decoded.envelope
    kind = env.kind or ""
    if kind not in ("harness.run.cancel.v1", "legacy.message"):
        logger.warning("harness cancel unsupported kind=%s", kind)
        return
    payload = env.payload or {}
    try:
        apply_harness_run_cancel(payload if isinstance(payload, dict) else {})
    except Exception:
        logger.exception("harness cancel apply failed corr=%s", env.correlation_id)


async def run_cancel_worker(stop_event: Any | None = None) -> None:
    """Subscribe to harness run cancel events and kill matching FCC motors."""
    if not settings.orion_bus_enabled or not settings.orion_harness_governor_enabled:
        logger.info("harness cancel worker idle (bus/governor disabled)")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    channel = settings.channel_harness_run_cancel
    await bus.connect()
    logger.info("subscribed cancel channel=%s", channel)

    try:
        async with bus.subscribe(channel) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=1.2,
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await handle_cancel_bus_message(bus, msg)
                except Exception:
                    logger.exception("unhandled harness cancel worker error")
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()
