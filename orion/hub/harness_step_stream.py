from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from orion.schemas.harness_finalize import HarnessRunStepV1

logger = logging.getLogger("orion.hub.harness_step_stream")

FrameSender = Callable[[dict[str, Any]], Awaitable[None]]


async def relay_harness_run_steps(
    bus: Any,
    *,
    correlation_id: str,
    channel: str,
    send_frame: FrameSender,
    stop_event: asyncio.Event,
) -> None:
    """Forward harness governor FCC steps to Hub WS as claude_step frames."""
    if bus is None:
        return
    try:
        async with bus.subscribe(channel) as pubsub:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5),
                        timeout=0.6,
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                payload = decoded.envelope.payload
                if not isinstance(payload, dict):
                    continue
                try:
                    step_event = HarnessRunStepV1.model_validate(payload)
                except Exception:
                    continue
                if step_event.correlation_id != correlation_id:
                    continue
                await send_frame(
                    {
                        "kind": "claude_step",
                        "mode": "orion",
                        "correlation_id": correlation_id,
                        "step": step_event.step,
                        "step_index": step_event.step_index,
                    }
                )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("harness step relay failed corr=%s", correlation_id, exc_info=True)
