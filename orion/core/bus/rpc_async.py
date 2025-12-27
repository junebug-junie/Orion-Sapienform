"""orion.core.bus.rpc_async

Async RPC helper for PubSub.

Hub (HTTP/WebSocket) and any other "client" style service should *not* be
implementing subscribe loops, thread bridges, or polling boilerplate.

This module provides a single ergonomic primitive:

  request_and_wait(bus, intake_channel, reply_channel, payload)

It:
  - subscribes first (avoids race conditions)
  - publishes JSON
  - waits for exactly one reply
  - decodes JSON automatically
  - closes subscriptions safely
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .service_async import OrionBusAsync

logger = logging.getLogger("orionbus.rpc")


def normalize_channel(channel: str) -> str:
    """Normalize known legacy aliases.

    Some older code used `orion:cortex:*` while services listen on
    `orion-cortex:*`. Normalize here so callers don't care.
    """

    if channel.startswith("orion:cortex:request"):
        return channel.replace("orion:cortex:request", "orion-cortex:request", 1)
    if channel.startswith("orion:cortex:result"):
        return channel.replace("orion:cortex:result", "orion-cortex:result", 1)
    return channel


def _decode_pubsub_data(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")

    # If it's a JSON string, parse it.
    if isinstance(raw, str):
        s = raw.strip()
        if s and (s[0] in "{[\""):
            try:
                return json.loads(s)
            except Exception:
                return raw
        return raw

    return raw


async def request_and_wait(
    bus: OrionBusAsync,
    *,
    intake_channel: str,
    reply_channel: str,
    payload: Dict[str, Any],
    timeout_sec: float = 60.0,
) -> Any:
    """RPC over PubSub.

    Subscribe first, then publish, then wait for the first reply message.

    Returns the decoded message "data" (often a dict).
    """

    if not bus or not getattr(bus, "enabled", False):
        raise RuntimeError("Orion bus is disabled")
    if getattr(bus, "client", None) is None:
        raise RuntimeError("Orion bus is not connected")

    intake = normalize_channel(intake_channel)
    reply = normalize_channel(reply_channel)

    ps = bus.pubsub()
    await ps.subscribe(reply)

    t0 = time.monotonic()
    try:
        # Publish after subscription is active.
        await bus.publish(intake, payload)

        # Wait for a single reply.
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout_sec:
                raise asyncio.TimeoutError(f"RPC timed out waiting for {reply}")

            msg = await ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not msg:
                await asyncio.sleep(0.01)
                continue

            # msg shape: {type, channel, pattern?, data}
            return _decode_pubsub_data(msg.get("data"))

    finally:
        try:
            await ps.unsubscribe(reply)
        except Exception:
            pass
        try:
            await ps.close()
        except Exception:
            pass


async def collect_replies(
    pubsub,
    *,
    expected_count: int,
    timeout_sec: float,
    match: Optional[Callable[[Any], bool]] = None,
) -> List[Any]:
    """Collect up to N PubSub messages.

    Use this when you already subscribed to the reply channel and need to
    gather multiple replies (fan-in) with a timeout.

    Args:
        pubsub: A redis-py asyncio PubSub instance.
        expected_count: Number of replies to wait for.
        timeout_sec: Overall timeout.
        match: Optional predicate that receives decoded message data and
            returns True if it should be included.
    """

    if expected_count <= 0:
        return []

    t0 = time.monotonic()
    out: List[Any] = []

    while len(out) < expected_count:
        elapsed = time.monotonic() - t0
        if elapsed > timeout_sec:
            break

        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if not msg:
            await asyncio.sleep(0.01)
            continue

        data = _decode_pubsub_data(msg.get("data"))
        if match is not None:
            try:
                if not match(data):
                    continue
            except Exception:
                # If the match fn fails, ignore this message rather than
                # kill the collector loop.
                continue

        out.append(data)

    return out
