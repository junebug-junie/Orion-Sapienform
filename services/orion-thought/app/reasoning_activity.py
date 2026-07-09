"""Reasoning-activity projection — rolling-window aggregate of ReasoningCallV1.

orion-thought consumes per-call `ReasoningCallV1` telemetry (metadata only — no
trace text) off the bus, holds a *capped* in-memory window, and materializes a
`ReasoningActivityV1` snapshot for φ (spark-introspector) to read over HTTP.

Thin-service discipline: this module MUST NOT import `orion.substrate.*` (it drags
the heavy graph engine + `requests` this container does not ship). Only the bus
core and the telemetry schema are allowed.

Robustness discipline: `record`, `snapshot`, and the worker loop NEVER raise. A
bad message is logged and skipped; an empty window yields a zeroed projection.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, deque
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Deque

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.telemetry.reasoning import ReasoningActivityV1, ReasoningCallV1

from .settings import settings

logger = logging.getLogger("orion-thought.reasoning")

# Cap how many distinct modes we surface in `by_mode` — the projection is a
# health readout, not an unbounded histogram.
_BY_MODE_MAX_ENTRIES = 16

# Poll get_message with a 1s timeout; after this many idle polls verify Redis
# still lists us as a subscriber (silent pubsub disconnect leaves health green
# but drops messages). Mirrors the thought worker.
_PUBSUB_IDLE_POLLS_BEFORE_HEALTH = 30


class ReasoningActivityStore:
    """Capped rolling window of `ReasoningCallV1`, materialized on demand.

    The buffer is a `deque(maxlen=max_calls)` — recording past the cap evicts the
    oldest call, so memory is bounded regardless of producer rate.
    """

    def __init__(self, window_sec: float, max_calls: int) -> None:
        # Clamp: window_sec feeds ReasoningActivityV1(window_sec=...) which is
        # ge=0.0, including on the empty/error fallback paths — a negative config
        # value must not 500 the endpoint.
        self.window_sec = max(0.0, float(window_sec))
        self.max_calls = max(1, int(max_calls))
        self._calls: Deque[ReasoningCallV1] = deque(maxlen=self.max_calls)

    def record(self, call: ReasoningCallV1) -> None:
        """Append one call. Never raises."""
        try:
            self._calls.append(call)
        except Exception as exc:  # noqa: BLE001 — record must never break the loop
            logger.warning("reasoning_activity record failed err=%s", exc)

    def snapshot(self, now: datetime) -> ReasoningActivityV1:
        """Materialize the window ending at ``now``. Never raises."""
        try:
            return self._snapshot(now)
        except Exception as exc:  # noqa: BLE001 — snapshot must never break the endpoint
            logger.warning("reasoning_activity snapshot failed err=%s", exc)
            return ReasoningActivityV1(generated_at=now, window_sec=self.window_sec)

    def _snapshot(self, now: datetime) -> ReasoningActivityV1:
        cutoff = now - timedelta(seconds=self.window_sec)
        in_window = [c for c in list(self._calls) if _emitted_at(c) >= cutoff]

        call_count = len(in_window)
        if call_count == 0:
            return ReasoningActivityV1(generated_at=now, window_sec=self.window_sec)

        reasoning_call_count = sum(1 for c in in_window if c.reasoning_present)
        thinking_call_count = sum(1 for c in in_window if c.thinking_enabled)
        reasoning_present_rate = reasoning_call_count / call_count

        completion_values = [c.completion_tokens for c in in_window if c.completion_tokens is not None]
        completion_tokens_sum = sum(completion_values)
        completion_tokens_p50 = float(median(completion_values)) if completion_values else 0.0

        thinking_values = [c.thinking_tokens for c in in_window if c.thinking_tokens is not None]
        thinking_tokens_sum = sum(thinking_values) if thinking_values else None

        mode_counts = Counter(c.mode for c in in_window)
        by_mode = dict(mode_counts.most_common(_BY_MODE_MAX_ENTRIES))

        return ReasoningActivityV1(
            generated_at=now,
            window_sec=self.window_sec,
            call_count=call_count,
            reasoning_call_count=reasoning_call_count,
            thinking_call_count=thinking_call_count,
            reasoning_present_rate=reasoning_present_rate,
            completion_tokens_sum=completion_tokens_sum,
            completion_tokens_p50=completion_tokens_p50,
            thinking_tokens_sum=thinking_tokens_sum,
            by_mode=by_mode,
        )


def _emitted_at(call: ReasoningCallV1) -> datetime:
    """Return an aware UTC emitted_at so naive/aware comparisons never crash."""
    ts = call.emitted_at
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


# Module-level singleton built from settings — shared by the worker and the HTTP
# endpoint so reads see what the bus wrote.
store = ReasoningActivityStore(
    window_sec=settings.reasoning_activity_window_sec,
    max_calls=settings.reasoning_activity_max_calls,
)


def _decode_reasoning_call(bus: OrionBusAsync, raw_msg: dict[str, Any]) -> ReasoningCallV1 | None:
    """Decode a bus message into a `ReasoningCallV1`, or None if unusable.

    Defensive: any decode/validation failure returns None (logged) rather than
    raising, so one malformed message never kills the worker loop.
    """
    try:
        decoded = bus.codec.decode(raw_msg.get("data"))
        payload = decoded.envelope.payload if decoded.ok else None
        if not isinstance(payload, dict):
            logger.warning("reasoning_call decode: non-dict payload ok=%s", getattr(decoded, "ok", None))
            return None
        return ReasoningCallV1.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — never crash the loop on a bad message
        logger.warning("reasoning_call decode failed err=%s", exc)
        return None


async def _reasoning_channel_subscribers(bus: OrionBusAsync, channel: str) -> int:
    """Return subscriber count for channel, or -1 when the probe itself fails."""
    try:
        pairs = await bus.redis.pubsub_numsub(channel)
    except Exception as exc:  # noqa: BLE001 — probe must not take down the worker
        logger.warning("pubsub health probe failed channel=%s err=%s", channel, exc)
        return -1
    for name, count in pairs:
        key = name.decode() if isinstance(name, bytes) else str(name)
        if key == channel:
            return int(count)
    return 0


async def run_reasoning_worker(stop_event: asyncio.Event | None = None) -> None:
    """Subscribe the reasoning-call channel and feed the activity store.

    Mirrors ``run_bus_worker``: reconnect-with-backoff, pubsub-health probe on
    idle, and a fully defensive per-message path. Consuming is harmless — with no
    producer the store simply stays empty and the projection reads as zeroed.
    """
    if not settings.orion_bus_enabled:
        logger.info("Bus disabled; reasoning worker not started")
        return

    channel = settings.channel_reasoning_call
    backoff_sec = 1.0

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        bus = OrionBusAsync(url=settings.orion_bus_url)
        reconnect = False
        idle_polls = 0
        try:
            await bus.connect()
            logger.info("reasoning worker subscribed channel=%s", channel)
            async with bus.subscribe(channel) as pubsub:
                backoff_sec = 1.0
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        idle_polls += 1
                        if idle_polls >= _PUBSUB_IDLE_POLLS_BEFORE_HEALTH:
                            idle_polls = 0
                            subs = await _reasoning_channel_subscribers(bus, channel)
                            if subs == 0:
                                logger.warning(
                                    "pubsub subscription missing channel=%s; reconnecting",
                                    channel,
                                )
                                reconnect = True
                                break
                        continue
                    except (ConnectionError, OSError) as exc:
                        logger.warning(
                            "reasoning pubsub read failed channel=%s err=%s; reconnecting",
                            channel,
                            exc,
                        )
                        reconnect = True
                        break

                    if not msg or msg.get("type") not in ("message", "pmessage"):
                        continue
                    idle_polls = 0
                    call = _decode_reasoning_call(bus, msg)
                    if call is not None:
                        store.record(call)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reasoning worker disconnect channel=%s", channel)
            reconnect = True
        finally:
            with suppress(Exception):
                await bus.close()

        if stop_event is not None and stop_event.is_set():
            return
        if not reconnect:
            return
        await asyncio.sleep(backoff_sec)
        backoff_sec = min(backoff_sec * 2, 30.0)
