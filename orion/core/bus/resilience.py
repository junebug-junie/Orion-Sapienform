from __future__ import annotations

import asyncio
import inspect
import logging as _logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Sequence

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback when loguru is unavailable
    import logging
    logger = logging.getLogger("orion.bus")

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope

_sub_logger = _logging.getLogger("orion.bus.resilience")

MessageHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]


@dataclass
class SubscriptionState:
    """Live state of one run_subscription listener, for a service's /health."""

    name: str
    channel: str
    subscribed: bool = False
    running: bool = True
    messages_total: int = 0
    failures_total: int = 0
    last_subscribed_at: float | None = None
    last_message_at: float | None = None
    last_failure_at: float | None = None
    last_failure: str | None = None


# One entry per listener name in this process. A listener that is not subscribed
# (backing off after a failure, or stopped) reads as down.
_SUBSCRIPTION_STATES: dict[str, SubscriptionState] = {}


def subscription_health(now: float | None = None) -> dict[str, Any]:
    """Every run_subscription listener in this process: ``ok`` is False when any is down.

    Times are wall-clock epoch seconds; ``*_age_sec`` is how long ago, from ``now``.
    A quiet channel is not down: ``last_message_age_sec`` only says when traffic last came.
    A stopped listener stays listed as down, so stopping one deliberately at runtime
    leaves health degraded. A handler that hangs keeps ``subscribed`` True: this
    reports whether the subscription is open, not whether messages are processed.
    """
    now = time.time() if now is None else now
    listeners = []
    # list() first: a health monitor may read this from a worker thread while
    # the loop registers a listener.
    for state in sorted(list(_SUBSCRIPTION_STATES.values()), key=lambda st: st.name):
        row = asdict(state)
        for key in ("last_subscribed_at", "last_message_at", "last_failure_at"):
            at = row[key]
            row[key.replace("_at", "_age_sec")] = None if at is None else round(now - at, 1)
        listeners.append(row)
    down = [row["name"] for row in listeners if not row["subscribed"]]
    return {"ok": not down, "down": down, "listeners": listeners}


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
    channel: str | Sequence[str],
    handle: MessageHandler,
    *,
    stop_event: asyncio.Event | None,
    name: str,
    patterns: bool = False,
    max_backoff_sec: float = 60.0,
    stuck_reads: int = 5,
) -> None:
    """Pass every message on ``channel`` to ``handle`` until ``stop_event`` is set.

    ``channel`` is one channel or several; with ``patterns=True`` they are glob
    patterns (PSUBSCRIBE) and messages arrive as ``pmessage``.

    ``handle`` is awaited before the next message is read. A handler that must not
    hold up the stream should start its own task and return None -- returning the
    task would await it here.

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
    channels = (channel,) if isinstance(channel, str) else tuple(channel)
    label = ",".join(channels)
    backoff = min(1.0, max_backoff_sec)
    state = SubscriptionState(name=name, channel=label)
    previous = _SUBSCRIPTION_STATES.get(name)
    if previous is not None and previous.running:
        _sub_logger.warning(
            "%s registered twice while the first is still running; health now reports only the newer one",
            name,
        )
    _SUBSCRIPTION_STATES[name] = state
    try:
        while stop_event is None or not stop_event.is_set():
            _sub_logger.info("%s subscribing channel=%s", name, label)
            try:
                subscription = (
                    bus.subscribe(*channels, patterns=True) if patterns else bus.subscribe(*channels)
                )
                async with subscription as pubsub:
                    state.subscribed = True
                    state.last_subscribed_at = time.time()
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
                        state.messages_total += 1
                        state.last_message_at = time.time()
                        try:
                            result = handle(msg)
                            if inspect.isawaitable(result):
                                await result
                        except Exception:
                            _sub_logger.exception("%s handle_failed channel=%s", name, label)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.subscribed = False
                state.failures_total += 1
                state.last_failure_at = time.time()
                state.last_failure = f"{type(exc).__name__}: {exc}"[:200]
                _sub_logger.exception(
                    "%s subscription_failed channel=%s retry_in=%.0fs", name, label, backoff
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
        state.subscribed = False
        state.running = False
        _sub_logger.info("%s stopped channel=%s", name, label)
