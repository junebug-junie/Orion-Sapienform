from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Callable
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.harness_finalize import HarnessPostTurnClosureV1

from .settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.runtime.post_turn_closure")

_CLOSURE_KIND = "harness.post_turn.closure.v1"
ClosureHandler = Callable[[HarnessPostTurnClosureV1], None]


async def handle_post_turn_closure_message(
    closure: HarnessPostTurnClosureV1,
    *,
    on_closure: ClosureHandler | None = None,
) -> None:
    """Consume a post-turn closure molecule (log + optional substrate side-effect)."""
    logger.info(
        "post_turn_closure received corr=%s surprise_unresolved=%s outcome_id=%s grammar_events=%d",
        closure.correlation_id,
        closure.surprise_unresolved,
        closure.outcome_molecule_id,
        len(closure.grammar_event_ids),
    )
    if on_closure is not None:
        on_closure(closure)


async def _handle_bus_message(
    bus: OrionBusAsync,
    raw_msg: dict[str, Any],
    *,
    settings: Settings,
    on_closure: ClosureHandler | None = None,
) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("post_turn_closure decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    corr = str(env.correlation_id or uuid4())
    kind = env.kind or ""
    if kind not in (_CLOSURE_KIND, "legacy.message"):
        logger.warning("post_turn_closure unsupported kind=%s corr=%s", kind, corr)
        return

    try:
        closure = HarnessPostTurnClosureV1.model_validate(env.payload or {})
    except ValueError as exc:
        logger.error("post_turn_closure invalid payload corr=%s err=%s", corr, exc)
        return

    await handle_post_turn_closure_message(closure, on_closure=on_closure)


async def run_post_turn_closure_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event | None = None,
    *,
    settings: Settings | None = None,
    on_closure: ClosureHandler | None = None,
) -> None:
    """Subscribe to post_turn_closure channel and apply substrate closure handling."""
    s = settings or get_settings()
    if not s.orion_bus_enabled:
        logger.info("Bus disabled; post_turn_closure listener not started")
        return
    if not s.enable_post_turn_closure_listener:
        logger.info("post_turn_closure listener disabled by config")
        return

    channel = s.channel_post_turn_closure
    logger.info("post_turn_closure listener subscribing channel=%s", channel)

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
                    await _handle_bus_message(bus, msg, settings=s, on_closure=on_closure)
                except Exception:
                    logger.exception("post_turn_closure unhandled bus worker error")
    except asyncio.CancelledError:
        raise
    finally:
        logger.info("post_turn_closure listener stopped channel=%s", channel)


async def start_post_turn_closure_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event,
    *,
    settings: Settings | None = None,
    on_closure: ClosureHandler | None = None,
) -> asyncio.Task[None]:
    task = asyncio.create_task(
        run_post_turn_closure_listener(
            bus,
            stop_event,
            settings=settings,
            on_closure=on_closure,
        ),
        name="substrate-post-turn-closure-listener",
    )
    return task


async def stop_post_turn_closure_listener(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
