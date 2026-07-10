from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.schemas.drives import GoalProposalV1

from .settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.runtime.goal_context_listener")

_GOAL_PROPOSAL_KIND = "memory.goals.proposed.v1"


async def _handle_bus_message(
    bus: OrionBusAsync,
    raw_msg: dict[str, Any],
    *,
    settings: Settings,
) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("goal_context decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    corr = str(env.correlation_id or uuid4())
    kind = env.kind or ""
    if kind not in (_GOAL_PROPOSAL_KIND, "legacy.message"):
        logger.warning("goal_context unsupported kind=%s corr=%s", kind, corr)
        return

    try:
        goal = GoalProposalV1.model_validate(env.payload or {})
    except ValueError as exc:
        logger.error("goal_context invalid payload corr=%s err=%s", corr, exc)
        return

    # Imported locally (not at module top-level) so tests can monkeypatch
    # orion.substrate.attention.goal_context.set_active_goal per-call without a
    # stale module-level binding.
    from orion.substrate.attention.goal_context import set_active_goal

    set_active_goal(goal)
    logger.info(
        "goal_context updated corr=%s artifact_id=%s drive_origin=%s priority=%.3f status=%s",
        corr,
        goal.artifact_id,
        goal.drive_origin,
        goal.priority,
        goal.proposal_status,
    )


async def run_goal_context_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event | None = None,
    *,
    settings: Settings | None = None,
) -> None:
    """Subscribe to the goal-proposal channel and keep the attention goal-context store current.

    Always runs whenever the bus is enabled (no separate feature flag): it only
    updates an in-memory store that nothing reads unless
    ``ORION_ATTENTION_TOPDOWN_ENABLED`` is separately true, so it is harmless
    when idle.
    """
    s = settings or get_settings()
    if not s.orion_bus_enabled:
        logger.info("Bus disabled; goal_context listener not started")
        return

    channel = s.channel_goal_proposal
    logger.info("goal_context listener subscribing channel=%s", channel)

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
                    await _handle_bus_message(bus, msg, settings=s)
                except Exception:
                    logger.exception("goal_context unhandled bus worker error")
    except asyncio.CancelledError:
        raise
    finally:
        logger.info("goal_context listener stopped channel=%s", channel)


async def start_goal_context_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event,
    *,
    settings: Settings | None = None,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        run_goal_context_listener(bus, stop_event, settings=settings),
        name="substrate-goal-context-listener",
    )


async def stop_goal_context_listener(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
