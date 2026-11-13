from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.resilience import run_subscription
from orion.schemas.field_goal import FieldGoalProvenanceV1

from .settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.runtime.goal_context_listener")

# Repointed 2026-07-30 alongside orion/bus/channels.yaml's schema_id/message_kind
# change (memory.goals.proposed.v1 -> memory.field_goals.proposed.v1) -- see
# docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-
# observability-design.md. GoalProposalV1 is no longer this channel's schema.
_GOAL_PROPOSAL_KIND = "memory.field_goals.proposed.v1"


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
        goal = FieldGoalProvenanceV1.model_validate(env.payload or {})
    except ValueError as exc:
        logger.error("goal_context invalid payload corr=%s err=%s", corr, exc)
        return

    # Imported locally (not at module top-level) so tests can monkeypatch
    # orion.substrate.attention.goal_context.set_active_goal per-call without a
    # stale module-level binding.
    from orion.substrate.attention.goal_context import set_active_goal

    set_active_goal(goal)
    logger.info(
        "goal_context updated corr=%s artifact_id=%s priority=%.3f status=%s",
        corr,
        goal.artifact_id,
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

    await run_subscription(
        bus,
        s.channel_goal_proposal,
        lambda msg: _handle_bus_message(bus, msg, settings=s),
        stop_event=stop_event,
        name="goal_context listener",
    )


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
