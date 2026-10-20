"""Generic bus listener that keeps a service's local ``goal_state.py`` cache current.

Reused by every non-``orion-substrate-runtime`` consumer of
``orion:memory:goals:proposed`` that needs real goal-gated capability checks
(``orion-spark-concept-induction``, ``orion-world-pulse`` as of 2026-07-30) --
mirrors ``services/orion-substrate-runtime/app/goal_context_listener.py``'s
subscribe/decode/validate shape exactly, so the two listeners stay obviously the
same pattern rather than silently drifting. Kept here (in the shared ``orion``
package) instead of duplicated per-service per CLAUDE.md's cross-service-seam
rule -- this is real, reused logic, not test scaffolding.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import uuid4

from orion.autonomy.goal_state import set_active_goal
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.resilience import run_subscription
from orion.schemas.field_goal import FieldGoalProvenanceV1

logger = logging.getLogger("orion.autonomy.goal_state_listener")

_GOAL_PROPOSAL_KIND = "memory.field_goals.proposed.v1"
DEFAULT_GOAL_CHANNEL = "orion:memory:goals:proposed"


async def _handle_bus_message(bus: OrionBusAsync, raw_msg: dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("goal_state decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    corr = str(env.correlation_id or uuid4())
    kind = env.kind or ""
    if kind not in (_GOAL_PROPOSAL_KIND, "legacy.message"):
        logger.warning("goal_state unsupported kind=%s corr=%s", kind, corr)
        return

    try:
        goal = FieldGoalProvenanceV1.model_validate(env.payload or {})
    except ValueError as exc:
        logger.error("goal_state invalid payload corr=%s err=%s", corr, exc)
        return

    set_active_goal(goal)
    logger.info(
        "goal_state updated corr=%s artifact_id=%s priority=%.3f status=%s",
        corr,
        goal.artifact_id,
        goal.priority,
        goal.proposal_status,
    )


async def run_goal_state_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event | None = None,
    *,
    channel: str = DEFAULT_GOAL_CHANNEL,
) -> None:
    """Subscribe to the goal-proposal channel and keep this process's goal_state
    cache current. Always safe to run: it only updates an in-memory store that
    nothing reads unless the caller's capability-gating path actually consults
    ``get_active_goal()``, so it is harmless when idle. A dropped subscription is
    opened again with backoff (``run_subscription``) until ``stop_event`` is set.
    """
    await run_subscription(
        bus,
        channel,
        lambda msg: _handle_bus_message(bus, msg),
        stop_event=stop_event,
        name=f"goal-state-listener:{channel}",
    )


async def start_goal_state_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event,
    *,
    channel: str = DEFAULT_GOAL_CHANNEL,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        run_goal_state_listener(bus, stop_event, channel=channel),
        name="autonomy-goal-state-listener",
    )


async def stop_goal_state_listener(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
