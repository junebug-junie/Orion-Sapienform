"""orion-memory-consolidation's half of the memory confirmation loop.

Two entry points around ``orion.memory.episode.confirmation``:

* ``handle_loop_outcome``: the bus consumer of ``orion:attention:loop_outcome``. Applies a
  ``memory-confirm-*`` outcome as soon as the Hub publishes it. Other loop ids are ignored.
* ``run_confirmation_loop``: the ticker. Each pass expires overdue cards, catches up on any outcome
  the bus dropped (read from ``attention_loop_outcome``), then opens cards for waiting high-stakes
  memories up to the 5-card cap. A failure is logged and the next tick retries.

Both are gated on ``MEMORY_CONFIRMATION_LOOP_ENABLED``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.memory.episode.confirmation import (
    OutcomeToApply,
    apply_outcome,
    memory_id_from_loop,
    run_tick,
)
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

logger = logging.getLogger(__name__)

LOOP_OUTCOME_KIND = "attention.loop.outcome.v1"


def outcome_from_envelope(env: BaseEnvelope) -> OutcomeToApply | None:
    """The bus payload as an outcome to apply, or None when it is not a memory confirmation."""
    if env.kind != LOOP_OUTCOME_KIND:
        return None
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        event = AttentionLoopOutcomeV1.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 -- a malformed event must not kill the hunter
        logger.warning("memory_confirmation_outcome_invalid error=%s", exc)
        return None
    if memory_id_from_loop(event.loop_id) is None:
        return None
    return OutcomeToApply(
        outcome_id=event.outcome_id,
        loop_id=event.loop_id,
        verdict=event.verdict,
        note=event.note,
        features=dict(event.features_at_close or {}),
        actor=event.actor,
    )


async def handle_loop_outcome(env: BaseEnvelope, *, pool: Any, settings: Any) -> str | None:
    if not settings.MEMORY_CONFIRMATION_LOOP_ENABLED or pool is None:
        return None
    outcome = outcome_from_envelope(env)
    if outcome is None:
        return None
    async with pool.acquire() as conn:
        result = await apply_outcome(conn, outcome)
    logger.info("memory_confirmation_applied outcome_id=%s loop_id=%s result=%s via=bus",
                outcome.outcome_id, outcome.loop_id, result)
    return result


async def run_confirmation_loop(pool: Any, settings: Any) -> None:
    while True:
        if settings.MEMORY_CONFIRMATION_LOOP_ENABLED:
            try:
                summary = await run_tick(
                    pool,
                    daily_cap=int(settings.MEMORY_CONFIRMATION_DAILY_CAP),
                    tz_name=str(settings.MEMORY_CONFIRMATION_TZ),
                )
                if any(summary.values()):
                    logger.info("memory_confirmation_tick %s", summary)
            except Exception:  # noqa: BLE001 -- the next tick retries; the table is the truth
                logger.exception("memory_confirmation_tick_failed")
        await asyncio.sleep(float(settings.MEMORY_CONFIRMATION_TICK_SEC))
