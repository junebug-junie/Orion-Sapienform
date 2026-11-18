"""Settle-time scoring of world-action episodes (attend-to-act loop), on a clock.

Moved here from orion-feedback-runtime when the feedback rung retired (2026-11-04,
docs/superpowers/specs/2026-11-04-retire-feedback-and-consolidation-rungs.md). The pass itself is
``orion.autonomy.world_settlement_store.score_due_world_episodes``. Every
SQL_WRITER_WORLD_SETTLEMENT_INTERVAL_SEC it scores due episodes and logs ``[WORLD_SETTLEMENT] scored=``
when any were scored, plus ``[WORLD_SETTLEMENT] alive`` once an hour. 0 disables it. It runs once at start, then sleeps: a sleep-first loop never runs
while the service is crash-looping.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.settings import Settings

logger = logging.getLogger("sql-writer.world_settlement_loop")

ALIVE_EVERY_SEC = 3600.0


def run_one_pass() -> int:
    from app.db import engine
    from orion.autonomy.world_settlement_store import WorldSettlementStore, score_due_world_episodes

    return score_due_world_episodes(WorldSettlementStore(engine))


async def world_settlement_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "sql_writer_world_settlement_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.info("world_settlement_loop DISABLED (SQL_WRITER_WORLD_SETTLEMENT_INTERVAL_SEC=%s)", interval)
        return
    logger.info("world_settlement_loop starting interval_sec=%.0f", interval)
    passes = scored_total = failed = 0
    last_alive = time.monotonic()
    while True:
        try:
            scored = await asyncio.to_thread(run_one_pass)
            passes += 1
            scored_total += scored
            if scored:
                logger.info("[WORLD_SETTLEMENT] scored=%d", scored)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed += 1
            logger.warning("world_settlement_cycle_failed error=%s", exc)
        # Most passes find nothing (a shed is rare), so a quiet log cannot tell idle from dead: one
        # line an hour says the loop is alive and what it did.
        if time.monotonic() - last_alive >= ALIVE_EVERY_SEC:
            logger.info("[WORLD_SETTLEMENT] alive passes=%d scored=%d failed=%d", passes, scored_total, failed)
            passes = scored_total = failed = 0
            last_alive = time.monotonic()
        await asyncio.sleep(interval)
