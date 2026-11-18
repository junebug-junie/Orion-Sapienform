"""Clocked wrapper for app/vision_rollup.py (walkway rollups).

Work first, then sleep: a sleep-first loop never runs at all in a process that
restarts faster than its interval. One bounded pass per interval in a worker
thread, best-effort. While the migration is not applied the loop logs one clear
ERROR and backs off (doubling, capped at an hour).

Trace: one ``[VISION_ROLLUP]`` line per tick with hours/days/weeks/months
written per stream, ``ready_until``, and whether the story gate is open.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.settings import Settings
from app.vision_rollup import MigrationMissing, RollupConfig, run_one_rollup_cycle

logger = logging.getLogger("sql-writer.vision_rollup_loop")

MAX_BACKOFF_SEC = 3600.0


def next_backoff(interval: float, current: Optional[float]) -> float:
    return min(MAX_BACKOFF_SEC, max(interval, (current or interval) * 2.0))


def rollup_config(settings: Settings) -> RollupConfig:
    return RollupConfig(
        streams=tuple(s.strip() for s in str(settings.vision_rollup_streams).split(",") if s.strip()),
        tz=str(settings.vision_local_tz),
        parked_min_hours=int(settings.vision_rollup_parked_min_hours),
        catchup_hours=int(settings.vision_rollup_catchup_hours),
        prop_individual_ids=tuple(
            s.strip() for s in str(settings.vision_motion_prop_individual_ids).split(",") if s.strip()),
    )


async def vision_rollup_loop(settings: Settings) -> None:
    enabled = bool(getattr(settings, "vision_rollup_enabled", False))
    interval = float(getattr(settings, "vision_rollup_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if not enabled or interval <= 0 or not postgres_uri:
        logger.info("vision_rollup_loop DISABLED (VISION_ROLLUP_ENABLED=%s interval=%s)", enabled, interval)
        return
    cfg = rollup_config(settings)
    logger.info("vision_rollup_loop starting interval_sec=%.0f cfg=%s", interval, cfg)
    backoff: Optional[float] = None
    while True:
        try:
            summary = await asyncio.to_thread(run_one_rollup_cycle, postgres_uri=postgres_uri, cfg=cfg)
            backoff = None
            logger.info("[VISION_ROLLUP] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except MigrationMissing as exc:
            backoff = next_backoff(interval, backoff)
            logger.error("[VISION_ROLLUP] migration not applied: %s -- retrying in %.0fs", exc, backoff)
        except Exception as exc:
            logger.warning("vision_rollup_cycle_failed error=%s", exc)
        await asyncio.sleep(backoff or interval)
