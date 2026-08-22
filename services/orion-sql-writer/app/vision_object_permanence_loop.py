"""Periodic object-permanence sweep. Timer-driven by design -- see
vision_object_permanence.py's module docstring for why an event-triggered
version cannot exist: a departure is a non-event, and nothing fires when a
thing stops being there.

Mirrors grammar_retention_loop.py's shape: sleep first (never adds load at
boot, exactly when this service is replaying its bus backlog), one bounded
pass per interval, best-effort so a bad cycle never kills the loop.
"""

from __future__ import annotations

import asyncio
import logging

from app.settings import Settings
from app.vision_object_permanence import run_one_sweep_cycle

logger = logging.getLogger("sql-writer.vision_object_permanence_loop")


async def vision_object_permanence_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "vision_permanence_sweep_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.info(
            "vision_object_permanence_loop DISABLED "
            "(VISION_PERMANENCE_SWEEP_INTERVAL_SEC=%s) -- object permanence will not update.",
            interval,
        )
        return

    # Reuses settings.postgres_uri -- sql-writer's own primary DB connection --
    # rather than a second field aliased to the same POSTGRES_URI env var.
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if not postgres_uri:
        logger.info("vision_object_permanence_loop DISABLED -- no Postgres URI configured.")
        return

    lookback_ceiling = float(
        getattr(settings, "vision_permanence_lookback_ceiling_sec", 3600.0) or 3600.0
    )
    absence_fraction = float(getattr(settings, "vision_permanence_absence_fraction", 0.1) or 0.1)
    min_absence_sec = float(getattr(settings, "vision_permanence_min_absence_sec", 3600.0) or 3600.0)
    max_absence_sec = float(
        getattr(settings, "vision_permanence_max_absence_sec", 86400.0) or 86400.0
    )

    logger.info(
        "vision_object_permanence_loop starting interval_sec=%.0f lookback_ceiling_sec=%.0f "
        "absence_fraction=%.3f min_absence_sec=%.0f max_absence_sec=%.0f",
        interval, lookback_ceiling, absence_fraction, min_absence_sec, max_absence_sec,
    )

    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            # Re-raise, do not suppress: a suppressed cancellation during sleep
            # would fall through into the try block below and run one more
            # sweep cycle after being told to stop -- the exact bug
            # grammar_retention_loop's own handling avoids (it re-raises here
            # too, then signals its worker thread separately in the second
            # try block, which this loop does not need since to_thread's
            # blocking call below has no long-running work to signal).
            raise

        try:
            summary = await asyncio.to_thread(
                run_one_sweep_cycle,
                postgres_uri=postgres_uri,
                lookback_ceiling_sec=lookback_ceiling,
                absence_fraction=absence_fraction,
                min_absence_sec=min_absence_sec,
                max_absence_sec=max_absence_sec,
            )
            logger.info("[VISION_PERMANENCE] sweep_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Best-effort: one bad cycle (a transient DB blip, say) must not
            # end the loop. The next interval tries again.
            logger.warning("vision_object_permanence_cycle_failed error=%s", exc)
