"""Clocked wrapper for app/vision_office_sessions.py.

Same shape as vision_individuals_loop.py: sleep first, one bounded pass per
interval in a worker thread, best-effort. While the migration is not applied
the loop logs one clear ERROR and backs off (doubling, capped at an hour).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.settings import Settings
from app.vision_individuals_loop import individuals_config, next_backoff
from app.vision_office_sessions import MigrationMissing, OfficeSessionsConfig, run_one_office_cycle

logger = logging.getLogger("sql-writer.vision_office_sessions_loop")


def office_sessions_config(settings: Settings) -> OfficeSessionsConfig:
    return OfficeSessionsConfig(
        streams=tuple(s.strip() for s in str(settings.vision_office_sessions_streams or "").split(",") if s.strip()),
        settle_sec=float(settings.vision_office_sessions_settle_sec),
        lookback_hours=float(settings.vision_office_sessions_lookback_hours),
        ask_daily_cap=int(settings.vision_office_ask_daily_cap),
        local_tz=str(settings.vision_local_tz),
    )


async def vision_office_sessions_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "vision_office_sessions_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if interval <= 0 or not postgres_uri:
        logger.info("vision_office_sessions_loop DISABLED (VISION_OFFICE_SESSIONS_INTERVAL_SEC=%s)", interval)
        return
    cfg = office_sessions_config(settings)
    icfg = individuals_config(settings)
    logger.info("vision_office_sessions_loop starting interval_sec=%.0f cfg=%s", interval, cfg)
    backoff: Optional[float] = None
    while True:
        await asyncio.sleep(backoff or interval)
        try:
            summary, opened = await asyncio.to_thread(run_one_office_cycle, postgres_uri=postgres_uri, cfg=cfg,
                                                      icfg=icfg)
            backoff = None
            for ask in opened:
                logger.info("[ORION_ASK] opened ask_id=%s source_ref=%s question=%r (office session)",
                            ask.get("ask_id"), ask.get("source_ref"), ask.get("question"))
            logger.info("[VISION_OFFICE_SESSIONS] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except MigrationMissing as exc:
            backoff = next_backoff(interval, backoff)
            logger.error("[VISION_OFFICE_SESSIONS] migration not applied: %s -- retrying in %.0fs", exc, backoff)
        except Exception as exc:
            logger.warning("vision_office_sessions_cycle_failed error=%s", exc)
