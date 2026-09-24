"""Clocked wrappers for app/vision_rhythm.py.

Same shape as vision_object_permanence_loop.py, plus the migration-missing
backoff shared with vision_individuals_loop.py. Writes the
``orion:vision:expect:<stream>`` Redis key on the ORION_BUS_URL Redis.

Two loops: the rhythm cycle (fit, emit, grade; VISION_RHYTHM_INTERVAL_SEC,
900 s) and the cheap expect-key refresh (open-window query + SET/DEL only;
VISION_EXPECT_REFRESH_INTERVAL_SEC, 60 s), so attention steering starts
within a minute of a window opening instead of up to a rhythm tick late.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.settings import Settings
from app.vision_individuals import MigrationMissing
from app.vision_individuals_loop import next_backoff
from app.vision_rhythm import RhythmConfig, run_one_expect_refresh, run_one_rhythm_cycle

logger = logging.getLogger("sql-writer.vision_rhythm_loop")


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in (raw or "").split(",") if p.strip())


def rhythm_config(settings: Settings) -> RhythmConfig:
    return RhythmConfig(
        min_occurrences=int(settings.vision_rhythm_min_occurrences),
        min_days=int(settings.vision_rhythm_min_days),
        history_days=float(settings.vision_rhythm_history_days),
        bandwidth_min=float(settings.vision_rhythm_bandwidth_min),
        min_confidence=float(settings.vision_rhythm_min_confidence),
        max_per_subject=int(settings.vision_rhythm_max_per_subject),
        score_lag_sec=float(settings.vision_rhythm_score_lag_sec),
        labels=_csv(settings.vision_rhythm_labels),
        label_streams=_csv(settings.vision_rhythm_label_streams),
        arrival_gap_sec=float(settings.vision_rhythm_arrival_gap_sec),
        local_tz=str(settings.vision_local_tz),
        min_coverage=float(settings.vision_rhythm_min_coverage),
    )


async def vision_rhythm_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "vision_rhythm_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if interval <= 0 or not postgres_uri:
        logger.info("vision_rhythm_loop DISABLED (VISION_RHYTHM_INTERVAL_SEC=%s)", interval)
        return
    cfg = rhythm_config(settings)
    redis_url = str(settings.orion_bus_url or "") or None
    logger.info("vision_rhythm_loop starting interval_sec=%.0f cfg=%s", interval, cfg)
    backoff: Optional[float] = None
    while True:
        await asyncio.sleep(backoff or interval)
        try:
            summary = await asyncio.to_thread(
                run_one_rhythm_cycle, postgres_uri=postgres_uri, cfg=cfg, redis_url=redis_url)
            backoff = None
            logger.info("[VISION_RHYTHM] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except MigrationMissing as exc:
            backoff = next_backoff(interval, backoff)
            logger.error("[VISION_RHYTHM] migration not applied: %s -- retrying in %.0fs", exc, backoff)
        except Exception as exc:
            logger.warning("vision_rhythm_cycle_failed error=%s", exc)


async def vision_expect_refresh_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "vision_expect_refresh_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    redis_url = str(settings.orion_bus_url or "") or None
    if interval <= 0 or not postgres_uri or not redis_url:
        logger.info("vision_expect_refresh_loop DISABLED (VISION_EXPECT_REFRESH_INTERVAL_SEC=%s)", interval)
        return
    from sqlalchemy import create_engine

    # One small pool for the life of the loop: this ticks every minute.
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
    logger.info("vision_expect_refresh_loop starting interval_sec=%.0f", interval)
    backoff: Optional[float] = None
    try:
        while True:
            await asyncio.sleep(backoff or interval)
            try:
                summary = await asyncio.to_thread(run_one_expect_refresh, engine=engine, redis_url=redis_url)
                backoff = None
                if summary.get("open_streams"):
                    logger.debug("[VISION_EXPECT] refresh %s", summary)
            except asyncio.CancelledError:
                raise
            except MigrationMissing as exc:
                backoff = next_backoff(interval, backoff)
                logger.error("[VISION_EXPECT] migration not applied: %s -- retrying in %.0fs", exc, backoff)
            except Exception as exc:
                logger.warning("vision_expect_refresh_failed error=%s", exc)
    finally:
        engine.dispose()
