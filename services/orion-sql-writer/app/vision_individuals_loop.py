"""Clocked wrapper for app/vision_individuals.py.

Same shape as vision_object_permanence_loop.py: sleep first, one bounded pass
per interval in a worker thread, best-effort. One addition: while the walkway
migration is not applied, the loop logs one clear ERROR and backs off
(doubling, capped at an hour) instead of warning every tick.

Opened asks are durable rows in ``orion_ask``; the Hub's ask card reads that
table. There is deliberately no ``orion:ask:opened`` bus event: nothing would
subscribe to it (the Hub has no push channel for panels), and the metric
lineage gate refuses channels with no consumer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.settings import Settings
from app.vision_individuals import IndividualsConfig, MigrationMissing, run_one_individuals_cycle

logger = logging.getLogger("sql-writer.vision_individuals_loop")

MAX_BACKOFF_SEC = 3600.0


def next_backoff(interval: float, current: Optional[float]) -> float:
    return min(MAX_BACKOFF_SEC, max(interval, (current or interval) * 2.0))


def individuals_config(settings: Settings) -> IndividualsConfig:
    return IndividualsConfig(
        match_threshold=float(settings.vision_individuals_match_threshold),
        merge_gap_sec=float(settings.vision_individuals_merge_gap_sec),
        batch_rows=int(settings.vision_individuals_batch_rows),
        lookback_ceiling_sec=float(settings.vision_individuals_lookback_ceiling_sec),
        crop_retention_days=float(settings.vision_crop_retention_days),
        sighting_retention_days=float(settings.vision_sighting_retention_days),
        patio_present_sec=float(settings.vision_patio_present_sec),
        patio_grace_sec=float(settings.vision_patio_grace_sec),
        attention_threshold=float(settings.vision_attention_threshold),
        ask_min_sightings=int(settings.vision_ask_min_sightings),
        ask_min_days=int(settings.vision_ask_min_days),
        ask_expiry_days=float(settings.vision_ask_expiry_days),
        ask_daily_cap=int(settings.orion_ask_daily_cap),
        local_tz=str(settings.vision_local_tz),
        settle_sec=float(settings.vision_individuals_settle_sec),
        candidate_days=float(settings.vision_individuals_candidate_days),
        ask_cooldown_days=float(settings.vision_ask_cooldown_days),
    )


async def vision_individuals_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "vision_individuals_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if interval <= 0 or not postgres_uri:
        logger.info("vision_individuals_loop DISABLED (VISION_INDIVIDUALS_INTERVAL_SEC=%s)", interval)
        return
    from orion.vision.zones import load_zones

    try:
        zones = load_zones()
    except Exception as exc:
        logger.error("vision_zones_load_failed error=%s -- running with no zones (no patio rule here!)", exc)
        zones = {}
    cfg = individuals_config(settings)
    logger.info("vision_individuals_loop starting interval_sec=%.0f cfg=%s zone_streams=%s",
                interval, cfg, sorted(zones))
    backoff: Optional[float] = None
    while True:
        await asyncio.sleep(backoff or interval)
        try:
            summary, opened = await asyncio.to_thread(
                run_one_individuals_cycle, postgres_uri=postgres_uri, cfg=cfg, zones_by_stream=zones)
            backoff = None
            for ask in opened:
                logger.info("[ORION_ASK] opened ask_id=%s source_ref=%s question=%r",
                            ask.get("ask_id"), ask.get("source_ref"), ask.get("question"))
            logger.info("[VISION_INDIVIDUALS] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except MigrationMissing as exc:
            backoff = next_backoff(interval, backoff)
            logger.error("[VISION_INDIVIDUALS] migration not applied: %s -- retrying in %.0fs", exc, backoff)
        except Exception as exc:
            logger.warning("vision_individuals_cycle_failed error=%s", exc)
