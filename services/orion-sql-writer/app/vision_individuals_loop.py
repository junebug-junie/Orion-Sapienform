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
import time
from typing import Optional

from app.person_record import PersonRecordConfig
from app.settings import Settings
from app.vision_individuals import IndividualsConfig, MigrationMissing, run_one_individuals_cycle
from app.vision_person_model import PersonModelConfig

logger = logging.getLogger("sql-writer.vision_individuals_loop")

MAX_BACKOFF_SEC = 3600.0


def next_backoff(interval: float, current: Optional[float]) -> float:
    return min(MAX_BACKOFF_SEC, max(interval, (current or interval) * 2.0))


def individuals_config(settings: Settings) -> IndividualsConfig:
    return IndividualsConfig(
        match_threshold=float(settings.vision_individuals_match_threshold),
        continuation_enabled=bool(settings.vision_individuals_continuation_enabled),
        continuation_threshold=float(settings.vision_individuals_continuation_threshold),
        continuation_gap_sec=float(settings.vision_individuals_continuation_gap_sec),
        continuation_max_disp=float(settings.vision_individuals_continuation_max_disp),
        merge_gap_sec=float(settings.vision_individuals_merge_gap_sec),
        batch_rows=int(settings.vision_individuals_batch_rows),
        lookback_ceiling_sec=float(settings.vision_individuals_lookback_ceiling_sec),
        crop_retention_days=float(settings.vision_crop_retention_days),
        crop_retention_keep_labels=tuple(
            lbl.strip() for lbl in str(settings.vision_crop_retention_keep_labels).split(",")
            if lbl.strip() and lbl.strip() != "0"),
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
        thumb_retention_days=float(settings.vision_crop_thumb_retention_days),
        face_match_threshold=float(settings.vision_individuals_face_match_threshold),
        stationary_min_sec=float(settings.vision_stationary_min_sec),
        stationary_iou=float(settings.vision_stationary_iou),
        stationary_max_gap_sec=float(settings.vision_stationary_max_gap_sec),
        ask_face_min_faces=int(settings.vision_ask_face_min_faces),
        same_ask_min_face_cosine=float(settings.vision_same_ask_min_face_cosine),
        confirm_ask_daily_cap=int(settings.vision_confirm_ask_daily_cap),
        person_confirm_ask_daily_cap=int(settings.vision_person_confirm_ask_daily_cap),
        split_on_verdict_enabled=bool(settings.vision_split_on_verdict_enabled),
        split_follow_min_face_cosine=float(settings.vision_split_follow_min_face_cosine),
        split_follow_margin=float(settings.vision_split_follow_margin),
        split_max_move_fraction=float(settings.vision_split_max_move_fraction),
        cannot_link_margin=float(settings.vision_individuals_cannot_link_margin),
    )


def person_record_config(settings: Settings) -> Optional[PersonRecordConfig]:
    """None when PERSON_RECORD_ENABLED is off: no person links, facts or person questions."""
    if not settings.person_record_enabled:
        return None
    return PersonRecordConfig(
        ask_daily_cap=int(settings.person_fact_ask_daily_cap),
        ask_cooldown_days=float(settings.vision_ask_cooldown_days),
        ask_expiry_days=float(settings.vision_ask_expiry_days),
        local_tz=str(settings.vision_local_tz),
    )


def person_model_config(settings: Settings) -> Optional[PersonModelConfig]:
    """None when VISION_PERSON_MODEL_ENABLED is off: no album, bars or guesses are written."""
    if not settings.vision_person_model_enabled:
        return None
    return PersonModelConfig(
        alpha=float(settings.vision_person_false_accept_alpha),
        top_k=int(settings.vision_person_top_k),
        fallback_channel=str(settings.vision_person_fallback_channel),
        candidate_days=float(settings.vision_individuals_candidate_days),
        local_tz=str(settings.vision_local_tz),
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
    person_cfg = person_model_config(settings)
    record_cfg = person_record_config(settings)
    person_every = float(settings.vision_person_model_interval_sec)
    logger.info("vision_individuals_loop starting interval_sec=%.0f cfg=%s zone_streams=%s person_model=%s person_record=%s",
                interval, cfg, sorted(zones), person_cfg, record_cfg)
    backoff: Optional[float] = None
    person_last = 0.0
    while True:
        await asyncio.sleep(backoff or interval)
        try:
            run_person = person_cfg is not None and time.monotonic() - person_last >= person_every
            summary, opened = await asyncio.to_thread(
                run_one_individuals_cycle, postgres_uri=postgres_uri, cfg=cfg, zones_by_stream=zones,
                person_cfg=person_cfg if run_person else None, person_record_cfg=record_cfg)
            if run_person and "person_people" in summary:   # it really ran (not skipped / failed)
                person_last = time.monotonic()
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
