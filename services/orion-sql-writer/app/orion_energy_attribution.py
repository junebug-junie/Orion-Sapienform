"""Hourly attribution of each host's above-rest energy: orion_energy_attribution_hourly.

Every ORION_ENERGY_ATTRIBUTION_INTERVAL_SEC this recomputes the last RECOMPUTE_HOURS complete UTC
hours for every host in config/physiology/rest_body.yaml (backfilling up to BACKFILL_HOURS when the
table is empty or behind), with the same split the per-act meter uses
(orion/grammar/act_energy_fill.py ``attribute_host_hour``). Recomputing recent hours picks up acts
that opened after their leases ran.

Closure, per row: above_rest = sponsored + metered + unattributed. Unattributed is stored as
computed; a negative value is over-attribution and is logged as a warning, never clipped.

``metered_wh`` is what is ATTRIBUTABLE to acts: the share of leases whose turn has an act. What acts
were actually CHARGED is the sum of ``grammar_acts.cost_json.wh_by_host``; the physiology report
compares the two per day (double counting shows there, e.g. two acts on one correlation_id).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.settings import Settings

logger = logging.getLogger("sql-writer.orion_energy_attribution")

RECOMPUTE_HOURS = 3
BACKFILL_HOURS = 48
# An hour is computed once its last samples have landed (30 s cadence, bus lag).
SETTLE = timedelta(minutes=3)
# Over-attribution beyond float noise.
NEGATIVE_TOLERANCE_WH = 0.01

# Cheap pre-check: a host with no chassis reading in the span (hecate asleep, any host offline) is
# skipped instead of recomputing its whole backfill every tick.
_HAS_CHASSIS_SQL = text(
    "SELECT 1 FROM orion_biometrics_summary WHERE node = :node AND timestamp >= :since AND timestamp < :until "
    "AND measurements ? 'chassis_watts' LIMIT 1"
)

_LATEST_SQL = text("SELECT MAX(hour_start) FROM orion_energy_attribution_hourly WHERE host = :host")

UPSERT_SQL = text(
    """
INSERT INTO orion_energy_attribution_hourly (hour_start, host, rest_w, rest_wh, above_rest_wh, sponsored_wh,
    metered_wh, leased_unmetered_wh, unattributed_wh, covered_sec, split_unknown_sec, rest_source, updated_at)
VALUES (:hour_start, :host, :rest_w, :rest_wh, :above_rest_wh, :sponsored_wh, :metered_wh, :leased_unmetered_wh,
    :unattributed_wh, :covered_sec, :split_unknown_sec, :rest_source, now())
ON CONFLICT (hour_start, host) DO UPDATE
SET rest_w = EXCLUDED.rest_w, rest_wh = EXCLUDED.rest_wh, above_rest_wh = EXCLUDED.above_rest_wh,
    sponsored_wh = EXCLUDED.sponsored_wh, metered_wh = EXCLUDED.metered_wh,
    leased_unmetered_wh = EXCLUDED.leased_unmetered_wh, unattributed_wh = EXCLUDED.unattributed_wh,
    covered_sec = EXCLUDED.covered_sec, split_unknown_sec = EXCLUDED.split_unknown_sec,
    rest_source = EXCLUDED.rest_source, updated_at = EXCLUDED.updated_at
"""
)


def hours_to_compute(latest: datetime | None, now: datetime) -> list[datetime]:
    """Complete UTC hours to (re)compute, oldest first: the last RECOMPUTE_HOURS, plus any gap since
    the newest stored row, never more than BACKFILL_HOURS back."""
    last = (now - SETTLE).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    first = last - timedelta(hours=RECOMPUTE_HOURS - 1)
    if latest is None:
        first = last - timedelta(hours=BACKFILL_HOURS - 1)
    elif latest + timedelta(hours=1) < first:
        first = max(latest + timedelta(hours=1), last - timedelta(hours=BACKFILL_HOURS - 1))
    return [first + timedelta(hours=i) for i in range(int((last - first) / timedelta(hours=1)) + 1)]


def run_one_attribution(engine=None, now: datetime | None = None) -> int:
    from orion.grammar.act_energy_fill import rest_body_config

    if engine is None:
        from app.db import engine
    now = now or datetime.now(timezone.utc)
    cfg, err = rest_body_config()
    if cfg is None:
        logger.warning("orion_energy_attribution_skipped reason=%s", err)
        return 0
    for host, why in cfg.refused.items():
        logger.warning("orion_energy_attribution_host_refused host=%s reason=%s", host, why)
    written = 0
    for host in sorted(cfg.hosts):
        try:
            written += _run_host(engine, host, now)
        except Exception as exc:  # noqa: BLE001 -- one host failing must not stop the others
            logger.warning("orion_energy_attribution_host_failed host=%s error=%s", host, exc, exc_info=True)
    return written


def _run_host(engine, host: str, now: datetime) -> int:
    from orion.grammar.act_energy_fill import attribute_host_hour

    with engine.connect() as conn:
        latest = conn.execute(_LATEST_SQL, {"host": host}).scalar()
        latest = latest.astimezone(timezone.utc) if latest is not None else None
        hours = hours_to_compute(latest, now)
        if not hours:
            return 0
        fmt = "%Y-%m-%d %H:%M:%S"
        if conn.execute(_HAS_CHASSIS_SQL, {"node": host, "since": hours[0].strftime(fmt),
                                           "until": (hours[-1] + timedelta(hours=1)).strftime(fmt)}).first() is None:
            return 0
    written = 0
    for hour in hours:
        with engine.begin() as conn:
            got, rest_w, source = attribute_host_hour(conn, host, hour, hour + timedelta(hours=1), now=now)
            if got is None or got.covered_s <= 0:
                continue  # no chassis reading this hour: unknown, no row (never 0)
            un = got.unattributed_wh
            if un < -NEGATIVE_TOLERANCE_WH:
                logger.warning("orion_energy_attribution_negative host=%s hour=%s unattributed_wh=%.3f "
                               "above_rest_wh=%.3f sponsored_wh=%.3f metered_wh=%.3f (over-attribution)",
                               host, hour.isoformat(), un, got.above_rest_wh, got.sponsored_wh, got.metered_wh)
            conn.execute(UPSERT_SQL, {
                "hour_start": hour, "host": host, "rest_w": rest_w, "rest_wh": got.rest_wh,
                "above_rest_wh": got.above_rest_wh, "sponsored_wh": got.sponsored_wh,
                "metered_wh": got.metered_wh, "leased_unmetered_wh": got.leased_unmetered_wh,
                "unattributed_wh": un, "covered_sec": got.covered_s, "split_unknown_sec": got.split_unknown_s,
                "rest_source": source})
            written += 1
    return written


async def orion_energy_attribution_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "orion_energy_attribution_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.warning("orion_energy_attribution_loop DISABLED (ORION_ENERGY_ATTRIBUTION_INTERVAL_SEC=%s); no "
                       "hourly rest / sponsored / metered / unattributed energy per host", interval)
        return
    logger.info("orion_energy_attribution_loop starting interval_sec=%.0f", interval)
    while True:
        try:
            rows = await asyncio.to_thread(run_one_attribution)
            logger.info("[ORION_ENERGY_ATTRIBUTION] tick_complete rows_upserted=%d", rows)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("orion_energy_attribution_failed error=%s", exc, exc_info=True)
        await asyncio.sleep(interval)
