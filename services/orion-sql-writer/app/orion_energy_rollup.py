"""Hourly kWh per Orion power source, kept after the raw watts are pruned.

Every ORION_ENERGY_ROLLUP_INTERVAL_SEC this recomputes the last few hours of
``orion_energy_hourly`` from ``orion_biometrics_cluster`` (per-node wall watts) and
``home_cooling_sample`` (the cabinet AC plug). An empty table backfills from every raw row
still on disk. The Hub's Orion energy tab reads it (orion-hub/scripts/energy_routes.py).
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text

from app.settings import Settings

logger = logging.getLogger("sql-writer.orion_energy_rollup")

# Late rows (a bus backlog replay) inside this window are picked up on the next pass.
RECOMPUTE_HOURS = 6

# A sample's watts are held until the next sample, but never longer than this. Cluster
# samples arrive every ~15 s and the Shelly plug every ~5 s; a longer silence is a gap and
# is left uncovered instead of being filled in with the last reading.
NODE_MAX_HOLD_SEC = 60
COOLING_MAX_HOLD_SEC = 30

# The WHERE on the upsert keeps a better row: raw rows only ever arrive or get pruned, so a
# recompute that covers less of an hour than the stored row is reading a half-pruned hour.
ROLLUP_SQL = text(
    """
WITH since AS (
  SELECT COALESCE(MAX(hour_start), '-infinity'::timestamptz) - make_interval(hours => :recompute_hours) AS t
  FROM orion_energy_hourly
), s AS (
  SELECT n.key AS source, n.key AS series, c.observed_at AS t,
         (n.value->>'chassis_watts')::float AS w, :node_hold AS hold
  FROM orion_biometrics_cluster c, jsonb_each(c.measurements_by_node) n
  WHERE c.observed_at >= (SELECT t FROM since)
    AND jsonb_typeof(n.value->'chassis_watts') = 'number'
  UNION ALL
  SELECT 'cooling', zwave_node_id::text, ts, cooling_watts, :cooling_hold
  FROM home_cooling_sample
  WHERE ts >= (SELECT t FROM since)
    AND cooling_watts IS NOT NULL AND NOT COALESCE(stale, false) AND device_online
), d AS (
  SELECT source, t, w,
         LEAST(EXTRACT(EPOCH FROM LEAD(t) OVER (PARTITION BY source, series ORDER BY t) - t), hold) AS dt
  FROM s
)
INSERT INTO orion_energy_hourly (hour_start, source, kwh, covered_sec, sample_count, updated_at)
SELECT date_trunc('hour', t), source, SUM(w * dt) / 3600000.0, SUM(dt), COUNT(*), now()
FROM d
WHERE dt IS NOT NULL AND dt >= 0
GROUP BY 1, 2
ON CONFLICT (hour_start, source) DO UPDATE
SET kwh = EXCLUDED.kwh, covered_sec = EXCLUDED.covered_sec,
    sample_count = EXCLUDED.sample_count, updated_at = EXCLUDED.updated_at
WHERE EXCLUDED.covered_sec >= orion_energy_hourly.covered_sec
"""
)


def run_one_rollup(engine=None) -> int:
    if engine is None:
        from app.db import engine

    with engine.begin() as conn:
        result = conn.execute(
            ROLLUP_SQL,
            {
                "recompute_hours": RECOMPUTE_HOURS,
                "node_hold": NODE_MAX_HOLD_SEC,
                "cooling_hold": COOLING_MAX_HOLD_SEC,
            },
        )
        return int(result.rowcount or 0)


async def orion_energy_rollup_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "orion_energy_rollup_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.warning(
            "orion_energy_rollup_loop DISABLED (ORION_ENERGY_ROLLUP_INTERVAL_SEC=%s); Orion's "
            "per-node kWh is lost once raw cluster watts are pruned",
            interval,
        )
        return
    logger.info("orion_energy_rollup_loop starting interval_sec=%.0f", interval)
    while True:
        try:
            rows = await asyncio.to_thread(run_one_rollup)
            logger.info("[ORION_ENERGY_ROLLUP] tick_complete rows_upserted=%d", rows)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("orion_energy_rollup_failed error=%s", exc)
        await asyncio.sleep(interval)
