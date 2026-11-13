"""Bounded, fail-open read of the office sound level for the situation brief.

The office sound meter (DFRobot SEN0232 on pan-1's office Nano) is a real
dBA meter, unlike the cabinet's uncalibrated USB mic. orion-office-env
converts it to `office_sound_dba` and orion-sql-writer stores it in
`office_env_sample.payload_json -> 'measurements'`, one row about every
30 s (producer `services/orion-office-env/app/sample.py` `_sound_reading`).
This is the office where Juniper works, not the server cabinet and not the
closet.

One query returns the newest row plus the last 24 h of readings, so the
brief can say both "how loud is it now" (in dBA) and "is that usual for
this office". Same engine and timeouts as `cabinet_sound_reader`; never
raises into turn assembly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional

from sqlalchemy import text

from .cabinet_sound_reader import MIN_BASELINE_ROWS, _get_engine

logger = logging.getLogger(__name__)

_DBA = "(payload_json->'measurements'->>'office_sound_dba')::float"


class OfficeSound(NamedTuple):
    """Newest office sound reading plus its 24 h band (dBA)."""

    dba: float
    received_at: datetime
    # Band fields are None when the last 24 h holds fewer than MIN_BASELINE_ROWS readings.
    p10_dba: Optional[float]
    median_dba: Optional[float]
    p90_dba: Optional[float]
    recent_median_dba: Optional[float]  # last `recent_minutes`; None if no rows


def fetch_office_sound(
    *,
    node: str,
    baseline_hours: int = 24,
    recent_minutes: int = 10,
    now: Optional[datetime] = None,
) -> Optional[OfficeSound]:
    """None on no DSN, no row with a sound reading, or any error. The newest row
    WITH a reading is used: one out-of-range read (no value) must not hide the last one.

    Staleness is the caller's call: it gets `received_at`, not a verdict."""
    engine = _get_engine()
    if engine is None:
        return None
    now_dt = now or datetime.now(timezone.utc)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "WITH latest AS ("
                    f"  SELECT ts, {_DBA} AS dba FROM office_env_sample"
                    "   WHERE node = :node AND NOT stale"
                    f"   AND {_DBA} IS NOT NULL ORDER BY ts DESC LIMIT 1"
                    "), band AS ("
                    "  SELECT count(*) AS n, "
                    f"  percentile_cont(0.1) WITHIN GROUP (ORDER BY {_DBA}) AS p10, "
                    f"  percentile_cont(0.5) WITHIN GROUP (ORDER BY {_DBA}) AS p50, "
                    f"  percentile_cont(0.9) WITHIN GROUP (ORDER BY {_DBA}) AS p90, "
                    f"  percentile_cont(0.5) WITHIN GROUP (ORDER BY {_DBA}) "
                    "     FILTER (WHERE ts >= :recent_cutoff) AS recent "
                    "  FROM office_env_sample "
                    "  WHERE node = :node AND ts >= :cutoff AND NOT stale "
                    f"  AND {_DBA} IS NOT NULL"
                    ") SELECT latest.ts, latest.dba, band.n, band.p10, band.p50, band.p90, band.recent "
                    "FROM latest CROSS JOIN band"
                ),
                {
                    "node": node,
                    "cutoff": now_dt - timedelta(hours=baseline_hours),
                    "recent_cutoff": now_dt - timedelta(minutes=recent_minutes),
                },
            ).one_or_none()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_office_sound_read_failed err=%s", exc)
        return None

    if row is None:
        return None
    ts, dba, count, p10, p50, p90, recent = row
    if dba is None or ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    has_band = bool(count) and count >= MIN_BASELINE_ROWS and None not in (p10, p50, p90)
    return OfficeSound(
        dba=float(dba),
        received_at=ts.astimezone(timezone.utc),
        p10_dba=float(p10) if has_band else None,
        median_dba=float(p50) if has_band else None,
        p90_dba=float(p90) if has_band else None,
        recent_median_dba=float(recent) if recent is not None else None,
    )
