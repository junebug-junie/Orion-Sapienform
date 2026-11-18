"""Bounded, fail-open read of the cabinet mic's recent loudness history for
the situation brief.

The live mic file (`/run/orion-audio/latest.json`) only says how loud the
cabinet is in the last half second. To tell Orion whether that is loud *for
this cabinet*, this reads what orion-biometrics already stored:
`orion_biometrics_summary.measurements->>'cabinet_ambient_rms'` on node
`athena`, one row about every 30 s (written by sql-writer from the
biometrics summary; producer `orion/telemetry/ambient_audio.py`
`extract_ambient_audio_measurements`).

Why stored history and not an in-process EWMA: the situation brief only
runs on chat turns. An EWMA fed per turn samples when someone talks, not
when the fans change, and resets to "whatever it heard first" on every Hub
restart. The stored rows are the real 24 h of readings.

Same shape as `reverie_reader.py` / `perception_reader.py`: module-level
cached engine, DSN from `POSTGRES_URI` -> `DATABASE_URL`, per-connection
statement and connect timeouts, never raises into turn assembly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_ENGINE = None
_ENGINE_URL: str | None = None

_QUERY_STATEMENT_TIMEOUT_MS = 1500
_CONNECT_TIMEOUT_SEC = 3

# Rows stored per day at ~30 s cadence is ~2,800; fewer than this in the
# baseline window means the history is too thin to call anything "usual".
MIN_BASELINE_ROWS = 60


def _dsn() -> str:
    return (os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL") or "").strip()


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}",
                "connect_timeout": _CONNECT_TIMEOUT_SEC,
            },
        )
        _ENGINE_URL = url
    return _ENGINE


def biometrics_summary_cutoff(value: datetime) -> str:
    """UTC instant formatted for TEXT compares on
    `orion_biometrics_summary.timestamp` (varchar shaped
    `YYYY-MM-DD HH:MM:SS.ffffff+00`). A text compare keeps the
    `(node, timestamp)` index usable; casting the column to timestamptz
    made the same 24 h query ~10x slower (405 ms vs 36 ms, measured live
    2026-10-09). Same format Hub's `cabinet_ambient_routes.py` uses."""
    v = value.astimezone(timezone.utc)
    return (
        f"{v.year:04d}-{v.month:02d}-{v.day:02d} "
        f"{v.hour:02d}:{v.minute:02d}:{v.second:02d}.{v.microsecond:06d}+00"
    )


class CabinetSoundHistory(NamedTuple):
    """Cabinet mic RMS (16-bit PCM units) over the last day."""

    rows: int
    p10_rms: float
    median_rms: float
    p90_rms: float
    recent_median_rms: Optional[float]  # last `recent_minutes`; None if no rows


def fetch_cabinet_sound_history(
    *,
    node: str = "athena",
    baseline_hours: int = 24,
    recent_minutes: int = 10,
    now: Optional[datetime] = None,
) -> Optional[CabinetSoundHistory]:
    """Loudness percentiles over the last `baseline_hours`, plus the median of
    the last `recent_minutes`. None on no DSN, too few rows, or any error."""
    engine = _get_engine()
    if engine is None:
        return None
    now_dt = now or datetime.now(timezone.utc)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT count(*), "
                    "percentile_cont(0.1) WITHIN GROUP (ORDER BY (measurements->>'cabinet_ambient_rms')::float), "
                    "percentile_cont(0.5) WITHIN GROUP (ORDER BY (measurements->>'cabinet_ambient_rms')::float), "
                    "percentile_cont(0.9) WITHIN GROUP (ORDER BY (measurements->>'cabinet_ambient_rms')::float), "
                    "percentile_cont(0.5) WITHIN GROUP (ORDER BY (measurements->>'cabinet_ambient_rms')::float) "
                    "  FILTER (WHERE timestamp >= :recent_cutoff) "
                    "FROM orion_biometrics_summary "
                    "WHERE node = :node AND timestamp >= :cutoff "
                    "AND measurements ? 'cabinet_ambient_rms'"
                ),
                {
                    "node": node,
                    "cutoff": biometrics_summary_cutoff(now_dt - timedelta(hours=baseline_hours)),
                    "recent_cutoff": biometrics_summary_cutoff(now_dt - timedelta(minutes=recent_minutes)),
                },
            ).one()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_cabinet_sound_read_failed err=%s", exc)
        return None

    count, p10, p50, p90, recent = row
    if not count or count < MIN_BASELINE_ROWS or p10 is None or p50 is None or p90 is None:
        return None
    return CabinetSoundHistory(
        rows=int(count),
        p10_rms=float(p10),
        median_rms=float(p50),
        p90_rms=float(p90),
        recent_median_rms=float(recent) if recent is not None else None,
    )
