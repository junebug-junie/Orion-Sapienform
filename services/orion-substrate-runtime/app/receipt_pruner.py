from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.receipt_pruner")

UNAPPLIED_STATE_DELTAS_EXISTS = """
      SELECT 1
      FROM jsonb_array_elements(
        CASE
          WHEN jsonb_typeof(r.receipt_json->'state_deltas') = 'array'
          THEN r.receipt_json->'state_deltas'
          ELSE '[]'::jsonb
        END
      ) AS d(elem)
      WHERE (d.elem->>'delta_id') IS NOT NULL
        AND (d.elem->>'delta_id') NOT IN (
          SELECT delta_id FROM substrate_field_applied_deltas
        )
"""

SAFE_PRUNE_WHERE = f"""
  r.expires_at IS NOT NULL
  AND r.expires_at < now()
  AND (
    r.receipt_status = 'error'
    OR NOT EXISTS (
{UNAPPLIED_STATE_DELTAS_EXISTS}
    )
  )
"""

SAFE_PRUNE_BATCH_DELETE_SQL = f"""
DELETE FROM substrate_reduction_receipts
WHERE ctid IN (
  SELECT r.ctid
  FROM substrate_reduction_receipts r
  WHERE {SAFE_PRUNE_WHERE}
  LIMIT :batch_size
)
"""

# Backward-compatible export for tests that assert SQL shape.
SAFE_PRUNE_DELETE_SQL_FRAGMENT = f"""
DELETE FROM substrate_reduction_receipts r
WHERE {SAFE_PRUNE_WHERE}
"""

_cached_disk_critical = False
_cached_table_critical = False
_last_emergency_prune_monotonic = 0.0


def table_size_gb(engine: Engine) -> float:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT pg_total_relation_size('substrate_reduction_receipts') AS bytes"
            )
        ).mappings().first()
    bytes_val = int(row["bytes"] or 0)
    return bytes_val / (1024**3)


def disk_usage_pct(path: str) -> float | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    if usage.total == 0:
        return None
    return 100.0 * usage.used / usage.total


def measure_pressure_state(engine: Engine, settings: Settings) -> tuple[bool, bool]:
    disk_critical = False
    if os.path.exists(settings.receipt_postgres_data_path):
        disk_pct = disk_usage_pct(settings.receipt_postgres_data_path)
        if disk_pct is not None:
            disk_critical = disk_pct >= settings.receipt_disk_critical_pct

    size_gb = table_size_gb(engine)
    table_critical = size_gb >= settings.receipt_critical_table_gb
    return disk_critical, table_critical


def refresh_pressure_cache(engine: Engine, settings: Settings) -> tuple[bool, bool]:
    global _cached_disk_critical, _cached_table_critical
    _cached_disk_critical, _cached_table_critical = measure_pressure_state(engine, settings)
    return _cached_disk_critical, _cached_table_critical


def get_cached_pressure_state() -> tuple[bool, bool]:
    return _cached_disk_critical, _cached_table_critical


def run_safe_prune(engine: Engine, *, batch_size: int = 10000) -> int:
    total_deleted = 0
    while True:
        with engine.begin() as conn:
            result = conn.execute(text(SAFE_PRUNE_BATCH_DELETE_SQL), {"batch_size": batch_size})
        deleted = result.rowcount or 0
        total_deleted += deleted
        if deleted < batch_size:
            break
    if total_deleted:
        logger.warning("substrate_receipt_safe_prune deleted=%s", total_deleted)
    return total_deleted


def run_emergency_prune(engine: Engine, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    debug_cutoff = now - timedelta(minutes=s.receipt_retention_success_minutes)
    error_cutoff = now - timedelta(hours=s.receipt_retention_error_hours)
    unapplied_guard = f"NOT EXISTS ({UNAPPLIED_STATE_DELTAS_EXISTS})"
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                DELETE FROM substrate_reduction_receipts r
                WHERE r.expires_at IS NOT NULL AND r.expires_at < :cutoff
                  AND r.receipt_kind = 'success' AND r.receipt_status = 'ok'
                  AND r.is_full_payload = false
                  AND {unapplied_guard}
                """
            ),
            {"cutoff": now},
        )
        conn.execute(
            text(
                f"""
                DELETE FROM substrate_reduction_receipts r
                WHERE r.receipt_kind = 'debug_sample'
                  AND r.created_at < :cutoff
                  AND {unapplied_guard}
                """
            ),
            {"cutoff": debug_cutoff},
        )
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_receipts
                WHERE receipt_kind = 'error'
                  AND created_at < :cutoff
                """
            ),
            {"cutoff": error_cutoff},
        )
    logger.error("substrate_receipt_emergency_prune_completed")


def maybe_run_emergency_prune(engine: Engine, settings: Settings) -> bool:
    global _last_emergency_prune_monotonic

    disk_critical, table_critical = refresh_pressure_cache(engine, settings)
    if not (disk_critical or table_critical):
        return False
    if settings.receipt_max_table_gb <= 0:
        return False

    cooldown = float(settings.receipt_prune_interval_sec)
    now_mono = time.monotonic()
    if now_mono - _last_emergency_prune_monotonic < cooldown:
        return False

    run_emergency_prune(engine, settings)
    _last_emergency_prune_monotonic = now_mono
    return True


def log_receipt_pressure(engine: Engine, settings: Settings) -> None:
    size_gb = table_size_gb(engine)
    disk_pct = disk_usage_pct(settings.receipt_postgres_data_path)
    if size_gb >= settings.receipt_warn_table_gb:
        logger.warning(
            "substrate_receipt_table_pressure size_gb=%.2f warn_gb=%.2f",
            size_gb,
            settings.receipt_warn_table_gb,
        )
    if size_gb >= settings.receipt_critical_table_gb:
        logger.error(
            "substrate_receipt_table_critical size_gb=%.2f critical_gb=%.2f",
            size_gb,
            settings.receipt_critical_table_gb,
        )
    if disk_pct is not None and disk_pct >= settings.receipt_disk_critical_pct:
        logger.error(
            "substrate_postgres_disk_critical path=%s used_pct=%.1f",
            settings.receipt_postgres_data_path,
            disk_pct,
        )
