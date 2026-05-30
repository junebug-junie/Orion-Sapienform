from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.receipt_pruner")

SAFE_PRUNE_DELETE_SQL_FRAGMENT = """
DELETE FROM substrate_reduction_receipts r
WHERE r.expires_at IS NOT NULL
  AND r.expires_at < now()
  AND (
    r.receipt_status = 'error'
    OR NOT EXISTS (
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
    )
  )
"""


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


def run_safe_prune(engine: Engine) -> int:
    with engine.begin() as conn:
        result = conn.execute(text(SAFE_PRUNE_DELETE_SQL_FRAGMENT))
    deleted = result.rowcount or 0
    if deleted:
        logger.warning("substrate_receipt_safe_prune deleted=%s", deleted)
    return deleted


def run_emergency_prune(engine: Engine, *, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_receipts
                WHERE expires_at IS NOT NULL AND expires_at < :cutoff
                  AND receipt_kind = 'success' AND receipt_status = 'ok'
                  AND is_full_payload = false
                """
            ),
            {"cutoff": now},
        )
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_receipts
                WHERE receipt_kind = 'debug_sample'
                  AND created_at < :cutoff
                """
            ),
            {"cutoff": now - timedelta(hours=24)},
        )
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_receipts
                WHERE receipt_kind = 'error'
                  AND created_at < :cutoff
                """
            ),
            {"cutoff": now - timedelta(hours=48)},
        )
    logger.error("substrate_receipt_emergency_prune_completed")


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
