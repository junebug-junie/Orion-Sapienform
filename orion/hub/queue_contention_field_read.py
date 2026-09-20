"""Read-only latest FieldState queue_contention_* for Hub hire disclosure.

Mirrors ``substrate_field_routes._load_latest_field`` / tension trigger SQL:
one ORDER BY generated_at DESC row from ``substrate_field_state``. Never
recomputes EWMA and never falls back to Hub Redis keys.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def reading_from_field_state(state: Any) -> tuple[float | None, str | None]:
    """Extract score/driver from a FieldStateV1-like object.

    Uses real attribute loads so metric-lineage blast radius sees Hub as a
    consumer of ``queue_contention_score`` / ``queue_contention_driver``.
    """
    try:
        score = state.queue_contention_score
        driver = state.queue_contention_driver
        if score is None:
            return None, None
        value = float(score)
        driver_s = str(driver).strip() if driver is not None else None
        if not driver_s:
            driver_s = None
        return value, driver_s
    except Exception:  # noqa: BLE001
        return None, None


def reading_from_field_json(payload: Any) -> tuple[float | None, str | None]:
    """Parse a field_json dict/str into score/driver without recomputing."""
    try:
        data = payload
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return None, None
        from orion.schemas.field_state import FieldStateV1

        return reading_from_field_state(FieldStateV1.model_validate(data))
    except Exception:  # noqa: BLE001
        return None, None


def read_latest_queue_contention(
    *,
    engine: Any | None = None,
) -> tuple[float | None, str | None]:
    """Return ``(queue_contention_score, queue_contention_driver)`` or Nones.

    Fail-open: missing URI, empty table, malformed JSON, or missing fields
    all yield ``(None, None)`` so disclosure omits the queue line.
    """
    try:
        eng = engine
        if eng is None:
            # Shared Hub Postgres factory (same as tension_outreach_trigger).
            from scripts.pg_engine import get_engine

            eng = get_engine()
        if eng is None:
            return None, None

        from sqlalchemy import text

        with eng.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT field_json
                    FROM substrate_field_state
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """
                ),
            ).mappings().first()
        if not row:
            return None, None
        return reading_from_field_json(row.get("field_json"))
    except Exception:  # noqa: BLE001 — disclosure fail-open
        logger.debug("queue_contention_field_read_failed", exc_info=True)
        return None, None
