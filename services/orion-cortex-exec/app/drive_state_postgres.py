"""Bounded fail-open latest drive_audits fetch for chat stance.

Mirrors Mind/Thought's Postgres measurement rail (bus → sql-writer →
drive_audits). Maps the latest subject='orion' row into the chat_drive_state
shape expected by autonomy_slice / router. Never raises to callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DRIVE_AUDITS_LATEST_QUERY_FOR_STANCE = (
    "SELECT dominant_drive, active_drives, drive_pressures, summary, tension_kinds, "
    "COALESCE(observed_at, created_at) AS observed_at "
    "FROM drive_audits WHERE subject = 'orion' "
    "ORDER BY COALESCE(observed_at, created_at) DESC LIMIT 1"
)

_DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS = 300
# Hard cap for the in-transaction Postgres statement_timeout. Keep this at or
# below CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC (default 0.4s) so the asyncio
# wait_for and the DB kill agree on budget. Raising the env timeout alone does
# not raise this cap.
_ENGINE = None
_ENGINE_URL: str | None = None


def _dsn() -> str:
    # Same conjourney instance sql-writer writes drive_audits into. Prefer the
    # action-outcome DSN already wired into cortex-exec; fall back to the
    # endogenous SQL reader DSN if operators only set that one.
    return (
        os.getenv("ORION_ACTION_OUTCOME_DB_URL", "").strip()
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "").strip()
    )


def _timeout_sec() -> float:
    raw = os.getenv("CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC", "0.4").strip()
    try:
        return max(0.01, float(raw))
    except ValueError:
        return 0.4


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(url, pool_pre_ping=True)
        _ENGINE_URL = url
    return _ENGINE


def _coerce_jsonb(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def _row_to_stance_drive_state(row: dict[str, Any]) -> dict[str, Any] | None:
    active_drives = _coerce_jsonb(row.get("active_drives"))
    dominant_drive = row.get("dominant_drive")
    summary = row.get("summary")
    # Quiet tick: schema-valid empty content → treat as no signal (Mind contract).
    if not dominant_drive and not summary and not (isinstance(active_drives, list) and active_drives):
        return None

    pressures_raw = _coerce_jsonb(row.get("drive_pressures"))
    pressures: dict[str, float] = {}
    if isinstance(pressures_raw, dict):
        for key, val in pressures_raw.items():
            try:
                pressures[str(key)] = float(val)
            except (TypeError, ValueError):
                continue

    # drive_audits stores active_drives (list), not the full activations bool map
    # from DriveStateV1. Approximate activations as True for listed drives —
    # faithful enough for stance/autonomy_slice; avoid a second SoR table.
    activations: dict[str, bool] = {}
    if isinstance(active_drives, list):
        for item in active_drives:
            name = str(item or "").strip()
            if name:
                activations[name] = True

    tension_raw = _coerce_jsonb(row.get("tension_kinds"))
    tension_kinds: list[str] = []
    if isinstance(tension_raw, list):
        tension_kinds = [str(t) for t in tension_raw if str(t).strip()]

    return {
        "pressures": pressures,
        "activations": activations,
        "dominant_drive": str(dominant_drive).strip() if dominant_drive else None,
        "summary": str(summary).strip() if summary else None,
        "tension_kinds": tension_kinds,
    }


def _query_latest_drive_audit_row_sync() -> dict[str, Any] | None:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("chat_stance_drive_state_dsn_unset")
    with engine.begin() as conn:
        conn.execute(
            text(f"SET LOCAL statement_timeout = '{_DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS}ms'")
        )
        row = conn.execute(text(DRIVE_AUDITS_LATEST_QUERY_FOR_STANCE)).mappings().first()
    return dict(row) if row is not None else None


async def fetch_drive_state_for_chat_stance(
    correlation_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bounded fail-open fetch of latest drive_audits row mapped to chat_drive_state."""
    timeout_sec = _timeout_sec()
    diagnostics: dict[str, Any] = {
        "correlation_id": correlation_id,
        "timeout_sec": timeout_sec,
        "ok": False,
        "elapsed_ms": 0,
        "timed_out": False,
        "exception_type": None,
        "reason": "start",
        "source": "drive_audits",
    }
    if not _dsn():
        diagnostics.update({"ok": True, "reason": "dsn_unset"})
        return None, diagnostics

    t0 = time.perf_counter()
    try:
        # wait_for cannot cancel the worker thread; on asyncio timeout the
        # thread keeps the pooled connection until SET LOCAL statement_timeout
        # kills the query (≤ _DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS).
        row = await asyncio.wait_for(
            asyncio.to_thread(_query_latest_drive_audit_row_sync),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "timed_out": True,
                "exception_type": type(exc).__name__,
                "reason": "timeout",
            }
        )
        logger.warning(
            "chat_stance_drive_state_fetch_timeout correlation_id=%s elapsed_ms=%s timeout_sec=%s",
            correlation_id,
            elapsed_ms,
            timeout_sec,
        )
        return None, diagnostics
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "exception_type": type(exc).__name__,
                "reason": "exception",
                "degradation_reason": str(exc),
            }
        )
        logger.warning(
            "chat_stance_drive_state_fetch_failed correlation_id=%s elapsed_ms=%s exc_type=%s err=%s",
            correlation_id,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
        return None, diagnostics

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    diagnostics["elapsed_ms"] = elapsed_ms
    if row is None:
        diagnostics.update({"ok": True, "reason": "no_rows"})
        return None, diagnostics

    mapped = _row_to_stance_drive_state(row)
    if mapped is None:
        diagnostics.update({"ok": True, "reason": "no_meaningful_content"})
        return None, diagnostics

    diagnostics.update({"ok": True, "reason": "success"})
    logger.info(
        "chat_stance_drive_state_fetch_result correlation_id=%s ok=true elapsed_ms=%s dominant_drive=%s",
        correlation_id,
        elapsed_ms,
        mapped.get("dominant_drive"),
    )
    return mapped, diagnostics


def reset_drive_state_postgres_engine_for_tests() -> None:
    """Test helper: drop cached engine between DSN monkeypatches."""
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None
