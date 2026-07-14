"""Reverie thought persistence (Phase A store).

Best-effort writer for `SpontaneousThoughtV1` into `substrate_reverie_thought`
(migration `manual_migration_substrate_reverie_thought.sql`). Backs the hub
`_reverie_section` panel. Uses a direct sqlalchemy DSN (see `_database_url`) —
never the heavy `orion.substrate` package this thin service does not ship.

Discipline: persistence is best-effort. A DB failure degrades to a logged miss
(returns False) and never breaks the reverie tick. Idempotent on `thought_id`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("orion-thought.store")

if TYPE_CHECKING:
    from orion.schemas.reverie import SpontaneousThoughtV1

_engine = None


def _database_url() -> str:
    # Direct DSN — deliberately NOT via orion.substrate.felt_state_reader, whose
    # package __init__ drags the full graph engine (requests etc.) this thin
    # service does not ship. Writes land where the hub panel reads (conjourney).
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def persist_reverie_thought(thought: "SpontaneousThoughtV1") -> bool:
    """Insert one spontaneous thought. Returns True on write, False on any miss.

    Never raises — a persistence failure must not break the tick.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_thought
                        (thought_id, correlation_id, created_at, salience,
                         interpretation, thought_json)
                    VALUES
                        (:thought_id, :correlation_id, :created_at, :salience,
                         :interpretation, CAST(:thought_json AS jsonb))
                    ON CONFLICT (thought_id) DO NOTHING
                    """
                ),
                {
                    "thought_id": thought.thought_id,
                    "correlation_id": thought.correlation_id,
                    "created_at": thought.created_at,
                    "salience": float(thought.salience),
                    "interpretation": thought.interpretation,
                    "thought_json": json.dumps(thought.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("reverie thought persist failed id=%s err=%s", thought.thought_id, exc)
        return False


def persist_salience_trace(trace) -> bool:
    """Persist one salience trace row. Never raises; idempotent on trace_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO attention_salience_trace
                        (trace_id, loop_id, theme_key, description, correlation_id, salience,
                         weights_version, scope, features, created_at)
                    VALUES
                        (:trace_id, :loop_id, :theme_key, :description, :correlation_id, :salience,
                         :weights_version, :scope, CAST(:features AS jsonb), :created_at)
                    ON CONFLICT (trace_id) DO NOTHING
                    """
                ),
                {
                    "trace_id": trace.trace_id,
                    "loop_id": trace.loop_id,
                    "theme_key": trace.theme_key,
                    "description": trace.description,
                    "correlation_id": trace.correlation_id,
                    "salience": float(trace.salience),
                    "weights_version": trace.weights_version,
                    "scope": trace.scope,
                    "features": json.dumps(trace.features),
                    "created_at": trace.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("salience trace persist failed id=%s err=%s", trace.trace_id, exc)
        return False


def persist_reverie_chain(chain) -> bool:
    """Insert one reverie chain readout. Never raises; idempotent on chain_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_chain
                        (chain_id, created_at, theme_key, terminal_reason,
                         ema_salience, committed_proposal_id, chain_json)
                    VALUES
                        (:chain_id, :created_at, :theme_key, :terminal_reason,
                         :ema_salience, :committed_proposal_id, CAST(:chain_json AS jsonb))
                    ON CONFLICT (chain_id) DO NOTHING
                    """
                ),
                {
                    "chain_id": chain.chain_id,
                    "created_at": chain.created_at,
                    "theme_key": chain.theme_key,
                    "terminal_reason": chain.terminal_reason,
                    "ema_salience": float(chain.ema_salience),
                    "committed_proposal_id": chain.committed_proposal_id,
                    "chain_json": json.dumps(chain.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("reverie chain persist failed id=%s err=%s", chain.chain_id, exc)
        return False


def persist_compaction_request(request) -> bool:
    """Enqueue one compaction request (Phase E). Never raises; idempotent."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dream_compaction_request_queue
                        (request_id, theme, op_hint, reason, origin_chain_id,
                         created_at, request_json)
                    VALUES
                        (:request_id, :theme, :op_hint, :reason, :origin_chain_id,
                         :created_at, CAST(:request_json AS jsonb))
                    ON CONFLICT (request_id) DO NOTHING
                    """
                ),
                {
                    "request_id": request.request_id,
                    "theme": request.theme,
                    "op_hint": request.op_hint,
                    "reason": request.reason,
                    "origin_chain_id": request.origin_chain_id,
                    "created_at": request.created_at,
                    "request_json": json.dumps(request.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("compaction request persist failed id=%s err=%s", request.request_id, exc)
        return False


def load_recent_chain_theme_events(limit: int) -> list[tuple[str, object]]:
    """Recent (theme_key, created_at) chain rows for the resonance detector.

    Read-only, best-effort — returns [] on any miss so the tripwire degrades to
    "no evidence" rather than raising. Skips null/unknown themes."""
    limit = max(0, int(limit))
    if limit == 0:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT theme_key, created_at FROM substrate_reverie_chain "
                        "WHERE theme_key IS NOT NULL AND theme_key <> 'unknown' "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [(str(r["theme_key"]), r["created_at"]) for r in rows if r.get("created_at")]
    except Exception as exc:
        logger.debug("resonance chain-event load failed: %s", exc)
        return []


def persist_resonance_alert(alert) -> bool:
    """Persist one resonance alert. Never raises; idempotent on alert_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_resonance_alert
                        (alert_id, theme_key, violation_count, refractory_sec,
                         min_gap_sec, occurrences, created_at, alert_json)
                    VALUES
                        (:alert_id, :theme_key, :violation_count, :refractory_sec,
                         :min_gap_sec, :occurrences, :created_at, CAST(:alert_json AS jsonb))
                    ON CONFLICT (alert_id) DO NOTHING
                    """
                ),
                {
                    "alert_id": alert.alert_id,
                    "theme_key": alert.theme_key,
                    "violation_count": int(alert.violation_count),
                    "refractory_sec": float(alert.refractory_sec),
                    "min_gap_sec": float(alert.min_gap_sec),
                    "occurrences": int(alert.occurrences),
                    "created_at": alert.created_at,
                    "alert_json": json.dumps(alert.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("resonance alert persist failed id=%s err=%s", alert.alert_id, exc)
        return False


def load_recent_resonance_alerts(theme_key: str, limit: int = 2) -> list[dict]:
    """Most recent persisted resonance alerts for one theme, newest first.

    Read-only, best-effort — returns [] on any miss. Used by the health monitor
    to compare violation_count across the last 2 samples (is this theme's
    resonance getting worse, or is a new alert just re-reporting the same
    historical burst still inside the detector's lookback window?).
    """
    limit = max(0, int(limit))
    if limit == 0 or not theme_key:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT theme_key, violation_count, refractory_sec, min_gap_sec, "
                        "occurrences, created_at FROM substrate_reverie_resonance_alert "
                        "WHERE theme_key = :theme_key "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"theme_key": theme_key, "limit": limit},
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("resonance alert history load failed theme=%s err=%s", theme_key, exc)
        return []


def reverie_refractory_is_suppressed(theme_key: str, now) -> bool:
    """True if the theme is currently suppressed. Best-effort (False on error)."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT suppressed_until FROM substrate_reverie_refractory "
                        "WHERE theme_key = :k"
                    ),
                    {"k": theme_key},
                )
                .mappings()
                .first()
            )
        if not row:
            return False
        until = row.get("suppressed_until")
        return until is not None and until > now
    except Exception:
        return False


def load_recent_loop_outcomes(loop_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Most recent `attention_loop_outcome` verdict per loop_id (orion-hub table,
    written by a human's Resolve/Dismiss action; read directly, no orion-hub import).

    Read-only, best-effort — returns {} on any miss or empty input, so a lookup
    failure never breaks a reverie tick. Keyed by the *bare* loop id, matching
    `attention_loops_store.suppress_loop`'s write format (see `chain.theme_key_for`
    for the sibling refractory-key fix this mirrors).

    Returns {loop_id: {"verdict": str, "note": str, "age_days": int}} — age is
    computed here (deterministic code), not left for the prompt/LLM to infer from
    a raw timestamp.
    """
    ids = [str(i) for i in (loop_ids or []) if i]
    if not ids:
        return {}
    try:
        from sqlalchemy import bindparam, text

        engine = _get_engine()
        stmt = text(
            """
            SELECT DISTINCT ON (loop_id) loop_id, verdict, note, created_at
            FROM attention_loop_outcome
            WHERE loop_id IN :ids
            ORDER BY loop_id, created_at DESC
            """
        ).bindparams(bindparam("ids", expanding=True))
        with engine.connect() as conn:
            rows = conn.execute(stmt, {"ids": ids}).mappings().all()
    except Exception as exc:
        logger.debug("loop outcome load failed ids=%s err=%s", ids, exc)
        return {}

    now = datetime.now(timezone.utc)
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        created = r.get("created_at")
        age_days = None
        if isinstance(created, datetime):
            c = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            age_days = max(0, int((now - c).total_seconds() // 86400))
        out[str(r["loop_id"])] = {
            "verdict": str(r.get("verdict") or ""),
            "note": str(r.get("note") or ""),
            "age_days": age_days,
        }
    return out


def reverie_refractory_suppress(theme_key: str, until) -> bool:
    """Upsert a refractory suppression window. Never raises."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_refractory (theme_key, suppressed_until)
                    VALUES (:k, :until)
                    ON CONFLICT (theme_key)
                    DO UPDATE SET suppressed_until = EXCLUDED.suppressed_until,
                                  updated_at = now()
                    """
                ),
                {"k": theme_key, "until": until},
            )
        return True
    except Exception as exc:
        logger.warning("reverie refractory suppress failed theme=%s err=%s", theme_key, exc)
        return False
