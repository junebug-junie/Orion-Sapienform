"""Dream cycle v2 store. Reads four producer tables; writes only v2 cycle/observation tables.

Read-only on every source (orion_metacog, dream_compaction_request_queue,
substrate_reverie_resonance_alert, memory_crystallizations, chat_history_log).
Writes exactly CYCLE_WRITE_TABLES -- a test pins that no canonical memory
table ever appears in a write statement here.

Every loader retains the existing empty/None fallback and logs why; source and
cycle-clock errors also invalidate observation history: a missing table
(migration not applied yet) must read as "nothing to replay", never raise.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.dream_cycle import DreamCycleV1

from app.settings import settings

logger = logging.getLogger("orion-dream.cycle_store")

CYCLE_WRITE_TABLES = ("dream_cycle", "dream_replay_item", "dream_hypothesis", "dream_pressure_observation")

# One row per THING (`dedupe_key`), newest first, in [since, until). The key
# rule lives here once; app/replay.py row_key only namespaces it.
#   metacog          trigger_reason with ids/numbers -> '#': structured, written
#                    by the producer (`transport:rpc_timeout:<channel>`). Not
#                    `summary`: model prose, reworded per row for one event.
#                    No reason -> the row id (counts as its own thing).
#   compaction       the theme.   resonance  the theme_key.
#   crystallization  activation events (auto_activate / approve), NOT
#                    updated_at: recall rewrites updated_at on all ~100
#                    rendered crystallizations per retrieval (retriever.py
#                    _apply_recall_boost), which is activity, not new material.
# `timestamp` on orion_metacog is a producer-written ISO string, not a
# timestamptz. Cast only well-formed values so one bad row cannot fail the read
# (CASE, not AND: Postgres does not promise AND short-circuits).
METACOG_KEY_RE = r"[0-9a-f]{8,}|-?[0-9]+(\.[0-9]+)?"

SOURCE_QUERIES: dict[str, str] = {
    "metacog": f"""
        SELECT id, summary, severity, trigger_kind, tags, dedupe_key FROM (
          SELECT DISTINCT ON (dedupe_key) id, summary, severity, trigger_kind, tags, dedupe_key, ts FROM (
            SELECT id, summary, severity, trigger_kind, tags,
                   COALESCE(regexp_replace(NULLIF(trigger_reason, ''), '{METACOG_KEY_RE}', '#', 'g'),
                            'id:' || id) AS dedupe_key,
                   CASE WHEN timestamp ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}T'
                        THEN CAST(timestamp AS timestamptz) END AS ts
              FROM orion_metacog
             WHERE severity IN ('degraded', 'critical')
          ) m WHERE ts > :since AND ts < :until
          ORDER BY dedupe_key, (severity = 'critical') DESC, ts DESC
        ) d ORDER BY ts DESC LIMIT :limit
    """,
    "compaction_request": """
        SELECT request_id, theme, reason, dedupe_key FROM (
          SELECT DISTINCT ON (lower(theme)) request_id, theme, reason, lower(theme) AS dedupe_key, created_at
            FROM dream_compaction_request_queue
           WHERE created_at > :since AND created_at < :until
           ORDER BY lower(theme), created_at DESC
        ) d ORDER BY created_at DESC LIMIT :limit
    """,
    "resonance": """
        SELECT alert_id, theme_key, violation_count, dedupe_key FROM (
          SELECT DISTINCT ON (lower(theme_key)) alert_id, theme_key, violation_count,
                 lower(theme_key) AS dedupe_key, created_at
            FROM substrate_reverie_resonance_alert
           WHERE created_at > :since AND created_at < :until
           ORDER BY lower(theme_key), violation_count DESC, created_at DESC
        ) d ORDER BY created_at DESC LIMIT :limit
    """,
    "crystallization": """
        SELECT crystallization_id, subject, summary, salience, tags, dedupe_key FROM (
          SELECT DISTINCT ON (c.crystallization_id) c.crystallization_id, c.subject, c.summary,
                 c.salience, c.tags, c.crystallization_id::text AS dedupe_key, h.created_at
            FROM memory_crystallization_history h
            JOIN memory_crystallizations c USING (crystallization_id)
           WHERE h.op IN ('auto_activate', 'approve') AND c.status = 'active'
             AND h.created_at > :since AND h.created_at < :until
           ORDER BY c.crystallization_id, h.created_at DESC
        ) d ORDER BY created_at DESC LIMIT :limit
    """,
}

# chat_history_log.created_at is `timestamp without time zone` defaulted by the
# server's now(), so compare against LOCALTIMESTAMP on the same server clock.
IDLE_MINUTES_SQL = """
    SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - max(created_at))) / 60.0 AS idle
      FROM chat_history_log
"""

# Two different clocks, on purpose (review finding, 2026-09-25):
#   window start = the last NON-failed cycle's started_at. started_at, not
#     ended_at, so rows written while a cycle ran (minutes of LLM calls) land
#     in the next window instead of falling between them; non-failed, so a
#     gateway outage does not throw the backlog away.
#   attempt end  = the last cycle of ANY status. Gates the min interval, so an
#     outage retries after DREAM_MIN_INTERVAL_HOURS instead of every tick.
LAST_WINDOW_START_SQL = "SELECT max(started_at) AS at FROM dream_cycle WHERE status <> 'failed'"
LAST_ATTEMPT_END_SQL = "SELECT max(ended_at) AS at FROM dream_cycle"

_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)

_engine = None


class SourceRows(dict):
    """Existing row mapping plus source failures for observation validity only."""
    def __init__(self):
        super().__init__()
        self.read_errors = []


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(settings.POSTGRES_URI, pool_pre_ping=True)
    return _engine


def load_source_rows(
    since: datetime, limit_per_source: int, until: Optional[datetime] = None
) -> dict[str, list[dict[str, Any]]]:
    """source_kind -> one row per thing in [since, until). A failing source is
    empty and logged, not fatal."""
    from sqlalchemy import text

    out = SourceRows()
    engine = _get_engine()
    params = {"since": since, "until": until or _FAR_FUTURE, "limit": int(limit_per_source)}
    for kind, sql in SOURCE_QUERIES.items():
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
            out[kind] = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("dream_cycle source read failed kind=%s err=%s", kind, exc)
            out.read_errors.append(kind)
            out[kind] = []
    return out


_history_engine = None


def persist_pressure_observation(observation) -> bool:
    """Append-only, idempotent check history. A missing migration is loud, not fatal.

    A separate small pool bounds instrumentation connection/lock/statement waits.
    Retention only touches this table, 1,000 expired rows at most per check.
    """
    global _history_engine
    from sqlalchemy import create_engine, text

    try:
        if _history_engine is None:
            _history_engine = create_engine(settings.POSTGRES_URI, pool_size=1, max_overflow=0,
                pool_timeout=2, connect_args={"connect_timeout": 2})
        with _history_engine.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
            conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
            conn.execute(text("""
                INSERT INTO dream_pressure_observation (check_id, observed_at, observation_json)
                VALUES (:check_id, :observed_at, CAST(:payload AS jsonb))
                ON CONFLICT (check_id) DO NOTHING
            """), dict(check_id=observation.check_id, observed_at=observation.observed_at,
                       payload=observation.model_dump_json()))
        # Separate transaction: cleanup cannot roll back the new observation.
        try:
            with _history_engine.begin() as conn:
                conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
                conn.execute(text("""
                    DELETE FROM dream_pressure_observation WHERE check_id IN (
                        SELECT check_id FROM dream_pressure_observation
                        WHERE created_at < now() - interval '30 days'
                        ORDER BY created_at LIMIT 1000
                    )
                """))
        except Exception:
            logger.exception("dream_pressure_history_retention_failed")
        return True
    except Exception:
        logger.exception("dream_pressure_history_write_failed check_id=%s", observation.check_id)
        return False


def load_idle_minutes() -> Optional[float]:
    """Minutes since the last chat turn. None = unknown (treated as NOT idle)."""
    try:
        from sqlalchemy import text

        with _get_engine().connect() as conn:
            row = conn.execute(text(IDLE_MINUTES_SQL)).mappings().first()
        if row is None or row["idle"] is None:
            return None
        return float(row["idle"])
    except Exception as exc:
        logger.warning("dream_cycle idle read failed err=%s", exc)
        return None


def _load_at(sql: str, *, read_errors=None, source="cycle_clock") -> Optional[datetime]:
    try:
        from sqlalchemy import text

        with _get_engine().connect() as conn:
            row = conn.execute(text(sql)).mappings().first()
        return row["at"] if row else None
    except Exception as exc:
        if read_errors is not None:
            read_errors.append(source)
        logger.warning("dream_cycle last-cycle read failed err=%s", exc)
        return None


def load_last_window_start(*, read_errors=None) -> Optional[datetime]:
    return _load_at(LAST_WINDOW_START_SQL, read_errors=read_errors, source="last_window_start")


def load_last_attempt_end(*, read_errors=None) -> Optional[datetime]:
    return _load_at(LAST_ATTEMPT_END_SQL, read_errors=read_errors, source="last_attempt_end")


def persist_cycle(cycle: DreamCycleV1) -> bool:
    """One transaction: cycle row, replay items, hypotheses. Never raises."""
    try:
        from sqlalchemy import text

        with _get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dream_cycle
                        (cycle_id, trigger, status, started_at, ended_at, pressure,
                         replay_count, hypothesis_count, no_link_count, unparseable_count,
                         llm_failures, compaction_delta_id, cycle_json)
                    VALUES
                        (:cycle_id, :trigger, :status, :started_at, :ended_at, :pressure,
                         :replay_count, :hypothesis_count, :no_link_count, :unparseable_count,
                         :llm_failures, :compaction_delta_id, CAST(:cycle_json AS jsonb))
                    ON CONFLICT (cycle_id) DO NOTHING
                    """
                ),
                {
                    "cycle_id": cycle.cycle_id,
                    "trigger": cycle.trigger,
                    "status": cycle.status,
                    "started_at": cycle.started_at,
                    "ended_at": cycle.ended_at,
                    "pressure": cycle.pressure.pressure,
                    "replay_count": len(cycle.replay),
                    "hypothesis_count": len(cycle.hypotheses),
                    "no_link_count": cycle.no_link_count,
                    "unparseable_count": cycle.unparseable_count,
                    "llm_failures": cycle.llm_failures,
                    "compaction_delta_id": cycle.compaction_delta_id,
                    "cycle_json": json.dumps(cycle.model_dump(mode="json")),
                },
            )
            for rank, item in enumerate(cycle.replay):
                conn.execute(
                    text(
                        """
                        INSERT INTO dream_replay_item
                            (cycle_id, ref_id, rank, source_kind, weight, reason)
                        VALUES (:cycle_id, :ref_id, :rank, :source_kind, :weight, :reason)
                        ON CONFLICT (cycle_id, ref_id) DO NOTHING
                        """
                    ),
                    {
                        "cycle_id": cycle.cycle_id,
                        "ref_id": item.ref_id,
                        "rank": rank,
                        "source_kind": item.source_kind,
                        "weight": item.weight,
                        "reason": item.reason,
                    },
                )
            for h in cycle.hypotheses:
                conn.execute(
                    text(
                        """
                        INSERT INTO dream_hypothesis
                            (hypothesis_id, cycle_id, arm, claim, why, ref_a, ref_b,
                             created_at, expires_at)
                        VALUES (:hypothesis_id, :cycle_id, :arm, :claim, :why, :ref_a, :ref_b,
                                :created_at, :expires_at)
                        ON CONFLICT (hypothesis_id) DO NOTHING
                        """
                    ),
                    h.model_dump(),
                )
        return True
    except Exception as exc:
        logger.warning("dream_cycle persist failed id=%s err=%s", cycle.cycle_id, exc)
        return False
