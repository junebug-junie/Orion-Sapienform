"""World-action episode ledger: one row per decision to act OR to hold back (attend-to-act loop D4).

Why a ledger and not just the existing tables: a world action's chain crosses seven tables and two
clocks (the dispatch tick and a 20-minute settle window), and its control arm never dispatches, so
it leaves no ``substrate_dispatch_results`` row at all. This row is the join key for the whole chain
and the PRECOMMIT: it is written, with the expected effect and the eligibility snapshot, BEFORE the
shed RPC is sent (closes the agency-episode audit's ``expectation_precommitted`` gap).

    broadcast log (broadcast_log_id, open_loop_id) -> proposal (proposal_id) -> policy decision
    (decision_id) -> dispatch frame (dispatch_frame_id, episode_id = dispatch_id) -> pool shed
    (shed_id) -> dispatch result (settlement_state) -> substrate_action_outcomes (outcome.outcome_row_id)
    -> attention_loop_outcome (loop_outcome_id) -> next broadcast tick

Writers: orion-execution-dispatch-runtime (decision + settlement), orion-feedback-runtime (scoring).
Readers: orion-proposal-runtime (in-flight check), the attend-act eval. Created lazily under a short
lock_timeout (never blocks a boot); the same DDL is in
services/orion-sql-db/manual_migration_world_action_episodes_v1.sql.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("orion.autonomy.world_episodes")

TABLE = "substrate_world_action_episodes"
ARMS = ("treated", "control")
# A decided episode older than this with no score is not "in flight" any more: the settle path has
# an orphan rule well inside it (TTL + 300 s), so this only bounds a dead scorer.
IN_FLIGHT_HORIZON_SEC = 6 * 3600.0

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    episode_id text PRIMARY KEY,
    template text NOT NULL,
    dispatch_kind text NOT NULL,
    target_id text NOT NULL,
    arm text NOT NULL,
    decided_at timestamptz NOT NULL,
    open_loop_id text,
    broadcast_log_id text,
    node_id text,
    proposal_id text,
    decision_id text,
    dispatch_frame_id text,
    eligibility jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    expected_effect jsonb,
    shed_id text,
    settlement_state text,
    settlement jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    scoring_due_at timestamptz NOT NULL,
    scored_at timestamptz,
    outcome jsonb,
    loop_outcome_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {TABLE}_unscored_idx ON {TABLE} (scoring_due_at) WHERE scored_at IS NULL;
CREATE INDEX IF NOT EXISTS {TABLE}_decided_idx ON {TABLE} (decided_at DESC)
"""


def ensure_table(conn, *, lock_timeout_ms: int = 3000) -> None:
    """``conn``: SQLAlchemy connection inside a transaction. Raises on failure (callers fail closed)."""
    from sqlalchemy import text

    conn.execute(text(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'"))
    for stmt in (s.strip() for s in DDL.split(";")):
        if stmt:
            conn.execute(text(stmt))


def insert_decision(conn, row: dict[str, Any]) -> bool:
    """Precommit. Idempotent on episode_id (a replayed tick keeps the first decision). True if new."""
    from sqlalchemy import text

    cols = ("episode_id", "template", "dispatch_kind", "target_id", "arm", "decided_at", "open_loop_id",
            "broadcast_log_id", "node_id", "proposal_id", "decision_id", "dispatch_frame_id", "eligibility",
            "expected_effect", "settlement", "scoring_due_at")
    vals = {c: row.get(c) for c in cols}
    # Both arms carry the clock they are scored on (ttl_sec), so a TTL change cannot split the arms.
    vals["settlement"] = vals["settlement"] or {}
    for c in ("eligibility", "expected_effect", "settlement"):
        vals[c] = None if vals[c] is None else json.dumps(vals[c], default=str)
    if row.get("arm") not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    res = conn.execute(text(
        f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ("
        + ", ".join(f"CAST(:{c} AS jsonb)" if c in ("eligibility", "expected_effect", "settlement") else f":{c}"
                    for c in cols)
        + ") ON CONFLICT (episode_id) DO NOTHING RETURNING episode_id"), vals).fetchone()
    return res is not None


def record_settlement(conn, *, episode_id: str, shed_id: str | None, state: str, settlement: dict[str, Any]) -> None:
    from sqlalchemy import text

    conn.execute(text(
        f"UPDATE {TABLE} SET shed_id = COALESCE(:shed_id, shed_id), settlement_state = :state, "
        "settlement = settlement || CAST(:settlement AS jsonb), updated_at = now() WHERE episode_id = :episode_id"),
        {"episode_id": episode_id, "shed_id": shed_id, "state": state,
         "settlement": json.dumps(settlement, default=str)})


def record_score(conn, *, episode_id: str, outcome: dict[str, Any], loop_outcome_id: str | None,
                 scored_at: datetime) -> bool:
    """Write the score once. False when the row was already scored (idempotent)."""
    from sqlalchemy import text

    res = conn.execute(text(
        f"UPDATE {TABLE} SET outcome = CAST(:outcome AS jsonb), loop_outcome_id = :loop, scored_at = :at, "
        "updated_at = now() WHERE episode_id = :episode_id AND scored_at IS NULL RETURNING episode_id"),
        {"episode_id": episode_id, "outcome": json.dumps(outcome, default=str), "loop": loop_outcome_id,
         "at": scored_at}).fetchone()
    return res is not None


def arm_of(conn, episode_id: str) -> str | None:
    from sqlalchemy import text

    value = conn.execute(text(f"SELECT arm FROM {TABLE} WHERE episode_id = :e"), {"e": episode_id}).scalar()
    return None if value is None else str(value)


def recent_treated(conn, *, template: str, since: datetime) -> list[dict[str, Any]]:
    """Treated episodes decided since ``since`` (for the arm-symmetric gap / daily-cap check)."""
    from sqlalchemy import text

    rows = conn.execute(text(
        f"SELECT episode_id, decided_at, settlement_state, settlement FROM {TABLE} "
        "WHERE template = :t AND arm = 'treated' AND decided_at >= :since ORDER BY decided_at"),
        {"t": template, "since": since}).mappings().fetchall()
    return [dict(r) for r in rows]


def in_flight(conn, *, template: str, now: datetime) -> list[str]:
    """Episodes of this template decided but not yet scored (either arm). One in flight blocks the
    next decision: arms never overlap, and a shed's tail never sits in the next 'before' reading."""
    from sqlalchemy import text

    rows = conn.execute(text(
        f"SELECT episode_id FROM {TABLE} WHERE template = :t AND scored_at IS NULL AND decided_at > :since "
        "ORDER BY decided_at DESC LIMIT 10"),
        {"t": template, "since": now - timedelta(seconds=IN_FLIGHT_HORIZON_SEC)}).fetchall()
    return [str(r[0]) for r in rows]


def due_for_scoring(conn, *, now: datetime, limit: int = 20) -> list[dict[str, Any]]:
    from sqlalchemy import text

    rows = conn.execute(text(
        f"SELECT * FROM {TABLE} WHERE scored_at IS NULL AND scoring_due_at <= :now "
        "ORDER BY scoring_due_at LIMIT :limit"), {"now": now, "limit": limit}).mappings().fetchall()
    return [dict(r) for r in rows]


def unsettled_treated(conn, *, limit: int = 20) -> list[dict[str, Any]]:
    from sqlalchemy import text

    rows = conn.execute(text(
        f"SELECT * FROM {TABLE} WHERE arm = 'treated' AND (settlement_state IS NULL OR settlement_state = 'active') "
        "AND decided_at > now() - interval '6 hours' ORDER BY decided_at LIMIT :limit"),
        {"limit": limit}).mappings().fetchall()
    return [dict(r) for r in rows]


__all__ = ["ARMS", "DDL", "IN_FLIGHT_HORIZON_SEC", "TABLE", "due_for_scoring", "ensure_table", "in_flight",
           "insert_decision", "record_score", "record_settlement", "unsettled_treated"]
