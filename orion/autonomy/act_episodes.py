"""Act episode ledger: one row per allocator decision to run, or hold back, a proven-effect act.

Thinking rung 3, stage 3 option B (Juniper 2026-10-21). The allocator may admit an act that is
proven to help (orion/autonomy/act_candidates.py ``proven_effect``). That claim is checked against a
randomized control arm: a share of those admissions (ORION_DISPATCH_HOLDBACK_FRACTION) is held back
and recorded here as ``control``; the rest are recorded as ``treated`` BEFORE the dispatch is sent
(precommit). The control arm's rate becomes the bar the act must beat once it has enough rows.

Separate from substrate_world_action_episodes on purpose: that ledger's scorer
(orion-feedback-runtime) scores every due row as a cabinet shed, so an act row there would be
mis-scored as one.

Outcome of a control row: for a producer-reported signal (``artifact_produced``) a held-back act
produced nothing, so its outcome is 0 by construction and is recorded at decision time. For a
world-reported signal (``juniper_replied``) the control outcome needs a resolver that does not
exist yet, so it stays NULL and does not count toward the control rate.

Writer: orion-execution-dispatch-runtime. Created lazily (short lock_timeout); the same DDL is in
services/orion-sql-db/manual_migration_act_episodes_v1.sql.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

TABLE = "substrate_act_episodes"
ARMS = ("treated", "control")
STRUCTURAL_ZERO_SIGNALS = frozenset({"artifact_produced"})

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    episode_id text PRIMARY KEY,
    act_label text NOT NULL,
    verb text NOT NULL,
    actor text NOT NULL,
    signal_id text NOT NULL,
    dispatch_kind text NOT NULL,
    target_id text NOT NULL,
    arm text NOT NULL,
    path text NOT NULL,
    decided_at timestamptz NOT NULL,
    horizon_at timestamptz NOT NULL,
    dispatch_frame_id text,
    expected jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    outcome_value double precision,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {TABLE}_act_idx ON {TABLE} (act_label, arm, decided_at DESC);
CREATE INDEX IF NOT EXISTS {TABLE}_horizon_idx ON {TABLE} (actor, horizon_at DESC)
"""


def ensure_table(conn, *, lock_timeout_ms: int = 3000) -> None:
    from sqlalchemy import text

    conn.execute(text(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'"))
    for stmt in (s.strip() for s in DDL.split(";")):
        if stmt:
            conn.execute(text(stmt))


def table_exists(conn) -> bool:
    from sqlalchemy import text

    return conn.execute(text("SELECT to_regclass(:t)"), {"t": TABLE}).scalar() is not None


def insert_decision(conn, row: dict[str, Any]) -> bool:
    """Precommit. Idempotent on episode_id (a replayed tick keeps its first arm). True if new."""
    from sqlalchemy import text

    if row.get("arm") not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    cols = ("episode_id", "act_label", "verb", "actor", "signal_id", "dispatch_kind", "target_id", "arm",
            "path", "decided_at", "horizon_at", "dispatch_frame_id", "expected", "outcome_value")
    vals = {c: row.get(c) for c in cols}
    vals["expected"] = json.dumps(vals["expected"] or {}, default=str)
    res = conn.execute(text(
        f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ("
        + ", ".join("CAST(:expected AS jsonb)" if c == "expected" else f":{c}" for c in cols)
        + ") ON CONFLICT (episode_id) DO NOTHING RETURNING episode_id"), vals).fetchone()
    return res is not None


def control_stats(conn, *, act_label: str) -> tuple[int, int]:
    """(resolved control rows, of which succeeded). (0, 0) while the table does not exist."""
    from sqlalchemy import text

    if not table_exists(conn):
        return 0, 0
    row = conn.execute(text(
        f"SELECT count(outcome_value), coalesce(sum(outcome_value), 0) FROM {TABLE} "
        "WHERE act_label = :a AND arm = 'control'"), {"a": act_label}).first()
    return int(row[0] or 0), int(row[1] or 0)


def in_flight(conn, *, actor: str, now: datetime) -> bool:
    """A decision of either arm still inside its act's horizon. A control row blocks re-admission
    for the same window a treated act would occupy: otherwise a held-back act is re-proposed
    seconds later and the "control" is only a delay."""
    from sqlalchemy import text

    if not table_exists(conn):
        return False
    return bool(conn.execute(text(
        f"SELECT EXISTS (SELECT 1 FROM {TABLE} WHERE actor = :a AND horizon_at > :now)"),
        {"a": actor, "now": now}).scalar())


__all__ = ["ARMS", "DDL", "STRUCTURAL_ZERO_SIGNALS", "TABLE", "control_stats", "ensure_table", "in_flight",
           "insert_decision", "table_exists"]
