"""Act episode ledger: one row per allocator decision to run, or hold back, a proven-effect act.

Thinking rung 3, stage 3 option B (Juniper 2026-10-21). The allocator may admit an act that is
proven to help (orion/autonomy/act_candidates.py ``proven_effect``). That claim is checked against a
randomized control arm: a share of those admissions (ORION_DISPATCH_HOLDBACK_FRACTION) is held back
and recorded here as ``control``; the rest are recorded as ``treated`` BEFORE the dispatch is sent
(precommit). The control arm's rate becomes the bar the act must beat once it has enough rows.

What this is NOT: a randomized treated-vs-control contrast. Treated rows are not resolved here; the
act's rate comes from its EffectPosterior, which pools every graded close of the act (allocator
dispatches and the scheduled-painting lane, and only runs that reached work). The control arm is
"dispatch withheld". The check is therefore "posterior lower bound vs the better of the act prior
and the control arm's rate" -- it can revoke a claim when not acting turns out as good as acting,
but for a producer-reported signal (control 0 by construction) the arm carries no information.

Separate from substrate_world_action_episodes on purpose: that ledger's scorer
(orion-feedback-runtime) scores every due row as a cabinet shed, so an act row there would be
mis-scored as one.

Which acts are held back at all (``holdback_rule``, rung 3 follow-up 2026-10-21): a control arm is
drawn only when the act's outcome can occur without the act. That is decided by where the act's
object comes from, not by a list of names:

- ``object_source="dispatch"``: the dispatch mints the object (a painting's run id is
  reverie_visual_run_id(dispatch_id), services/orion-thought/app/visual_steps.py). Withhold the
  dispatch and there is no object, so the outcome cannot occur: a control row would read 0 by
  construction and teach nothing. Not held back; the treated row records why.
- ``object_source="queue"``: the act picks an object that already exists and that another lane can
  act on (a reading pull picks a pending seed the self-scheduled lane would also read). A held-back
  decision still names that object, and whether it got the outcome anyway is observable. Held back.

Outcome of a row with an object (``object_ref`` = the act ledger key, e.g. ``read_seed:<seed_id>``):
resolved after ``horizon_at`` by ``resolve_due`` for BOTH arms the same way: 1 if a graded act of
the same verb/actor/signal with that key closed with outcome 1 between the decision and the horizon,
else 0. A decision the doer refused is voided (``void_reason``) and releases its in-flight hold.

Writer: orion-execution-dispatch-runtime. Created lazily (short lock_timeout); the same DDL is in
services/orion-sql-db/manual_migration_act_episodes_v1.sql.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

TABLE = "substrate_act_episodes"
ARMS = ("treated", "control")
# Where an act's object comes from. See the module docstring and holdback_rule.
OBJECT_SOURCES = ("dispatch", "queue")

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
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS object_ref text;
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS void_reason text;
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS resolved_at timestamptz;
CREATE INDEX IF NOT EXISTS {TABLE}_act_idx ON {TABLE} (act_label, arm, decided_at DESC);
CREATE INDEX IF NOT EXISTS {TABLE}_horizon_idx ON {TABLE} (actor, horizon_at DESC)
"""


def holdback_rule(object_source: str) -> tuple[bool, str]:
    """(draw a control arm?, the reason recorded on the decision). See the module docstring."""
    if object_source not in OBJECT_SOURCES:
        raise ValueError(f"object_source must be one of {OBJECT_SOURCES}")
    if object_source == "dispatch":
        return False, "holdback_exempt:outcome_requires_act:object_minted_by_dispatch"
    return True, "holdback_drawn:outcome_observable_without_act:object_preexists"


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
            "path", "decided_at", "horizon_at", "dispatch_frame_id", "expected", "outcome_value", "object_ref")
    vals = {c: row.get(c) for c in cols}
    vals["expected"] = json.dumps(vals["expected"] or {}, default=str)
    res = conn.execute(text(
        f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ("
        + ", ".join("CAST(:expected AS jsonb)" if c == "expected" else f":{c}" for c in cols)
        + ") ON CONFLICT (episode_id) DO NOTHING RETURNING episode_id"), vals).fetchone()
    return res is not None


def set_object(conn, episode_id: str, object_ref: str) -> None:
    """The object the doer actually chose for this decision (may differ from the precommitted one)."""
    from sqlalchemy import text

    conn.execute(text(f"UPDATE {TABLE} SET object_ref = :o WHERE episode_id = :e AND void_reason IS NULL"),
                 {"o": object_ref, "e": episode_id})


def void_decision(conn, episode_id: str, reason: str) -> None:
    """The doer refused (its gates run for both arms, before the arm matters), so this is not an
    episode in either arm: no outcome, and its in-flight hold ends now."""
    from sqlalchemy import text

    conn.execute(text(f"UPDATE {TABLE} SET void_reason = :r, horizon_at = decided_at, outcome_value = NULL "
                      "WHERE episode_id = :e AND outcome_value IS NULL"), {"r": reason[:200], "e": episode_id})


# Both arms, one rule: did a graded act of the same verb/actor/signal on this object close with
# outcome 1 after the decision and by the horizon? grammar_acts is the rung-2 act ledger
# (orion/grammar/act_ledger.py); test_before is the producer's own object key.
RESOLVE_DUE_SQL = f"""
UPDATE {TABLE} e
SET outcome_value = CASE WHEN EXISTS (
        SELECT 1 FROM grammar_acts a
        WHERE a.actor = e.actor AND a.verb = e.verb AND a.signal_id = e.signal_id
          AND a.test_before = e.object_ref
          AND a.outcome_value >= 1
          AND a.closed_at >= e.decided_at AND a.closed_at <= e.horizon_at) THEN 1.0 ELSE 0.0 END,
    resolved_at = :now
WHERE e.outcome_value IS NULL AND e.void_reason IS NULL AND e.object_ref IS NOT NULL
  AND e.horizon_at <= :now
RETURNING e.episode_id, e.arm, e.outcome_value
"""


def resolve_due(conn, *, now: datetime) -> list[tuple[str, str, float]]:
    """Resolve every due row that names an object. Rows with no object (an act whose object is
    minted by its own dispatch) are never resolved here."""
    from sqlalchemy import text

    if not table_exists(conn):
        return []
    rows = conn.execute(text(RESOLVE_DUE_SQL), {"now": now}).all()
    return [(str(r[0]), str(r[1]), float(r[2])) for r in rows]


def arm_of(conn, episode_id: str) -> str | None:
    from sqlalchemy import text

    if not table_exists(conn):
        return None
    value = conn.execute(text(f"SELECT arm FROM {TABLE} WHERE episode_id = :e"), {"e": episode_id}).scalar()
    return None if value is None else str(value)


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


__all__ = ["ARMS", "DDL", "OBJECT_SOURCES", "RESOLVE_DUE_SQL", "TABLE", "arm_of", "control_stats", "ensure_table",
           "holdback_rule", "in_flight", "insert_decision", "resolve_due", "set_object", "table_exists", "void_decision"]
