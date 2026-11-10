"""Which GPU pool turns an act is charged for (attention patch 3a, "reading's real cost").

The act-cost fill (orion/grammar/act_ledger.py) charges an act for the pool leases whose
``turn_correlation_id`` is one of the act's cost ids. For almost every act that is one id: its
own ``correlation_id``. Reading is the exception, and it was charged nothing:

* the read act's ``correlation_id`` is the seed's ``request_id`` (services/orion-hub/scripts/act_emit.py
  ``_seed_correlation``), and no lease ever carries it (live 2026-11-10: 0 of 1,117 read acts in 30
  days matched a lease);
* the read's model calls run inside durable-run turns, and every lease carries the TURN's id
  (``durable_turn_correlation_id``: run id, the run's correlation id, the pool hold, its generation).
  Hub records the last good turn of each stage as the seed's ``trace_id`` and ``stage2_trace_id``;
  a failed or retried attempt's turn is recorded nowhere but in the run (reading_durable_turn) and
  its pool hold (gpu_pool_events ``granted`` by ``durable-runs:<run_id>``).

So a read's cost ids are the seed's two trace ids plus every turn re-derived from its runs' holds
(live 30 days: the two trace ids reach 928 of 1,117 reads, 251.5 slot-hours; all turns reach
1,117 of 1,117, 335.4 slot-hours). Known limit: a turn replayed after a durable-runs restart
(``turn_fence``) is not re-derivable, and is only counted when it was the stage's last good turn.

A read's stage 2 runs AFTER its act closes (the act grades stage 1), so reading's cost is filled
late: ``late_cost_actor`` tells the close-time fill to leave it, and act_ledger.py
``refill_late_costs`` fills it once the read's stage 2 has ended.

The pool hold itself is not charged: the hold and the calls it wraps take one slot, and only the
calls carry the turn id (the same lease set as the spec's 131.6 slot-hours).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text

from orion.durable_runs.turn_identity import durable_turn_correlation_id

READING_ACTOR = "hub.world_pulse_read"
# Actors whose act closes before all of its work is done: the close-time fill leaves their GPU and
# energy cost, and refill_late_costs fills it once the work has ended.
LATE_COST_ACTORS = frozenset({READING_ACTOR})

_READ_SEED_SQL = text(
    "SELECT seed_id, trace_id, stage2_trace_id, created_at FROM world_pulse_read_seed "
    "WHERE request_id = CAST(:rid AS uuid)"
)
_READ_TURNS_SQL = text(
    "SELECT run_id, request_json->>'correlation_id' AS corr, created_at FROM reading_durable_turn "
    "WHERE seed_id = :seed ORDER BY stage, attempt"
)
# Every grant of the runs' pool holds (a hold re-granted after a restart has a new generation).
# generated_at bounds the scan to the read's own span (gpu_pool_events is indexed on it, not holder).
_HOLD_GRANTS_SQL = text(
    """
    SELECT DISTINCT holder, lease_id, detail->'grant'->>'generation' AS generation
      FROM gpu_pool_events
     WHERE event = 'granted' AND holder = ANY(:holders)
       AND generated_at >= :since AND generated_at <= :until
    """
)


def late_cost_actor(actor: Optional[str]) -> bool:
    return actor in LATE_COST_ACTORS


def _uuid_or_none(value: Optional[str]) -> Optional[str]:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        return None


def reading_cost_ids(conn: Any, request_id: str, *, until: datetime) -> list[str]:
    """Every turn id a read's leases carry: the seed's trace ids, then each run's re-derived turns.

    ``until``: the latest moment one of the read's holds can have been granted (the read's end)."""
    rid = _uuid_or_none(request_id)
    if rid is None:
        return []
    seed = conn.execute(_READ_SEED_SQL, {"rid": rid}).first()
    if seed is None:
        return []
    seed_id, trace_id, stage2_trace_id, seed_created = seed[0], seed[1], seed[2], seed[3]
    ids: list[str] = [str(x) for x in (trace_id, stage2_trace_id) if x]
    runs = {str(r[0]): str(r[1]) for r in conn.execute(_READ_TURNS_SQL, {"seed": seed_id}).fetchall() if r[1]}
    if runs:
        since = seed_created - timedelta(minutes=1) if seed_created is not None else until - timedelta(days=2)
        grants = conn.execute(_HOLD_GRANTS_SQL, {"holders": [f"durable-runs:{r}" for r in runs],
                                                 "since": since, "until": until}).fetchall()
        for holder, lease_id, generation in grants:
            run_id = str(holder).split(":", 1)[1]
            if lease_id and generation not in (None, "") and run_id in runs:
                ids.append(durable_turn_correlation_id(run_id, runs[run_id], str(lease_id), generation))
    return list(dict.fromkeys(ids))


def cost_correlation_ids(conn: Any, *, actor: Optional[str], correlation_id: Optional[str],
                         until: datetime) -> list[str]:
    """The turn ids an act is charged for. One id (its own correlation_id) for every actor but reading."""
    if actor == READING_ACTOR:
        return reading_cost_ids(conn, str(correlation_id), until=until) if correlation_id else []
    return [str(correlation_id)] if correlation_id else []


__all__ = ["LATE_COST_ACTORS", "READING_ACTOR", "cost_correlation_ids", "late_cost_actor", "reading_cost_ids"]
