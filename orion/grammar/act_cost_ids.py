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

The other direction (reading's energy, attention patch 3e): the hourly energy attribution
(orion/grammar/act_energy_fill.py ``attribute_host_hour``) splits each host-hour's watt-hours into
"metered" (a lease whose turn some act is charged for) and the rest. ``turns_charged_to_acts`` is
that test, and it is the inverse of ``cost_correlation_ids``: a turn is charged when an act's
correlation_id IS the turn, or when the turn is one of a read's turns and the read has an act
(``reading_turn_owners``). Both sides read the same turn ids, so a read's watt-hours are counted
once: in the read act's ``wh_by_host`` and in the hour's ``metered_wh``, never also in
``leased_unmetered_wh``. The read's own holds are energy-only ids (``reading_hold_ids``).

Known limits of that pairing (energy is lost, never counted twice): a read lease whose terminal
event lands after the read's end + 120 s, and a provisional fill never finalised inside the 96 h
window, are metered by the hour but charged to no act. With SQL_WRITER_ACT_ENERGY_COST_ENABLED off,
read turns are still metered (as every other act's are) but no act carries their Wh.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Sequence
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


# The other direction: which of these turn ids belong to a read that has an act.
_SEED_TRACE_OWNERS_SQL = text(
    """
    SELECT s.request_id::text, s.trace_id, s.stage2_trace_id
      FROM world_pulse_read_seed s
     WHERE s.request_id IS NOT NULL
       AND (s.trace_id = ANY(:corrs) OR s.stage2_trace_id = ANY(:corrs))
    """
)
# Every durable-runs hold granted in the window whose run is a reading turn (reading_durable_turn),
# with the run's own correlation id: enough to re-derive each turn id (durable_turn_correlation_id).
_READ_HOLD_OWNERS_SQL = text(
    """
    SELECT s.request_id::text, t.run_id, t.request_json->>'correlation_id', g.lease_id,
           g.detail->'grant'->>'generation'
      FROM gpu_pool_events g
      JOIN reading_durable_turn t ON t.run_id = substr(g.holder, 14)
      JOIN world_pulse_read_seed s ON s.seed_id = t.seed_id
     WHERE g.event = 'granted' AND g.holder LIKE 'durable-runs:%'
       AND g.generated_at >= :since AND g.generated_at <= :until
       AND s.request_id IS NOT NULL
    """
)
_READ_ACTS_SQL = text(
    "SELECT DISTINCT correlation_id FROM grammar_acts WHERE actor = :actor AND verb = 'look' "
    "AND correlation_id = ANY(:rids)"
)
# The lease's turn IS an act's correlation_id (every actor but reading).
_TURNS_WITH_ACT_SQL = text("SELECT DISTINCT correlation_id FROM grammar_acts WHERE correlation_id = ANY(:corrs)")


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


def reading_hold_ids(conn: Any, request_id: str) -> list[str]:
    """The turn ids a read's own pool HOLDS carry: each reading run's correlation id.

    Energy only (attention patch 3e review): a hold and the calls inside it are two leases on one
    card, and both meters split a card 1/n per lease, so the hold takes half of every call's watts
    and all of the card's above-idle draw between calls. That is the read's energy too. GPU
    slot-seconds leave the hold out (it and its calls take one slot)."""
    rid = _uuid_or_none(request_id)
    if rid is None:
        return []
    seed = conn.execute(_READ_SEED_SQL, {"rid": rid}).first()
    if seed is None:
        return []
    rows = conn.execute(_READ_TURNS_SQL, {"seed": seed[0]}).fetchall()
    return list(dict.fromkeys(str(r[1]) for r in rows if r[1]))


def cost_correlation_ids(conn: Any, *, actor: Optional[str], correlation_id: Optional[str],
                         until: datetime) -> list[str]:
    """The turn ids an act is charged for. One id (its own correlation_id) for every actor but reading."""
    if actor == READING_ACTOR:
        return reading_cost_ids(conn, str(correlation_id), until=until) if correlation_id else []
    return [str(correlation_id)] if correlation_id else []


def reading_turn_owners(conn: Any, corrs: Sequence[str], *, since: datetime, until: datetime) -> dict[str, str]:
    """turn id -> the read act's correlation_id (the seed's request_id), for every id in ``corrs``
    that is one of a read's turns, when that read has an act.

    The same turns the read act's energy is charged for: ``reading_cost_ids`` (the seed's two trace
    ids, and every turn re-derived from a reading run's pool hold) plus ``reading_hold_ids`` (the
    holds themselves). ``since``/``until`` bound the hold grants scanned: a turn's calls
    run inside its hold, so the window must start one longest hold before the leases looked at."""
    wanted = {str(c) for c in corrs if c}
    if not wanted:
        return {}
    owners: dict[str, str] = {}
    for rid, trace_id, stage2_trace_id in conn.execute(_SEED_TRACE_OWNERS_SQL, {"corrs": sorted(wanted)}).fetchall():
        for tid in (trace_id, stage2_trace_id):
            if tid and str(tid) in wanted:
                owners.setdefault(str(tid), str(rid))
    for rid, run_id, run_corr, lease_id, generation in conn.execute(
            _READ_HOLD_OWNERS_SQL, {"since": since, "until": until}).fetchall():
        if not (run_corr and lease_id) or generation in (None, ""):
            continue
        tid = durable_turn_correlation_id(str(run_id), str(run_corr), str(lease_id), generation)
        if tid in wanted:
            owners.setdefault(tid, str(rid))
        if str(run_corr) in wanted:  # the hold itself (reading_hold_ids): energy only
            owners.setdefault(str(run_corr), str(rid))
    if not owners:
        return {}
    with_act = {str(r[0]) for r in conn.execute(_READ_ACTS_SQL, {
        "actor": READING_ACTOR, "rids": sorted(set(owners.values()))}).fetchall()}
    return {tid: rid for tid, rid in owners.items() if rid in with_act}


def turns_with_own_act(conn: Any, corrs: Sequence[str]) -> set[str]:
    """Which of these turn ids are some act's own correlation_id (the pre-3e "metered" rule)."""
    wanted = sorted({str(c) for c in corrs if c})
    if not wanted:
        return set()
    return {str(r[0]) for r in conn.execute(_TURNS_WITH_ACT_SQL, {"corrs": wanted}).fetchall()}


def turns_charged_to_acts(conn: Any, corrs: Sequence[str], *, since: datetime, until: datetime) -> set[str]:
    """Which of these lease turn ids some act is charged for: the hourly attribution's "metered"."""
    wanted = sorted({str(c) for c in corrs if c})
    if not wanted:
        return set()
    direct = turns_with_own_act(conn, wanted)
    rest = [c for c in wanted if c not in direct]
    return direct | set(reading_turn_owners(conn, rest, since=since, until=until))


__all__ = ["LATE_COST_ACTORS", "READING_ACTOR", "cost_correlation_ids", "late_cost_actor", "reading_cost_ids",
           "reading_hold_ids",
           "reading_turn_owners", "turns_charged_to_acts", "turns_with_own_act"]
